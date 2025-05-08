import os
import time
import torch
import random
import argparse
import numpy as np
import open_clip
import torch_dct as dct
from tqdm import tqdm
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.autograd import Variable
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import importlib
import logging
import torch.nn.functional as F
from torchvision.transforms.functional import InterpolationMode
from torch.utils.tensorboard import SummaryWriter

# Set the maximum decompressed size for the image
ImageFile.LOAD_TRUNCATED_IMAGES = True  # To prevent errors from incomplete files
Image.MAX_IMAGE_PIXELS = None  # Allow loading images without a size limit

module_name = "capsule_network"
module = importlib.import_module(module_name)
CapsNet = getattr(module, "CapsNet")
CapsuleLoss = getattr(module, "CapsuleLoss")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CustomDatasetWithCaptions(torch.utils.data.Dataset):
    def __init__(self, real_dir, fake_dir, transform=None):
        self.real_dir = real_dir
        self.fake_dir = fake_dir
        self.transform = transform
        self.real_images = [(os.path.join(real_dir, img), 0) for img in os.listdir(real_dir) if img.endswith(('.png', '.jpg', '.jpeg'))]
        self.fake_images = [(os.path.join(fake_dir, img), 1) for img in os.listdir(fake_dir) if img.endswith(('.png', '.jpg', '.jpeg'))]
        self.all_images = self.real_images + self.fake_images

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        img_path, label = self.all_images[idx]
        try:
            with Image.open(img_path) as image:
                # Ensure image is in RGB format
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Apply transformations (e.g., resizing)
                if self.transform:
                    image = self.transform(image)
                
                # Convert transformed image to grayscale
                grayscale_tensor = transforms.functional.rgb_to_grayscale(image)
                
                # Scale to [-1, 1]
                grayscale_tensor = (grayscale_tensor * 2) - 1
                
                # Apply 2D DCT directly using dct_2d
                DCT_transform = dct.dct_2d(grayscale_tensor, norm='ortho')

        except Exception as e:
            print(f"Error processing image: {img_path}\nException: {e}")
            raise

        # Prepare caption
        caption = os.path.splitext(os.path.basename(img_path))[0]
        cleaned_caption = caption.replace('_', ' ')

        return image, DCT_transform, label, cleaned_caption


# Validation transform without augmentation
val_transform = transforms.Compose([transforms.Resize((320, 320), interpolation=InterpolationMode.BICUBIC),
                                    transforms.CenterCrop(320),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                         std=[0.26862954, 0.26130258, 0.27577711])])


def evaluate(CLIP_model, CAPS_model, tokenizer, test_dataset, BATCH_SIZE, CLASSES, device):
   loss_fn = CapsuleLoss()
   test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
   
   test_loss = 0.0
   test_predictions, test_true_labels = [], []
   total_tp, total_fp, total_tn, total_fn = 0, 0, 0, 0
   total_samples = 0
   
   with torch.no_grad():
       for step, (images, DCT_features, labels, captions) in enumerate(tqdm(test_loader)):
           batch_size = images.size(0)
           
           images = images.to(device)
           DCT_features = DCT_features.to(device)
           labels = labels.to(device)
           labels_onehot = torch.eye(CLASSES).to(device).index_select(dim=0, index=labels)
           captions_tokens = tokenizer(list(captions)).to(device)
           
           img_embedding = CLIP_model.encode_image(images).float()
           cap_embedding = CLIP_model.encode_text(captions_tokens)
           
           output = CAPS_model(img_embedding, cap_embedding, DCT_features, device=device)
           loss = loss_fn(output, labels_onehot)
           
           class_lengths = torch.sqrt((output ** 2).sum(dim=2))
           predictions = torch.max(class_lengths, dim=1)[1]
           
           test_loss += loss.item()
           total_samples += batch_size
           
           predictions = predictions.cpu().numpy()
           labels_np = labels.cpu().numpy()
           test_predictions.extend(predictions)
           test_true_labels.extend(labels_np)
           
           for pred, true_label in zip(predictions, labels_np):
               if pred == 1 and true_label == 1:
                   total_tp += 1
               elif pred == 1 and true_label == 0:
                   total_fp += 1
               elif pred == 0 and true_label == 0:
                   total_tn += 1
               elif pred == 0 and true_label == 1:
                   total_fn += 1
   
   avg_loss = test_loss / len(test_loader)
   precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
   recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
   f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
   accuracy = accuracy_score(test_true_labels, test_predictions)
   
   print("\nModel Performance:")
   print(f'Total test samples: {total_samples}')
   print(f'Average loss: {avg_loss:.4f}')
   print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, Accuracy: {accuracy:.4f}')
   print('Confusion Matrix (TP, FP, TN, FN):', total_tp, total_fp, total_tn, total_fn)
   print(f"Metrics (%): {precision*100:.2f} / {recall*100:.2f} / {f1*100:.2f} / {accuracy*100:.2f}")
    

def main():
    set_seed(24)
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_model", type=str, default="./MagicBrush_Trained_Models")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--image_root", type=str, default="./MagicBrush/BLIP_captioned_dataset/")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use (e.g., 0, 1, 2)")
    args = parser.parse_args()

    CLASSES = 2
    BATCH_SIZE = args.batch_size
    LOAD_MODEL = args.load_model
    IMAGE_ROOT = args.image_root
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')

    
    # Configuring model
    CLIP_model, _, _ = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup')
    tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup')
    
    CLIP_model.to(device)
    CLIP_model.eval()

    CAPS_Model = CapsNet().to(device)
    
    checkpoint = torch.load(LOAD_MODEL)
    CAPS_Model.load_state_dict(checkpoint['model_state_dict'])
    CAPS_Model.eval().to(device)

    # Paths to the directories containing source and target training images
    source_image_test = IMAGE_ROOT + "/test/source_images"
    target_image_test = IMAGE_ROOT + "/test/target_images"
    
    # Create the dataset
    test_dataset = CustomDatasetWithCaptions(real_dir=source_image_test, fake_dir=target_image_test, transform=val_transform)

    evaluate(CLIP_model, CAPS_Model, tokenizer, test_dataset, BATCH_SIZE, CLASSES, device)

if __name__ == '__main__':
    main()

