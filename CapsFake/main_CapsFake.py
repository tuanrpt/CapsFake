import os
import time
import torch
import random
import argparse
import torch_dct as dct
import numpy as np
import open_clip
from PIL import Image, ImageFile, ImageFilter
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.autograd import Variable
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import importlib
import logging
import torch.nn.functional as F
from torchvision.transforms.functional import InterpolationMode
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import shutil
from io import BytesIO

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

# Set the maximum decompressed size for the image
ImageFile.LOAD_TRUNCATED_IMAGES = True  # To prevent errors from incomplete files
Image.MAX_IMAGE_PIXELS = None  # Allow loading images without a size limit


# Hyperparameters
patience = 10


class EarlyStopper:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.val_acc = -10000.0

    def early_stop(self, curr_acc):
        if curr_acc > self.val_acc:
            self.val_acc = curr_acc
            self.counter = 0
        elif curr_acc < (self.val_acc - self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

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


class AddGaussianNoise:
    def __init__(self, mean=0., std=0.01):
        self.mean = mean
        self.std = std
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

class JPEGCompression:
    def __init__(self, quality_range=(60, 100)):
        self.quality_range = quality_range
        
    def __call__(self, img):
        quality = random.randint(*self.quality_range)
        buffer = io.BytesIO()
        img.save(buffer, "JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer)

class ProbabilisticTransform:
    """Applies a transform with a given probability."""
    def __init__(self, transform, probability=0.1):
        self.transform = transform
        self.probability = probability
        
    def __call__(self, img):
        if random.random() < self.probability:
            return self.transform(img)
        return img

train_transform = transforms.Compose([transforms.RandomResizedCrop(size=(320, 320), scale=(0.9, 1.0), ratio=(0.75, 1.3333),
                                                                   interpolation=InterpolationMode.BICUBIC, antialias=True),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.CenterCrop(320),
                                      ProbabilisticTransform(JPEGCompression(quality_range=(60, 100)), probability=0.1),
                                      ProbabilisticTransform(transforms.GaussianBlur(kernel_size=3), probability=0.1),
                                      ProbabilisticTransform(transforms.ColorJitter(brightness=0.2, contrast=0.2), probability=0.1),
                                      transforms.ToTensor(),
                                      ProbabilisticTransform(AddGaussianNoise(std=0.01), probability=0.1),
                                      transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                           std=[0.26862954, 0.26130258, 0.27577711])])

# Validation transform without augmentation
val_transform = transforms.Compose([transforms.Resize((320, 320), interpolation=InterpolationMode.BICUBIC),
                                    transforms.CenterCrop(320),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                         std=[0.26862954, 0.26130258, 0.27577711])])



def train(CLIP_model, CAPS_model, tokenizer, train_ds, val_ds, EPOCHS, BATCH_SIZE, CLASSES, LEARNING_RATE, WEIGHT_DECAY, log_dir, device):
    # Create a unique log directory based on the current time
    log_id_display = "="*10 + "Log ID:" + log_dir + "="*10
    logging.info(log_id_display)

    # TensorBoard writer
    writer = SummaryWriter(log_dir=log_dir)

    # Save the script
    train_file = os.path.realpath(__file__)  # Path of the current script
    current_dir = os.path.dirname(os.path.realpath(__file__))
    model_file = os.path.join(current_dir, module_name + ".py")

    shutil.copy(train_file, os.path.join(log_dir, os.path.basename(train_file)))
    shutil.copy(model_file, os.path.join(log_dir, os.path.basename(model_file)))

    # Optimizer with layer-specific learning rates
    optimizer = torch.optim.AdamW([{'params': CAPS_model.DCT_emb.parameters(), 'lr': LEARNING_RATE},
                                   {'params': CAPS_model.primary_capsules.parameters(), 'lr': LEARNING_RATE * 0.2},
                                   {'params': CAPS_model.digit_capsules.parameters(), 'lr': LEARNING_RATE * 0.2}
                                  ], weight_decay=WEIGHT_DECAY, betas=(0.9, 0.999))
    
    # More gradual learning rate decay
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode='min',
                                                           factor=0.5,     # More gradual reduction
                                                           patience=3,     # Wait longer before reducing
                                                           verbose=True,
                                                           threshold=1e-4,
                                                           cooldown=1,     # Add cooldown period
                                                           min_lr=1e-6)
    # Loss function
    loss_fn = CapsuleLoss()

    # Dataloaders
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4) #, collate_fn=custom_collate)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4) #, collate_fn=custom_collate)

    best_val_accuracy = 0.0
    patience_counter = 0

    for epoch in range(EPOCHS):
        CAPS_model.train()
        train_predictions, train_true_labels = [], []
        train_loss = 0.0

        train_progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} - Training")
        for images, DCT_features, labels, captions in train_progress:
            images, DCT_features, labels= images.to(device), DCT_features.to(device), labels.to(device)
            labels_onehot = torch.eye(CLASSES).to(device).index_select(dim=0, index=labels)

            captions_tokens = tokenizer(list(captions)).to(device)
            with torch.no_grad():
                img_embedding = CLIP_model.encode_image(images).float()
                cap_embedding = CLIP_model.encode_text(captions_tokens)

            # output, reconstructions, masked = CAPS_model(img_embedding, cap_embedding, device=device)
            output = CAPS_model(img_embedding, cap_embedding, DCT_features, device=device)
            loss = loss_fn(output, labels_onehot)
            # loss = loss_fn(img_embedding, output, labels_onehot, reconstructions)

            # Get predictions from capsule output
            class_lengths = torch.sqrt((output ** 2).sum(dim=2))  # Shape: (batch_size, num_classes)
            predictions = torch.max(class_lengths, dim=1)[1]  # Shape: (batch_size)
            
            # Optimize the model
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(CAPS_model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Update running loss
            train_loss += loss.item()
            
            # Convert predictions and labels to CPU numpy arrays for metric calculation
            predictions = predictions.cpu().numpy()
            labels_np = labels.cpu().numpy()
            
            # Extend lists for epoch-level metrics
            train_predictions.extend(predictions)
            train_true_labels.extend(labels_np)
            
            # Calculate batch metrics
            batch_precision = precision_score(labels_np, predictions, average='binary', zero_division=0)
            batch_recall = recall_score(labels_np, predictions, average='binary', zero_division=0)
            batch_f1 = f1_score(labels_np, predictions, average='binary', zero_division=0)
            batch_accuracy = accuracy_score(labels_np, predictions)

            # Update tqdm description
            train_progress.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "Pre": f"{batch_precision:.4f}",
                "Re": f"{batch_recall:.4f}",
                "F1": f"{batch_f1:.4f}",
                "Acc": f"{batch_accuracy:.4f}"
            })

        precision = precision_score(train_true_labels, train_predictions, average='binary', zero_division=0)
        recall = recall_score(train_true_labels, train_predictions, average='binary', zero_division=0)
        f1 = f1_score(train_true_labels, train_predictions, average='binary', zero_division=0)
        train_accuracy = accuracy_score(train_true_labels, train_predictions)

        # Log metrics
        writer.add_scalar("Train/Loss", train_loss / len(train_loader), epoch + 1)
        writer.add_scalar("Train/Pre", precision, epoch + 1)
        writer.add_scalar("Train/Re", recall, epoch + 1)
        writer.add_scalar("Train/F1", f1, epoch + 1)
        writer.add_scalar("Train/Acc", train_accuracy, epoch + 1)


        # Print epoch metrics
        print(f"Epoch {epoch + 1}: Train Loss = {train_loss / len(train_loader):.4f}, Train Acc = {train_accuracy:.4f}, "
              f"Pre = {precision:.4f}, Re = {recall:.4f}, F1 = {f1:.4f}")
        logging.info(f"Epoch {epoch + 1}: Train Loss = {train_loss / len(train_loader):.4f}, Train Acc = {train_accuracy:.4f}, "
                     f"Pre = {precision:.4f}, Re = {recall:.4f}, F1 = {f1:.4f}")

        # Validation phase
        val_accuracy, val_loss, val_precision, val_recall, val_f1 = evaluate(CLIP_model, CAPS_model, tokenizer, val_loader, loss_fn, CLASSES, device)
        writer.add_scalar("Validation/Loss", val_loss, epoch + 1)
        writer.add_scalar("Validation/Pre", val_precision, epoch + 1)
        writer.add_scalar("Validation/Re", val_recall, epoch + 1)
        writer.add_scalar("Validation/F1", val_f1, epoch + 1)
        writer.add_scalar("Validation/Acc", val_accuracy, epoch + 1)

        # Print validation metrics
        print(f"Epoch {epoch + 1}: Val Loss = {val_loss:.4f}, Pre = {val_precision:.4f}, "
              f"Re = {val_recall:.4f}, F1 = {val_f1:.4f}, Val Acc = {val_accuracy:.4f}")
        logging.info(f"Epoch {epoch + 1}: Val Loss = {val_loss:.4f}, Pre = {val_precision:.4f}, "
                     f"Re = {val_recall:.4f}, F1 = {val_f1:.4f}, Val Acc = {val_accuracy:.4f}")

        # Update scheduler
        scheduler.step(val_loss)

        # Save the best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            model_save_path = os.path.join(log_dir, "best_model.pth")
            torch.save({'model_state_dict': CAPS_model.state_dict()}, model_save_path)
            logging.info(f"Saved best model with validation accuracy: {val_accuracy:.4f}")

        # Early stopping
        if val_accuracy <= best_val_accuracy:
            patience_counter += 1
        else:
            patience_counter = 0

        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            logging.info(f"Early stopping triggered after {epoch + 1} epochs")
            break

    print(f"Training completed. Best validation accuracy: {best_val_accuracy:.4f}")
    logging.info(f"Training completed. Best validation accuracy: {best_val_accuracy:.4f}")
    logging.info(log_id_display)
    writer.close()

def evaluate(CLIP_model, CAPS_model, tokenizer, val_loader, loss_fn, CLASSES, device):
    CAPS_model.eval()
    val_loss = 0.0
    val_predictions, val_true_labels = [], []
    total_tp, total_fp, total_tn, total_fn = 0, 0, 0, 0

    with torch.no_grad():
        val_progress = tqdm(val_loader, desc="Validation Progress", leave=False)
        for images, DCT_features, labels, captions in val_progress:
            images, DCT_features, labels= images.to(device), DCT_features.to(device), labels.to(device)
            labels_onehot = torch.eye(CLASSES).to(device).index_select(dim=0, index=labels)

            captions_tokens = tokenizer(list(captions)).to(device)
            with torch.no_grad():
                img_embedding = CLIP_model.encode_image(images).float()
                cap_embedding = CLIP_model.encode_text(captions_tokens)
                
            output = CAPS_model(img_embedding, cap_embedding, DCT_features, device=device)
            loss = loss_fn(output, labels_onehot)
            
            # Get predictions from capsule output
            class_lengths = torch.sqrt((output ** 2).sum(dim=2))  # Shape: (batch_size, num_classes)
            classes = F.softmax(class_lengths, dim=1)
            predictions = torch.max(class_lengths, dim=1)[1]  # Shape: (batch_size)
            
            val_loss += loss.item()

            # Convert predictions and labels to CPU numpy arrays for metric calculation
            predictions = predictions.cpu().numpy()
            labels_np = labels.cpu().numpy()

            val_predictions.extend(predictions)
            val_true_labels.extend(labels.cpu().numpy())

            batch_tp = ((predictions == 1) & (labels.cpu().numpy() == 1)).sum()
            batch_fp = ((predictions == 1) & (labels.cpu().numpy() == 0)).sum()
            batch_tn = ((predictions == 0) & (labels.cpu().numpy() == 0)).sum()
            batch_fn = ((predictions == 0) & (labels.cpu().numpy() == 1)).sum()

            total_tp += batch_tp
            total_fp += batch_fp
            total_tn += batch_tn
            total_fn += batch_fn

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = accuracy_score(val_true_labels, val_predictions)
    return accuracy, val_loss / len(val_loader), precision, recall, f1


def main():
    set_seed(24)
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--savepath", type=str, default="./MagicBrush_Trained_Models")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--image_root", type=str, default="./MagicBrush/BLIP_captioned_dataset/")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use (e.g., 0, 1, 2)")
    args = parser.parse_args()
    
    os.makedirs(args.savepath, exist_ok=True)
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.lr
    Weight_Decay = args.weight_decay
    CLASSES = 2
    SAVE_MODEL = args.savepath
    IMAGE_ROOT = args.image_root
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    
    # Extract the third last directory name from the path
    path_parts = os.path.normpath(IMAGE_ROOT).split(os.sep)
    dataset_name = path_parts[-3] if len(path_parts) >= 3 else "unknown"
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(SAVE_MODEL, f"{dataset_name}_run_{timestamp}")
    
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "training.log")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s", handlers=[logging.StreamHandler(),logging.FileHandler(log_file)])

    # Configuring model
    CLIP_model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup')
    tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup')
    
    CLIP_model.to(device)
    CLIP_model.eval()

    CAPS_model = CapsNet().to(device)

    # Paths to the directories containing source and target training images
    source_image_train = IMAGE_ROOT + "/train/source_images"
    target_image_train = IMAGE_ROOT + "/train/target_images"
    
    # Paths to the directories containing source and target validation images
    source_image_val = IMAGE_ROOT + "/val/source_images"
    target_image_val = IMAGE_ROOT + "/val/target_images"


    # Create the dataset
    train_dataset = CustomDatasetWithCaptions(real_dir=source_image_train, fake_dir=target_image_train, transform=train_transform)
    val_dataset = CustomDatasetWithCaptions(real_dir=source_image_val, fake_dir=target_image_val, transform=val_transform)

    train(CLIP_model, CAPS_model, tokenizer, train_dataset, val_dataset, EPOCHS, BATCH_SIZE, CLASSES, LEARNING_RATE, Weight_Decay, log_dir, device)


if __name__ == '__main__':
    main()




