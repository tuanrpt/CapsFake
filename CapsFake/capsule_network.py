import os
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

def standard_scale(tensor):
    mean = tensor.mean()
    std = tensor.std()
    return (tensor - mean) / std

class DCT_emb(nn.Module):
    def __init__(self, in_channels=(320*320), out_channels=768):
        super(DCT_emb, self).__init__()
        self.Emb = nn.Linear(in_channels, out_channels)
        
    def forward(self, DCT_features):
        DCT_features_reshaped = DCT_features.view(DCT_features.size(0), -1)
        DCT_features_reshaped = torch.log(torch.abs(DCT_features_reshaped) + 1e-12)
        DCT_embedding = standard_scale(DCT_features_reshaped)
        return F.relu(self.Emb(DCT_embedding))


class PrimaryCaps(nn.Module):
    def __init__(self, num_capsules=8, in_channels=768, out_channels=64, num_routes=64):
        super(PrimaryCaps, self).__init__()
        self.num_routes = num_routes
        self.img_capsules = nn.ModuleList([nn.Linear(in_channels, out_channels) for _ in range(num_capsules)])
        self.capt_capsules = nn.ModuleList([nn.Linear(in_channels, out_channels) for _ in range(num_capsules)])
        self.dct_capsules = nn.ModuleList([nn.Linear(in_channels, out_channels) for _ in range(num_capsules)])

    def forward(self, img_emb, capt_emb, dct_emb):
        # get batch size of inputs
        batch_size = img_emb.size(0)
        # reshape convolutional layer outputs to be (batch_size, vector_dim=1152, 1)
        img_emb_vectorization = [capsule(img_emb).view(batch_size, self.num_routes, 1) for capsule in self.img_capsules]
        capt_emb_vectorization = [capsule(capt_emb).view(batch_size, self.num_routes, 1) for capsule in self.capt_capsules]
        dct_emb_vectorization = [capsule(dct_emb).view(batch_size, self.num_routes, 1) for capsule in self.dct_capsules]

        # stack up output vectors, img_emb_vectorization, one for each capsule
        img_emb_vectorization = torch.cat(img_emb_vectorization, dim=-1)  # (batch_size, dim, n_capsules)
        capt_emb_vectorization = torch.cat(capt_emb_vectorization, dim=-1)  # (batch_size, dim, n_capsules)
        dct_emb_vectorization = torch.cat(dct_emb_vectorization, dim=-1)  # (batch_size, dim, n_capsules)

        #Concating both the image and caption vectors
        combined_vectors = torch.cat((img_emb_vectorization, capt_emb_vectorization, dct_emb_vectorization), dim=1) # (batch_size,dim*3,n_capsules)
        
        # squashing the stack of vectors
        u_squash = self.squash(combined_vectors)  # (batch_size, dim, n_capsules)
        # print('Squashed_shape: ', u_squash.shape)
        return u_squash

    def squash(self, input_tensor, epsilon=1e-7):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        safe_norm = torch.sqrt(squared_norm + epsilon)  # Add epsilon to avoid division by zero
        output_tensor = squared_norm * input_tensor / ((1. + squared_norm) * safe_norm)
        return output_tensor


class DigitCaps(nn.Module):
    def __init__(self, num_capsules=2, num_routes=192, in_channels=8, out_channels=64):
        super(DigitCaps, self).__init__()

        self.in_channels = in_channels
        self.num_routes = num_routes
        self.num_capsules = num_capsules

        # self.W = nn.Parameter(torch.randn(1, 1, num_capsules, out_channels, in_channels))
        self.W = nn.Parameter(torch.randn(1, num_routes, num_capsules, out_channels, in_channels))

    def forward(self, x, device):  # x: u vector from Primary Caps (batch_size, dim, n_prime_capsules)
        batch_size = x.size(0)

        x = torch.stack([x] * self.num_capsules, dim=2).unsqueeze(4)  # (batch_size, dim, n_digit_capsules, n_prime_capsules, 1)

        W = torch.cat([self.W] * batch_size, dim=0)  # (batch_size, dim, n_digit_capsules, n_prime_capsules, 1)
        u_hat = torch.matmul(W, x)

        b_ij = Variable(torch.zeros(1, self.num_routes, self.num_capsules, 1))
        b_ij = b_ij.to(device)

        num_iterations = 3
        for iteration in range(num_iterations):
            c_ij = F.softmax(b_ij, dim=1)
            c_ij = torch.cat([c_ij] * batch_size, dim=0).unsqueeze(4)

            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)
            v_j = self.squash(s_j)

            if iteration < num_iterations - 1:
                a_ij = torch.matmul(u_hat.transpose(3, 4), torch.cat([v_j] * self.num_routes, dim=1))
                b_ij = b_ij + a_ij.squeeze(4).mean(dim=0, keepdim=True)

        return v_j.squeeze(1)

    def squash(self, input_tensor, epsilon=1e-7):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        safe_norm = torch.sqrt(squared_norm + epsilon)  # Add epsilon to avoid division by zero
        output_tensor = squared_norm * input_tensor / ((1. + squared_norm) * safe_norm)
        return output_tensor


class CapsNet(nn.Module):
    def __init__(self):
        super(CapsNet, self).__init__()

        self.DCT_emb = DCT_emb()
        self.primary_capsules = PrimaryCaps()
        self.digit_capsules = DigitCaps()

    def forward(self, img_emb, capt_emb, DCT_features, device=None):
        dct_emb = self.DCT_emb(DCT_features) # dct_emb = [bs, 768, 1]
        output = self.digit_capsules(self.primary_capsules(img_emb, capt_emb, dct_emb), device)
        #print('Out_from_primary:', output.shape)
        
        return output


class CapsuleLoss(nn.Module):
    def __init__(self):
        super(CapsuleLoss, self).__init__()

    def forward(self, predicted_labels, actual_labels):
        margin_loss = self.margin_loss(predicted_labels, actual_labels)
        return margin_loss

    def margin_loss(self, predicted_labels, actual_labels):
        batch_size = predicted_labels.size(0)
        v_c = torch.sqrt((predicted_labels ** 2).sum(dim=2, keepdim=True) + 1e-7)

        # Clip values for stability
        v_c = torch.clamp(v_c, min=0.0, max=1.0)

        left = F.relu(0.9 - v_c).view(batch_size, -1)
        right = F.relu(v_c - 0.1).view(batch_size, -1)

        loss = actual_labels * left + 0.5 * (1.0 - actual_labels) * right
        loss = loss.sum(dim=1).mean()

        return loss
