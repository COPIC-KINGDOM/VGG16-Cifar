import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from VGG16.VGG16_Cifar10_Model import VGG16_Cifar10_Model

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained VGG model and move it to GPU
vgg_model = VGG16_Cifar10_Model()
vgg_model = vgg_model.to(device)

# Define a new model with VGG's convolutional layers
vgg_model.eval()

# 选择输出文件夹
output_folder = '/media/tust/COPICAI/cifar_data_A'
heatmap_folder = '/media/tust/COPICAI/cifar_data_img_A'
os.makedirs(output_folder, exist_ok=True)
os.makedirs(heatmap_folder, exist_ok=True)

# Define transformation and load CIFAR-10 dataset
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

cifar10_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)
data_loader = DataLoader(cifar10_dataset, batch_size=1, shuffle=True)

class_outputs = {}

conv_layers = [
        vgg_model.conv1_1, vgg_model.relu1_1, vgg_model.conv1_2, vgg_model.relu1_2, vgg_model.pool1,
        vgg_model.conv2_1, vgg_model.relu2_1, vgg_model.conv2_2, vgg_model.relu2_2, vgg_model.pool2,
        vgg_model.conv3_1, vgg_model.relu3_1, vgg_model.conv3_2, vgg_model.relu3_2, vgg_model.conv3_3, vgg_model.relu3_3, vgg_model.pool3,
        vgg_model.conv4_1, vgg_model.relu4_1, vgg_model.conv4_2, vgg_model.relu4_2, vgg_model.conv4_3, vgg_model.relu4_3, vgg_model.pool4,
        vgg_model.conv5_1, vgg_model.relu5_1, vgg_model.conv5_2, vgg_model.relu5_2, vgg_model.conv5_3, vgg_model.relu5_3, vgg_model.pool5,
    ]

# Iterate over each label
for label in range(10):
    # Define output and heatmap folders for the current label
    output_folder = f'/media/tust/COPICAI/cifar_data_A/cifar_{label}_vgg_outputs'
    heatmap_folder = f'/media/tust/COPICAI/cifar_data_img_A/cifar_{label}_hot_img'
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(heatmap_folder, exist_ok=True)

    # Iterate over each image in the dataset
    for idx, batch in enumerate(data_loader):
        labels = batch[1]
        if labels.item() == label:
            input_image = batch[0].to(device)
            all_outputs = {}

            # Iterate over each layer and save outputs
            for i, layer in enumerate(conv_layers):
                input_image = layer(input_image)
                layer_output = input_image.squeeze().detach().cpu().numpy()

                for j in range(layer_output.shape[0]):
                    key = f'vgg_layer_{i + 1}_channel_{j + 1}_output'
                    all_outputs[key] = layer_output[j:j + 1]

                    # Optionally, save heatmap images
                    # channel_output = layer_output[j:j + 1].squeeze()
                    # plt.imshow(channel_output, cmap='viridis', aspect='auto')
                    # plt.title(f'Layer {i + 1} Channel {j + 1} Output')
                    # plt.colorbar()
                    # heatmap_filename = os.path.join(heatmap_folder,
                    #                                 f'vgg_layer_{i + 1}_channel_{j + 1}_output_heatmap_{idx}.png')
                    # plt.savefig(heatmap_filename)
                    # plt.close()

            npz_filename = os.path.join(output_folder, f'image_{idx}_npz.npz')
            np.savez(npz_filename, **all_outputs)
            print(f"Label {label}, Image {idx}: Outputs saved to {npz_filename}")

            del input_image, all_outputs
            torch.cuda.empty_cache()
