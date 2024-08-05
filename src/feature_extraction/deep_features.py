import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import SimpleITK as sitk
from scipy.ndimage import zoom


def load_model(model_name, in_channels):
    if model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
    elif model_name == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif model_name == 'densenet121':
        model = models.densenet121(pretrained=True)
    else:
        raise ValueError("Unsupported model name.")

    if in_channels != 3:
        if model_name == 'resnet50':
            model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        elif model_name == 'vgg16':
            model.features[0] = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        elif model_name == 'densenet121':
            model.features.conv0 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

    model = nn.Sequential(*list(model.children())[:-1])  # Remove the classification layer
    model.eval()
    return model


def preprocess_image(img_array, mask_array):
    img_array[mask_array == 0] = 0

    # Normalize image array
    img_tensor = torch.tensor(img_array, dtype=torch.float32)
    #img_tensor = img_tensor.permute(2, 0, 1)  # Change to (C, H, W) format

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485] * img_tensor.shape[0], std=[0.229] * img_tensor.shape[0]),
    ])

    img_tensor = preprocess(img_tensor)
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
    return img_tensor


    # img_array[mask_array == 0] = 0
    #
    # # Normalize image array
    # img_tensor = torch.tensor(img_array, dtype=torch.float32)
    # img_tensor = img_tensor.permute(2, 0, 1)  # Change to (C, H, W) format
    #
    # preprocess = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.Normalize(mean=[0.485] * img_tensor.shape[0], std=[0.229] * img_tensor.shape[0]),
    # ])
    #
    # img_tensor = preprocess(img_tensor)
    # img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
    # return img_tensor


def load_and_preprocess_image(image_path, mask_path):
    img = sitk.ReadImage(image_path)
    img_array = sitk.GetArrayFromImage(img)
    mask = sitk.ReadImage(mask_path)
    mask_array = sitk.GetArrayFromImage(mask)

    if len(img_array.shape) == 4:
        img_array = img_array[0]

    # Resize each slice of the 3D volume independently
    resized_img_slices = []
    resized_mask_slices = []
    for i in range(img_array.shape[0]):
        resized_img_slice = zoom(img_array[i], (224 / img_array.shape[1], 224 / img_array.shape[2]), order=3)
        resized_mask_slice = zoom(mask_array[i], (224 / mask_array.shape[1], 224 / mask_array.shape[2]), order=0)
        resized_img_slices.append(resized_img_slice)
        resized_mask_slices.append(resized_mask_slice)

    resized_img_array = np.stack(resized_img_slices, axis=0)
    resized_mask_array = np.stack(resized_mask_slices, axis=0)

    img_tensor = preprocess_image(resized_img_array, resized_mask_array)
    return img_tensor

def extract_deep_features(image_path, mask_path, model_name='resnet50'):
    img = sitk.ReadImage(image_path)
    img_array = sitk.GetArrayFromImage(img)

    if len(img_array.shape) == 4:
        img_array = img_array[0]

    in_channels = img_array.shape[0]

    model = load_model(model_name, in_channels)
    img_tensor = load_and_preprocess_image(image_path, mask_path)

    with torch.no_grad():
        features = model(img_tensor).squeeze().numpy()

    # Convert features to a dictionary
    feature_dict = {f"feature_{i}": feature for i, feature in enumerate(features)}
    return feature_dict



def save_deep_features(features, output_file):
    df = pd.DataFrame(features)
    df.to_csv(output_file, index=False)


def process_and_save_deep_features(image_dir, mask_dir, model_name, output_file):
    features_list = []
    for img_file in os.listdir(image_dir):
        if img_file.endswith('nii') or img_file.endswith('nii.gz') or img_file.endswith('mha'):
            img_path = os.path.join(image_dir, img_file)
            mask_path = os.path.join(mask_dir, img_file)
            features = extract_deep_features(img_path, mask_path, model_name)
            features_list.append(features)
    save_deep_features(features_list, output_file)

# Example usage:
# process_and_save_deep_features('path_to_image_dir', 'path_to_mask_dir', 'resnet50', 'output_features.csv')


