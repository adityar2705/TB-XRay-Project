import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import timm
from .config import DEVICE

def build_unet():
    model = smp.Unet('resnet18', classes=1, activation=None).to(DEVICE)
    return model

def build_vit(pretrained=True, num_classes=5):
    model = timm.create_model('vit_base_patch16_224', pretrained=pretrained, num_classes=num_classes)
    return model.to(DEVICE)
