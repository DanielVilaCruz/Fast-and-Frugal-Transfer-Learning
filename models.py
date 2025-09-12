import torchvision.models as models
import torch
import torch.nn as nn
import numpy as np
from torchvision.models import mobilenet_v3_small, mobilenet_v3_large, densenet121, densenet169, densenet201, vit_b_16, vit_l_16
import torch.nn.functional as F
import math 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class IntermediateFeatureExtractor(nn.Module):
    def __init__(self, last_layer=2, pretrained=True, base_model='resnet50', quant=None):
        super().__init__()
        self.base_model = base_model.lower()
        self.last_layer = last_layer
        self.quant = quant

        if self.base_model.startswith('vit'):
            self.init_vit(pretrained=pretrained)
        elif self.base_model.startswith("gnn"):
            self.init_gnn()
        else:
            self.init_cnn(base_model=base_model, pretrained=pretrained)

        for param in self.features.parameters():
            param.requires_grad = False

    def init_cnn(self, base_model, pretrained):
        if base_model == 'resnet18':
            base = models.resnet18(pretrained=pretrained)
        if base_model == 'resnet50':
            base = models.resnet50(pretrained=pretrained) 
        if base_model == 'resnet152':
            base = models.resnet152(pretrained=pretrained)
        elif base_model == "mobilenet_v3_large":
            base = mobilenet_v3_large(pretrained=pretrained) 
        elif base_model == "mobilenet_v3_small":
            base = mobilenet_v3_small(pretrained=pretrained) 
        elif base_model == "densenet121":
            base = densenet121(pretrained=pretrained)
        elif base_model == "densenet201":
            base = densenet201(pretrained=pretrained)
        elif base_model == "densenet169":
            base = densenet169(pretrained=pretrained)

        self.features = nn.Sequential(*list(base.children())[:-self.last_layer]) 

        # Add avg pooling layer
        if self.last_layer > 1:
            self.features.add_module('avgpool', nn.AdaptiveAvgPool2d((1, 1)))
        if self.quant:
            # self.convert_to_half()
            self.convert_precision(quant=self.quant)


    def convert_to_half(self):
        for module in self.modules():
            if not isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.half()

    def convert_precision(self, quant=torch.float32):
        for module in self.modules():
            # Don't convert batchnorm layers to half precision (to avoid instability)
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                continue
            module.to(dtype=quant)

    def init_vit(self, pretrained):
        """Initialize Vision Transformer with options for different variants"""
        vit_models = {
            'vit_b_16': vit_b_16,
            'vit_l_16': vit_l_16,
        }
                
        self.features = vit_models[self.base_model](pretrained=pretrained)
        self.features.head = nn.Identity()  # Remove the classification head


    def forward(self, x, edge_index=None):
        return self.features(x)


class ClassifierHead_spatial(nn.Module):
    def __init__(self, in_dim=512, grid_size=7, num_classes=10):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=1)  # (B, 512*7*7)
        self.fc = nn.Sequential(
            nn.Linear(in_dim * grid_size * grid_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        return self.fc(self.flatten(x))  # (B, num_classes)
    
def get_detection_layer(model):
    """
    Extract the detection layer from a ResNet model.
    This is typically the last fully connected layer before the output.
    """
    if isinstance(model, nn.Sequential):
        return model[-1]  # Last layer in a sequential model
    elif hasattr(model, 'fc'):
        return model.fc  # For ResNet models with a 'fc' attribute
    else:
        raise ValueError("Model does not have a detection layer.")

    
def get_head(data_features, data_labels, simple=True):
    
    num_classes = len(np.unique(data_labels))
    input_features_dim = data_features.shape[1] 

    if simple:
        head = nn.Linear(in_features=input_features_dim, out_features=num_classes, bias=True)
    else:
        w = 512 
        head = nn.Sequential(
            nn.Linear(input_features_dim, w),
            nn.BatchNorm1d(w),
            nn.ReLU(),
            nn.Linear(w, num_classes)
        )
    return head.to(device)


def get_full_model(model_name, pretrained=True):
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
    elif model_name == 'resnet152':
        model = models.resnet152(pretrained=pretrained)
    elif model_name == "mobilenet_v3_large":
        model = mobilenet_v3_large(pretrained=True) 
    elif model_name == "mobilenet_v3_small":
        model = mobilenet_v3_small(pretrained=True) 
    elif model_name == "densenet121":
        model = densenet121(pretrained=True)

    elif "vit" in model_name:
        vit_models = {
            'vit_b_16': vit_b_16,
            'vit_l_16': vit_l_16,
        }
        model = vit_models[model_name](pretrained=pretrained)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    return model

