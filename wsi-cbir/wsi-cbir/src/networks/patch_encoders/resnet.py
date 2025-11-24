### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from pathlib import Path
from typing import Union

### External Imports ###
import torch as tc
from torch.nn import Module

### Internal Imports ###
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import resnet50, ResNet50_Weights
from torchsummary import summary

########################



class ResNet18(tc.nn.Module):
    def __init__(self, weights=ResNet18_Weights.IMAGENET1K_V1):
        super().__init__()
        if weights is not None:
            self.model = resnet18(weights=weights)
        self.model.fc = tc.nn.Identity()

    def forward(self, images, metadata=None):
        return self.model(images)

    def load_model(self, weights_path):
        self.model.load_state_dict(weights_path)
        self.model.eval()

    def load_model_from_checkpoint(self, checkpoint_path):
        checkpoint = tc.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
    def get_transforms(self):
        return ResNet18_Weights.IMAGENET1K_V1.transforms(antialias=True)
        
class ResNet50(tc.nn.Module):
    def __init__(self, weights=ResNet50_Weights.IMAGENET1K_V2):
        super().__init__()
        if weights is not None:
            self.model = resnet50(weights=weights)
        self.model.fc = tc.nn.Identity()

    def forward(self, images, metadata=None):
        return self.model(images)

    def load_model(self, weights_path):
        self.model.load_state_dict(weights_path)
        self.model.eval()

    def load_model_from_checkpoint(self, checkpoint_path):
        checkpoint = tc.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
    def get_transforms(self):
        return ResNet50_Weights.IMAGENET1K_V2.transforms(antialias=True)
        
        
def test_resnet18():
    model = ResNet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    example_input = tc.randn((32, 3, 224, 224))
    example_result = model(example_input)
    print(f"Result shape: {example_result.shape}")
    summary(model.model.to("cuda:0"), (3, 224, 224))

def test_resnet50():
    model = ResNet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    example_input = tc.randn((32, 3, 224, 224))
    example_result = model(example_input)
    print(f"Result shape: {example_result.shape}")
    summary(model.model.to("cuda:0"), (3, 224, 224))

if __name__ == "__main__":
    test_resnet18()
    test_resnet50()


