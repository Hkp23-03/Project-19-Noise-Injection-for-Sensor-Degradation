import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50

class DeepLabV3PlusModel(nn.Module):
    """
    DeepLabV3 architecture wrapper using a ResNet50 backbone.
    Provides Atrous Spatial Pyramid Pooling (ASPP) to robustly capture
    multi-scale context, which is heavily useful when local pixels are degraded by noise.
    """
    def __init__(self, num_classes, pretrained_backbone=True, freeze_backbone=False):
        super(DeepLabV3PlusModel, self).__init__()
        
        # Load DeepLabV3 with a ResNet50 backbone
        self.model = deeplabv3_resnet50(pretrained=False, pretrained_backbone=pretrained_backbone)
        
        # Replace the classifier head to match our number of classes
        self.model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
        
        # Adjust the auxiliary classifier if present
        if self.model.aux_classifier is not None:
            self.model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
            
        if freeze_backbone:
            for param in self.model.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        # Torchvision segmentation models return a dictionary
        # 'out' is the main output, 'aux' is the auxiliary output
        return self.model(x)['out']
