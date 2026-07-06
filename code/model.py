import torch.nn as nn
import torchvision.models as models

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        backbone.fc = nn.Linear(512, 256)
        self.net = backbone

    def forward(self, x):
        return self.net(x)