import torch.nn as nn
from residualBlock import ResidualBlock

class OCRFeatureExtractor(nn.Module):
    def __init__(self, input_channels=1):
        super().__init__()
        # Set input_channels=1 for Grayscale, 3 for RGB
        super(OCRFeatureExtractor, self)
        
        # Block 0: Initial Convolution
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) 

        # Block 1: 64 -> 128 Channels
        self.layer1 = ResidualBlock(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) 

        # Block 2: 128 -> 256 Channels (Rectangular Pooling begins here)
        self.layer2 = ResidualBlock(128, 256)
        # kernel_size=(2, 1) halves the height, but keeps width exactly the same
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)) 

        # Block 3: 256 -> 512 Channels
        self.layer3 = ResidualBlock(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))

    def forward(self, x):
        # Pass through initial layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)

        # Pass through residual blocks and pooling
        x = self.layer1(x)
        x = self.pool2(x)

        x = self.layer2(x)
        x = self.pool3(x)

        x = self.layer3(x)
        x = self.pool4(x)
        
        return x
