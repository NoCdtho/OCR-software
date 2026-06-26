import torch.nn as nn
from CRNN.residualBlock import ResidualBlock
import torch.nn.functional as F

class CRNN(nn.Module):
    def __init__(self, num_classes, img_height=32):
        super().__init__()
        self.cnn = nn.Sequential(
            # initial conv
            nn.Conv2d(1, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # residual blocks with pooling (only vertical stride)
            ResidualBlock(64, 64),
            nn.MaxPool2d((2, 1)),   # H/2
            ResidualBlock(64, 128),
            nn.MaxPool2d((2, 1)),   # H/4
            ResidualBlock(128, 256),
            ResidualBlock(256, 256),
            nn.MaxPool2d((2, 1)),   # H/8
            ResidualBlock(256, 512),
            nn.MaxPool2d((2, 1)),   # H/16
            ResidualBlock(512, 512),
            nn.MaxPool2d((2, 1)),   # H/32 → height = 1
        )
        # Recurrent part
        self.rnn = nn.LSTM(512, 256, num_layers=2,
                           bidirectional=True, batch_first=True)
        self.fc = nn.Linear(512, num_classes)  # 256*2

    def forward(self, x):
        # x: (B, 1, H, W)
        feats = self.cnn(x)                    # (B, 512, 1, W')
        feats = feats.squeeze(2)               # (B, 512, W')
        feats = feats.permute(0, 2, 1)         # (B, W', 512) batch_first
        rnn_out, _ = self.rnn(feats)           # (B, W', 512)
        logits = self.fc(rnn_out)              # (B, W', num_classes)
        return F.log_softmax(logits, dim=2)