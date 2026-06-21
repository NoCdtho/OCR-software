import torch.nn as nn
from CNN_model import OCRFeatureExtractor

class CRNN(nn.Module):
    def __init__(self, input_channels=1, hidden_size=256, num_layers=2, num_classes=37):
        super(CRNN, self).__init__()

        #CNN Stage
        self.cnn = OCRFeatureExtractor(input_channels)

        #join CNN with LSTM by collapse height dimension
        # CNN output: (batch, 512, 2, 64) (batch, number of channel, height, width)
        # After collapsing height: (batch, 64, 1024)  ← 512*2 = 1024 features per time step
        self.lstm_input_size = 512 * 2  # channels × height

        #LSTM Stage
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,  # features per time step = one timestep = 1024 number of patterns 
            hidden_size=hidden_size,          # 256
            num_layers=num_layers,            # stacked LSTMs
            batch_first=True,                 # (batch, timesteps, features)
            bidirectional=True,               # reads text left→right AND right → left
            dropout=0.2                       # dropout between LSTM layers
        )

        # Output: map LSTM output to character classes
        # bidirectional → hidden_size * 2
        """
        It will convert the 256*2=512 numbers in the vector into 36 numbers in the vector mapping the num_clases
        that is the outputs where each number represent the character prediction by the model.
        """
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # Step 1: CNN feature extraction
        x = self.cnn(x)
        # x shape: (batch, 512, 2, 64) (batch, channel, height, width)

        # Step 2: Reshape for LSTM
        batch, channels, height, width = x.size()
        x = x.permute(0, 3, 1, 2) # →(batch, width, channels, height) change position here
        x = x.reshape(batch, width, channels * height) # can do operations here between 
        # x shape: (batch, 64, 1024)  ← 64 time steps, 1024 features each

        # Step 3: LSTM sequence modeling (sequence, timesteps, features)
        x, _ = self.lstm(x) 
        # x shape: (batch, 64, 512)  ← 512 = 256 * 2 (bidirectional)

        # Step 4: Project to character classes at each time step
        x = self.fc(x)
        # x shape: (batch, 64, num_classes)

        return x
    
