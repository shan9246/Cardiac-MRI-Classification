import torch
import torch.nn as nn
from torchvision import models
import torchvision.models as models

class MobileNetTransformerClassifier(nn.Module):
    def __init__(
        self,
        num_classes=5,
        d_model=256,
        nhead=8,
        num_layers=4,
        dropout=0.1
    ):
        super().__init__()

        # -------- CNN Backbone --------
        mobilenet = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.IMAGENET1K_V1
        )

        # Modify first conv for 1-channel MRI
        mobilenet.features[0][0] = nn.Conv2d(
            1, 32, kernel_size=3, stride=2, padding=1, bias=False
        )

        self.cnn = mobilenet.features
        self.feature_dim = 1280

        # -------- Feature Projection --------
        self.proj = nn.Linear(self.feature_dim, d_model)

        # -------- Positional Encoding --------
        self.pos_embed = nn.Parameter(torch.randn(1, 100, d_model))  # max T=100

        # -------- Transformer Encoder --------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # -------- Classification Head --------
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        """
        x: (B, T, 1, H, W)
        """
        B, T, C, H, W = x.shape

        # ---- CNN feature extraction ----
        x = x.view(B * T, C, H, W)
        feats = self.cnn(x).mean([2, 3])          # (B*T, 1280)
        feats = feats.view(B, T, self.feature_dim)

        # ---- Project to transformer dim ----
        feats = self.proj(feats)                  # (B, T, d_model)

        # ---- Add positional encoding ----
        feats = feats + self.pos_embed[:, :T, :]

        # ---- Transformer ----
        out = self.transformer(feats)             # (B, T, d_model)

        # ---- Temporal pooling (choose one) ----
        final = out.mean(dim=1)                    # Mean pooling
        # final = out[:, -1]                       # OR last token

        return self.fc(final)

# ============================================================
# ✅ ConvLSTM Cell Definition
# ============================================================
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3, padding=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            bias=True
        )

    def forward(self, x, prev_state):
        h_prev, c_prev = prev_state

        combined = torch.cat([x, h_prev], dim=1)
        gates = self.conv(combined)

        i, f, o, g = torch.chunk(gates, 4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch, shape, device):
        return (
            torch.zeros(batch, self.hidden_dim, *shape).to(device),
            torch.zeros(batch, self.hidden_dim, *shape).to(device),
        )


# ============================================================
# ✅ Model 1: MobileNet + LSTM Classifier
# ============================================================
class MobileNetLSTMClassifier(nn.Module):
    def __init__(self, num_classes=5, lstm_hidden=256, bidirectional=False):
        super().__init__()

        mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

        # Change input to single-channel MRI
        mobilenet.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)

        self.cnn = mobilenet.features  # extract features
        self.embedding_dim = 1280

        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        lstm_out_dim = lstm_hidden * (2 if bidirectional else 1)

        self.fc = nn.Linear(lstm_out_dim, num_classes)

    def forward(self, x):  # x: (B, T, 1, H, W)
        B, T, C, H, W = x.shape

        x = x.view(B*T, C, H, W)
        feats = self.cnn(x).mean([2,3])   # (B*T, 1280)
        feats = feats.view(B, T, -1)      # (B, T, 1280)

        lstm_out, _ = self.lstm(feats)
        final = lstm_out[:, -1, :]  # last timestep

        return self.fc(final)


# ============================================================
# ✅ Model 2: MobileNet + ConvLSTM Classifier
# ============================================================
class MobileNetConvLSTMClassifier(nn.Module):
    def __init__(self, num_classes=5, conv_lstm_hidden=128):
        super().__init__()

        mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

        # Change input to 1-channel MRI
        mobilenet.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)

        self.cnn = mobilenet.features
        self.feature_dim = 1280

        self.conv_lstm = ConvLSTMCell(input_dim=self.feature_dim, hidden_dim=conv_lstm_hidden)

        self.fc = nn.Linear(conv_lstm_hidden, num_classes)

    def forward(self, x):  # x: (B, T, 1, H, W)
        B, T, C, H, W = x.shape

        # Extract CNN features for every frame
        x = x.view(B*T, C, H, W)
        feats = self.cnn(x).mean([2,3])                 # (B*T, 1280)
        feats = feats.view(B, T, self.feature_dim, 1, 1)

        # Initialize ConvLSTM hidden state
        h, c = self.conv_lstm.init_hidden(B, (1, 1), x.device)

        # Feed temporal sequence through ConvLSTM
        for t in range(T):
            h, c = self.conv_lstm(feats[:, t], (h, c))

        final = h.view(B, -1)  # Flatten last ConvLSTM output
        return self.fc(final)
