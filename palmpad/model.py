import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class PalmPadModel(nn.Module):
    """
    PalmPad: 3-branch ResNet18 + LSTM + MLP touch classifier.

    At each time step:
      - palm branch  : 128x128 RGB crop of the palm
      - index branch : 128x128 RGB crop of the index fingertip
      - flow branch  : 128x128 optical-flow (2-channel) between consecutive frames

    The three 1000-d features are concatenated → LSTM → MLP → touch logits.

    Reference: He et al., CHI 2025. doi:10.1145/3706598.3714130
    """

    def __init__(
        self,
        time_steps: int = 2,
        lstm_hidden: int = 512,
        lstm_layers: int = 1,
        mlp_dropout: float = 0.5,
    ):
        super().__init__()
        self.time_steps = time_steps

        # Palm & index encoders — pretrained on ImageNet
        self.palm_enc = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.index_enc = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # Optical-flow encoder — 2-channel input, no ImageNet pretrain
        self.flow_enc = resnet18(weights=None)
        self.flow_enc.conv1 = nn.Conv2d(
            2, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        feat_dim = 1000  # ResNet18 default fc output
        combined = feat_dim * 3  # 3000

        self.lstm = nn.LSTM(
            input_size=combined,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=mlp_dropout if lstm_layers > 1 else 0.0,
        )

        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(mlp_dropout),
            nn.Linear(256, 2),
        )

    def forward(
        self,
        palm: torch.Tensor,   # (B, T, 3, 128, 128)
        index: torch.Tensor,  # (B, T, 3, 128, 128)
        flow: torch.Tensor,   # (B, T, 2, 128, 128)
    ) -> torch.Tensor:        # (B, 2)
        B, T = palm.shape[:2]

        features = []
        for t in range(T):
            p = self.palm_enc(palm[:, t])    # (B, 1000)
            i = self.index_enc(index[:, t])  # (B, 1000)
            f = self.flow_enc(flow[:, t])    # (B, 1000)
            features.append(torch.cat([p, i, f], dim=-1))  # (B, 3000)

        seq = torch.stack(features, dim=1)         # (B, T, 3000)
        out, _ = self.lstm(seq)                    # (B, T, hidden)
        return self.classifier(out[:, -1])         # (B, 2)
