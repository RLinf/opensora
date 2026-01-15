import torch
import torch.nn as nn
from torch.nn import functional as F

from opensora.registry import MODELS

class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            input_channels, num_channels, kernel_size=3, padding=1, stride=strides
        )
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(
                input_channels, num_channels, kernel_size=1, stride=strides
            )
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)


def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(
                Residual(input_channels, num_channels, use_1x1conv=True, strides=2)
            )
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk


class RewModel(nn.Module):
    def __init__(
        self,
        from_pretrained=None,
    ) -> None:
        super().__init__()
        b1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
        b3 = nn.Sequential(*resnet_block(64, 128, 2))
        b4 = nn.Sequential(*resnet_block(128, 256, 2))
        b5 = nn.Sequential(*resnet_block(256, 512, 2))
        self.net = nn.Sequential(
            b1,
            b2,
            b3,
            b4,
            b5,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

        if from_pretrained is not None:
            self.load_checkpoint(from_pretrained)
        else:
            raise ValueError("from_pretrained is required")

    def load_checkpoint(self, checkpoint_path):
        """
        Load model weights from checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file (.pth)
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False,)
        if 'model_state_dict' in checkpoint:
            self.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.load_state_dict(checkpoint)

    @torch.no_grad()
    def predict_rew(self, obs):
        # assert obs.max() <= 1.5 and obs.min() >= -1.5, f"obs.max() is {obs.max()}, and obs min is {obs.min()}"
        obs = obs.clamp(-1.0, 1.0)
        x = self.net(obs.to(dtype=torch.float32))
        # 分为 0 或 1
        x = torch.round(x)
        return x

    def forward(self, obs=None):
        return self.predict_rew(obs)


@MODELS.register_module()
def ResnetRM(from_pretrained=None):
    model = RewModel(from_pretrained=from_pretrained)
    return model