import torch
import torch.nn as nn

class InpaintModel(nn.Module):
    def __init__(self):
        super(InpaintModel, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()  # Output image with values between -1 and 1
        )

    def forward(self, mask_region):
        return self.model(mask_region)

    def inpaint(self, mask_region):
        mask_region = mask_region.unsqueeze(0)  # Add batch dimension
        return self(mask_region)
