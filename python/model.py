"""
The MNIST CNN from the architecture diagram:

    (1,28,28) -conv3x3-> (16,26,26) -relu-conv3x3-> (32,24,24) -relu-maxpool-> (32,12,12)
    -dropout-flatten-> 4608 -fc-> 64 -relu-dropout-fc-> 10

forward() returns raw logits (no log_softmax): training uses CrossEntropyLoss,
which applies log_softmax itself, and argmax is the same either way. Keeping the
last layer linear also keeps the C++/CUDA engine simpler.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# MNIST normalisation constants, shared with export.py and the C++ engine's test file
MEAN, STD = 0.1307, 0.3081


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(32 * 12 * 12, 64)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        return self.fc2(x)


def load(path: str, device: str = "cpu") -> Net:
    model = Net()
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device).eval()
