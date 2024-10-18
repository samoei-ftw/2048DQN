import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(16, 128)  # 16 input nodes (4x4 grid)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 4)  # 4 output nodes (one for each action: up, down, left, right)
    
    def forward(self, x):
        x = torch.flatten(x)  # Flatten the 4x4 grid to a 1D tensor
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # Output Q-values for each action
