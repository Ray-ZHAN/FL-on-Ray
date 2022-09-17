import torch.nn as nn
import torch.nn.functional as F

class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, kernel_size=(3, 3))
        self.fc = nn.Linear(192, 10)
        
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 3))
        x = x.view(-1, 192)
        x = self.fc(x)
        return F.log_softmax(x, dim = 1)
    
    def get_weight(self):
        return {k: v for k, v in self.state_dict().items()}
    
    def set_weight(self, weights):
        self.load_state_dict(weights)
        
class CIFARNet(nn.Module):
    pass