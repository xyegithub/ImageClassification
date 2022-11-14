import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self, use_arf=False, num_orientation=8):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 80, kernel_size=3, padding = 1)
        self.conv2 = nn.Conv2d(80, 160, kernel_size=3, padding = 1)
        self.conv3 = nn.Conv2d(160, 320, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(320, 640, kernel_size=3)
        self.global_average = nn.AdaptiveMaxPool2d((1, 1))
        self.fc1 = nn.Linear(640, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv4(x))

        x = self.global_average(x).squeeze()
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

