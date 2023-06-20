# Import libraries
import torch
from torch import batch_norm, nn 
from torch.utils.data import DataLoader 

from utils import *
from dataset import mnist

newline()

#gpu 
device = 'cuda' if torch.cuda.is_available() else "cpu"
print("Using {} device \n".format(device))

# Dataloader
training_data, test_data, NUM_CLASSES = mnist()
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, kernel_size=3):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = None
        if stride != 1:
            self.conv3 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=2)
        

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.conv3 != None:
            identity = self.conv3(identity)
        out += identity
        out = self.relu2(out)

        return out

# ResNet model
class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()

        self.initial_fc = nn.Linear(in_features=28*28, out_features=227*227)

        # Input 227, output = (227-7)/2+1 = 112
        self.conv0 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=1, bias=False)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        in_channels = out_channels = 64
        self.block1 = ResidualBlock(in_channels, out_channels)
        self.block2 = ResidualBlock(in_channels, out_channels)
        self.block3 = ResidualBlock(in_channels, out_channels)

        in_channels = out_channels = 128
        self.block4 = ResidualBlock(int(in_channels/2), out_channels, stride=2)
        self.block5 = ResidualBlock(in_channels, out_channels)
        self.block6 = ResidualBlock(in_channels, out_channels)
        self.block7 = ResidualBlock(in_channels, out_channels)

        in_channels, out_channels = 256, 256
        self.block8 = ResidualBlock(int(in_channels/2), out_channels, stride=2)
        self.block9 = ResidualBlock(in_channels, out_channels)
        self.block10 = ResidualBlock(in_channels, out_channels)
        self.block11 = ResidualBlock(in_channels, out_channels)
        self.block12 = ResidualBlock(in_channels, out_channels)
        self.block13 = ResidualBlock(in_channels, out_channels)

        in_channels, out_channels = 512, 512
        self.block14 = ResidualBlock(int(in_channels/2), out_channels, stride=2)
        self.block15 = ResidualBlock(in_channels, out_channels)
        self.block16 = ResidualBlock(in_channels, out_channels)

        self.avg_pool = nn.AvgPool2d(kernel_size=7, stride=1)
        # self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=512, out_features=NUM_CLASSES)             # change the in_features here
        self.softmax = nn.Softmax(dim=1)
       
    def forward(self, x):
        out = x.view(batch_size, -1)
        out = self.initial_fc(out)
        out = out.view(batch_size, 1, 227, 227)
        out = self.conv0(out)
        out = self.max_pool(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)

        out = self.block4(out)
        out = self.block5(out)
        out = self.block6(out)
        out = self.block7(out)
        
        out = self.block8(out)
        out = self.block9(out)
        out = self.block10(out)
        out = self.block11(out)
        out = self.block12(out)
        out = self.block13(out)
        
        out = self.block14(out)
        out = self.block15(out)
        out = self.block16(out)
        out = self.avg_pool(out)
        out = out.view(batch_size, -1)
        # out = self.flatten(out)
        out = self.fc(out)
        out = self.softmax(out)
        return out

model = ResNet().to(device)
for features, label in train_dataloader:
    probabilities = model(features)
    print(probabilities.shape)

# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Accuracy 
# Training 
# Testing 
