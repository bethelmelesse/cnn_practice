# Import libraries
import torch
from tqdm import tqdm
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

        self.conv0 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=1, bias=False)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.residual_blocks = nn.ModuleList()

        in_channels = out_channels = 64
        for _ in range(3):
            self.residual_blocks.append(ResidualBlock(in_channels, out_channels))

        in_channels = out_channels = 128
        self.residual_blocks.append(ResidualBlock(int(in_channels/2), out_channels, stride=2))
        for _ in range(3):
            self.residual_blocks.append(ResidualBlock(in_channels, out_channels))

        in_channels, out_channels = 256, 256
        self.residual_blocks.append(ResidualBlock(int(in_channels/2), out_channels, stride=2))
        for _ in range(5):
            self.residual_blocks.append(ResidualBlock(in_channels, out_channels))

        in_channels, out_channels = 512, 512
        self.residual_blocks.append(ResidualBlock(int(in_channels/2), out_channels, stride=2))
        for _ in range(2):
            self.residual_blocks.append(ResidualBlock(in_channels, out_channels))

        #self.avg_pool = nn.AvgPool2d(kernel_size=7, stride=1)        # using interpolation
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=1)         # using padding
        self.fc = nn.Linear(in_features=512, out_features=NUM_CLASSES)             # change the in_features here
        # self.softmax = nn.Softmax(dim=1)
       
    def forward(self, x):
        out = x.to(device)
        # out = nn.functional.interpolate(out, size=227, mode='nearest-exact')          # using interpolation
        padding = nn.ZeroPad2d((0, 9, 0, 9))                                            # using padding 
        out = padding(out)
        out = self.conv0(out)
        out = self.max_pool(out)
        for i in range(len(self.residual_blocks)):
            out = self.residual_blocks[i](out)

        out = self.avg_pool(out)
        out = out.view(out.shape[0], -1)
        # out = self.flatten(out)
        out = self.fc(out)
        # out = self.softmax(out) # it is included in the ce loss fn
        return out

model = ResNet().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

# Accuracy 
def accuracy(y_pred, target):
    train_acc = torch.sum(torch.argmax(y_pred, dim=1) == target)
    final_train_acc = train_acc/target.shape[0]
    return final_train_acc * 100

def train():
    cur_step = 0
    for features, label in train_dataloader:
        logits = model(features)
        label = label.to(device)

        loss = ce_loss(logits, label)

        optimizer.zero_grad()
        loss.backward()
        if cur_step % 100 == 0:
           accuracy_batch = accuracy(logits, label)
           print('\033[32m' + f"Step = {cur_step}, Train Loss = {loss:.3f}, Train Accuracy = {accuracy_batch:.2f}%" + '\033[0m')
        optimizer.step()
        cur_step += 1
        # print(cur_step)

def test():
    model.eval()
    total_loss = 0
    total_accuracy = 0
    step = 0
    
    for features, label in test_dataloader:
        with torch.no_grad():
            logits = model(features)
            label = label.to(device)
            loss = ce_loss(logits, label)
            total_loss += loss
            accuracy_batch = accuracy(logits, label)
            total_accuracy += accuracy_batch
            step += 1
    total_loss = total_loss / step
    total_accuracy = total_accuracy /step
    print('\033[31m' + f"Test Loss = {total_loss:.3f}, Test Accuracy = {total_accuracy:.2f}%" + '\033[0m')
    model.train()

for i in tqdm(range(epoch)):
    newline()
    newline()
    print('\033[34m' + f"Epoch {i+1}" + '\033[0m')
    train()
    test()