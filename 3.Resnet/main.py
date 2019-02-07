from __future__ import print_function
import time
import torch
import random
import datetime
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import Tensor
import functools
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision

########## for reproducing the same results ##########
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32
test_bs = 128

######################## Create Dataset and DataLoader ########################
mean = [0.49139968, 0.48215827, 0.44653124]
std = [0.24703233, 0.24348505, 0.26158768]

train_tfm = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧圖像隨機裁剪成32*3
    transforms.RandomHorizontalFlip(),  # 圖像一半的概率翻轉，一半的概率不翻轉
    transforms.ToTensor(),
    transforms.Normalize(mean, std),  # R,G,B每層的歸一化用到的均值和方差
])

test_tfm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

Train_Set = datasets.CIFAR10(root='./data', train=True, transform=train_tfm, download=True)
Test_Set = datasets.CIFAR10(root='./data', train=False, transform=test_tfm, download=True)

train_loader = DataLoader(Train_Set, batch_size=batch_size, shuffle=True,
                          num_workers=0, pin_memory=True)

test_loader = DataLoader(Test_Set, batch_size=test_bs, shuffle=False,
                         num_workers=0, pin_memory=True)

'''
# Show img 
for batch_idx, (data, target) in enumerate(train_loader):
    data.numpy()
    plt.imshow(data[0,0])
    plt.show()
'''


######################## Helper Function ########################
def train(model, data_loader, optimizer, epoch, verbose=True):
    model.train()
    loss_avg = 0.0
    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss_avg += loss.item()
        loss.backward()
        optimizer.step()
        verbose_step = len(data_loader) // 10
        if batch_idx % verbose_step == 0 and verbose:
            print('Train Epoch: {}  Step [{}/{} ({:.0f}%)]  Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(data_loader.dataset),
                       100. * batch_idx / len(data_loader), loss.item()))
    return loss_avg / len(data_loader)


def test(model, data_loader):
    with torch.no_grad():
        model.eval()
        test_loss = 0
        correct = 0
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

        test_loss /= len(data_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset),
            100. * correct / len(data_loader.dataset)))
    return float(correct) / len(data_loader.dataset)


def adjust_learning_rate(base_lr, optimizer, epoch, decay_step=None, epoch_list=None):
    '''
    Set base_lr as initial LR and divided by 10 per $decay_step epochs
    '''
    assert decay_step is None or epoch_list is None, "decay_step and epoch_list can only set one of them"

    if epoch_list is not None:
        index = 0
        for i, e in enumerate(epoch_list, 1):
            if epoch >= e:
                index = i
            else:
                break
        lr = base_lr * (0.1 ** index)

    elif decay_step is not None:
        lr = base_lr * (0.1 ** (epoch // decay_step))

    else:
        # default decay_step to 30
        lr = base_lr * (0.1 ** (epoch // 30))

    # way to change lr in model
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def init_weights(layer):
    if isinstance(layer, nn.Linear):
        nn.init.xavier_normal_(layer.weight)
        layer.bias.data.fill_(0.01)
    elif isinstance(layer, nn.Conv2d):
        nn.init.xavier_normal_(layer.weight)
        layer.bias.data.fill_(0.01)


def ResNet18():
    return ResNet(ResidualBlock, num_of_blocks=6)


######################## Build Model ########################


#####搭建ResNet#####
class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=10, num_of_blocks=0):
        super(ResNet, self).__init__()
        self.inchannel = 16
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 16, 2 * num_of_blocks, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 32, 2 * num_of_blocks, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 64, 2 * num_of_blocks, stride=2)

        self.fc = nn.Linear(256, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = F.avg_pool2d(out, 4)
        # print('avg_pool out', out.shape)
        out = out.view(out.size(0), -1)
        # print('out_view out', out.shape) ###???
        out = self.fc(out)
        return out


# Define Loss and optimizer
learning_rate = 0.1

net = ResNet18().to(device)

# Initialze weights
# net.apply(init_weights)


optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

num_epochs = 150
StartTime = time.time()

loss, val_acc, lr_curve = [], [], []
for epoch in range(num_epochs):
    lr = adjust_learning_rate(learning_rate, optimizer, epoch, epoch_list=[80, 110, 130])
    train_loss = train(net, train_loader, optimizer, epoch, verbose=True)
    valid_acc = test(net, test_loader)
    loss.append(train_loss)
    val_acc.append(valid_acc)
    lr_curve.append(lr)

EndTime = time.time()
print('Time Usage: ', str(datetime.timedelta(seconds=int(round(EndTime - StartTime)))))

plt.figure()
plt.plot(loss)
plt.title('Train Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()

plt.figure()
plt.plot(val_acc)
plt.title('Valid Acc')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.show()

plt.figure()
plt.plot(lr_curve)
plt.title('Learning Rate')
plt.xlabel('epochs')
plt.ylabel('lr')
plt.yscale('log')
plt.show()