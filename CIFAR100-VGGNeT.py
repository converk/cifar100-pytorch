# coding=utf-8
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as Data
import torch
import torch.nn as nn
from torch.autograd import Variable
# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LR=0.00008
EPOCH=30
BATCH_SIZE=180


# 定义对数据的预处理
transform = transforms.Compose([
        transforms.ToTensor(), # 转为Tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # 归一化
                             ])

# 训练集
trainset = torchvision.datasets.CIFAR100(
                    root='/input/1',
                    train=True,
                    download=False,
                    transform=transform)
#print(type(trainset.train_data))
trainloader = Data.DataLoader(
                    dataset=trainset,
                    batch_size=BATCH_SIZE,
                    shuffle=True,
                    num_workers=2)



#得到测试集的数据
testset = torchvision.datasets.CIFAR100(
                    '/input/1',
                    train=False,
                    download=False,
                    transform=transform)

testloader = Data.DataLoader(
                    dataset=testset,
                    batch_size=BATCH_SIZE,
                    shuffle=False,
                    num_workers=2)

class VGGNeT(nn.Module):
    def __init__(self):
        super(VGGNeT, self).__init__()
        self.features = self._make_layers()
        self.classifier = nn.Linear(512, 100)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self):
        cfg=[64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def train():
    #lenet = VGGNeT().to(device)
    lenet=torch.load('net2.pkl')
    optimizer=torch.optim.Adam(lenet.parameters(), lr=LR)
    loss_func=torch.nn.CrossEntropyLoss()

    for epoch in range(EPOCH):
        for i,(batch_x,batch_y) in enumerate(trainloader):
            x=Variable(batch_x.to(device))
            y=Variable(batch_y.to(device))
            out=lenet(x)
            loss = loss_func(out, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 200 == 0:
                torch.save(lenet,'net2.pkl')
                acc=test(lenet,testloader)
                print('Epoch1: ', epoch, '| test accuracy: %.4f' % acc)
    return lenet

def test(net,testloader):
    print('test:')
    acc,total=0, 0
    for input1, label in testloader:
        input1 = Variable(input1.to(device))
        label = Variable(label.to(device))
        out=net(input1)
        _,pre_y=torch.max(out, 1)
        total += label.size(0)
        acc += (pre_y == label).sum()
    return float(acc)/total

net = train()