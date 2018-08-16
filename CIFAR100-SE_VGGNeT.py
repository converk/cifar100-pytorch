# coding=utf-8
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as Data
import torch
from tensorboardX import SummaryWriter
import torch.nn as nn
from torch.autograd import Variable
# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

writer = SummaryWriter('/output/log')
LR=0.0002
EPOCH=30
BATCH_SIZE=100
local_dir='./cifar100'
far_dir='/input/1/'


# 定义对数据的预处理
train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
            transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

test_transform= transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

# 训练集
trainset = torchvision.datasets.CIFAR100(
                    root=far_dir,
                    train=True,
                    download=False,
                    transform=train_transform)
#print(type(trainset.train_data))
trainloader = Data.DataLoader(
                    dataset=trainset,
                    batch_size=BATCH_SIZE,
                    shuffle=True,
                    num_workers=8)



#得到测试集的数据
testset = torchvision.datasets.CIFAR100(
                    root=far_dir,
                    train=False,
                    download=False,
                    transform=test_transform)

testloader = Data.DataLoader(
                    dataset=testset,
                    batch_size=BATCH_SIZE,
                    shuffle=False,
                    num_workers=8)

class SE(nn.Module):
    def __init__(self,channels,ratio=16):
        super(SE,self).__init__()
        self.avg_pool=nn.AdaptiveAvgPool2d(1)
        self.fc=nn.Sequential(
            nn.Linear(channels,channels//ratio),
            nn.ReLU(inplace=True),
            nn.Linear(channels//ratio,channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b,c,_,_=x.size()
        y=self.avg_pool(x).view(b,c)
        y=self.fc(y).view(b,c,1,1)   #转化成概率
        #vis.heatmap()
        return x*y  #每个概率乘上相应的特征

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
                           SE(x,2),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def train():
    lenet=VGGNeT().to(device)
    optimizer=torch.optim.Adam(lenet.parameters(), lr=LR)
    loss_func=torch.nn.CrossEntropyLoss()
    j=0
    for epoch in range(EPOCH):
        train_acc,train_total=0,0
        for i,(batch_x,batch_y) in enumerate(trainloader):
            x=Variable(batch_x.to(device))
            y=Variable(batch_y.to(device))
            out=lenet(x)

            _,train_pre_y=torch.max(out,1)
            train_total+=y.size(0)
            train_acc+=(train_pre_y == y).sum()

            loss = loss_func(out, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 200 == 0:
                j+=1
                torch.save(lenet,'net5.pkl')
                acc_tr=float(train_acc)/train_total
                acc=test(lenet,testloader)
                print('Epoch: ', epoch, '| train loss:%.4f' % loss,'| train accuracy: %.4f' % acc_tr,'| test accuracy: %.4f' % acc)
                writer.add_scalars('data/group', {'train_loss':loss, 'train_acc':acc_tr, 'test_acc':acc}, j)
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
