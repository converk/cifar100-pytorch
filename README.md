# VGG_SE-cifar100-pytorch
***
没有使用VGG16或19是因为,我测试过之后发现比13只差不到1%(可能是我参数没调好),所以选择了更简单的13
***
## VGGNeT13
单独的VGGNeT的正确率在58%左右
![VGGNET](https://github.com/converk/VGG_SE-cifar100-pytorch/blob/master/VGG.png)

***
## 两种bitch_size下的情况
**bitch_size=180**
![VGGNET](https://github.com/converk/VGG_SE-cifar100-pytorch/blob/master/tensorboard-batch_size180.png)

**bitch_size=100**
![VGGNET](https://github.com/converk/VGG_SE-cifar100-pytorch/blob/master/tensorboard-batch_size100.png)

**不同bitch_size比较**

    对于100的无论是在训练集准确率,测试集准确率都要低于180,loss也略高于180
