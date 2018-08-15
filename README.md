# VGG_SE-cifar100-pytorch
**使用VGGNeT来实现cifar100的分类, 两个.py文件一个是加了SE结构,一个没有加,在我测试的过程中发现加上SE结构之后,正确率大概上升了2%左右**
***
两个网络都在zip包里面,两个.py文件里面的代码我又修改了一点, 所以.py文件里面的代码并不是产生下面两个正确率的代码, 两个zip包里面的两个网络才是实现下面正确率的网络
***
没有使用VGG16或19是因为,我测试过之后发现比13只差不到1%(可能是我参数没调好),所以选择了更简单的13
***
## VGGNeT13
单独的VGGNeT的正确率在58%左右
![VGGNET](https://github.com/converk/VGG_SE-cifar100-pytorch/blob/master/VGG.png)

---
## VGGNeT13+SENeT
单独的VGGNeT的正确率在60%左右
![SE_VGGNET](https://github.com/converk/VGG_SE-cifar100-pytorch/blob/master/VGG_SE.png)
