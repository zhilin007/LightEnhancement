from torchstat import stat
from models import *
from torchvision import models
import torch
net=EUUNet()
# net=models.resnet18()
# x=torch.ones([1,3,224,224])
# net(x)
# stat(net,(3,224,224))
stat(net,(4,1920,1072))