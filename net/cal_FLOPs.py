from torchstat import stat
from models import *
from torchvision import models
import torch
# net=EUUNet()
# net=SwiftNetSlim()
# net=EFFA(2,10)
net=FULLCONV()
# print(net)
# x=torch.ones([1,3,224,224])
# net(x)
stat(net,(3,1920,1056))
print(sum([p.numel() for p in net.parameters()]))