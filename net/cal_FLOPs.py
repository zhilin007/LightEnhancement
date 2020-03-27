from torchstat import stat
from models import *
from torchvision import models
import torch
# net=EUUNet()
# net=SwiftNetSlim()
net=FullConv_SwiftNet()
# print(net)
# x=torch.ones([1,3,224,224])
# net(x)
stat(net,(4,1920,1056))