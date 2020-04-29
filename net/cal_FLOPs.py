from torchstat import stat
from models import *
from torchvision import models
import torch
# net=EUUNet()
# net=SwiftNetSlim()
# net=EFFA(2,10)
net=Gen_Y_Swiftslim_BN2_Share(norm=True)
# print(net)
# x=torch.ones([1,3,224,224])
# net(x)
# stat(net,(4,1920,1080))
print(sum([p.numel() for p in net.parameters()]))