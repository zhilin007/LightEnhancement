from torchstat import stat
from models import *

net=UNet()
stat(net,(4,1920,1072))