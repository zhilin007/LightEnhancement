import torch,os,sys,torchvision,argparse
import torchvision.transforms as tfs
from torchvision.utils import make_grid
import time,math,os
import numpy as np
from torch.backends import cudnn
from torch import optim
import matplotlib.pyplot as plt
import torch,warnings
from torch import nn
# from tensorboardX import SummaryWriter
import torchvision.utils as vutils
warnings.filterwarnings('ignore')


parser=argparse.ArgumentParser()
#for train
parser.add_argument('--data',type=str,default='/data')
parser.add_argument('--steps',type=int,default=100000)
parser.add_argument('--device',type=str,default='cuda')
parser.add_argument('--eval_step',type=int,default=1000)
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--pth',type=str,default='',help='model name to save')
parser.add_argument('--net',type=str,default='')
parser.add_argument('--crop_size',type=int,default=256,help='crop_size')
parser.add_argument('--print',action='store_true')
parser.add_argument('--bs',type=int,default=1)
parser.add_argument('--l1loss',action='store_true',help='l1 loss func')
parser.add_argument('--mseloss',action='store_true',help='l2 loss')
parser.add_argument('--ssimloss',action='store_true',help='ssim loss for train')

#for tensorboard
parser.add_argument('--log_dir',type=str,default='logs')

#for test
#parser.add_argument('--save_fig',action='store_true',help='save fig in test_dataset.py')
#parser.add_argument('--save_grid',action='store_true',help='save grid fig in test_dataset.py')

#for data
parser.add_argument('--divisor',type=int,default=1,help='input fig must can be divisible by it')

opt=parser.parse_args()
cwd='/'.join(os.path.realpath(__file__).split('/')[:-2])
#/home/zhilin007/VS/derain/
opt.data=os.path.join(cwd,'data')
opt.pth=os.path.join(cwd,'net','best_pth',opt.pth)

if opt.pth.find('.pth') ==-1:
    opt.pth+='.pth'
step_save_pth=opt.pth.replace('best_pth','step_pth')
model_name=opt.pth.split(os.sep)[-1].split('.')[0]

if opt.print:
    print('\n'.join(['%s:%s' % item for item in opt.__dict__.items()]) )
print('model:',opt.pth)
