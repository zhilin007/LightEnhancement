from math import exp
import math
import numpy as np

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from math import exp
import math
import numpy as np
from skimage import measure as ms
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from  torchvision.transforms import ToPILImage
from  torchvision.transforms import ToTensor
import ssim_loss

def ssim(im1,im2):
    if im1.size()!=im2.size():
        print(im1.size(),im2.size())
    pred=im1[0];gt=im2[0]
    pred=np.transpose(pred.clamp(0,1).cpu().numpy(),(1,2,0))
    gt=np.transpose(gt.clamp(0,1).cpu().numpy(),(1,2,0))
    return ms.compare_ssim(pred,gt,multichannel=True,data_range=1)
def psnr(im1,im2):
    pred=im1[0];gt=im2[0]
    pred=np.transpose(pred.clamp(0,1).cpu().numpy(),(1,2,0))
    gt=np.transpose(gt.clamp(0,1).cpu().numpy(),(1,2,0))
    return ms.compare_psnr(pred,gt,data_range=1)

if __name__ == "__main__":
    pass
 