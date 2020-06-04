import torch.utils.data as data
import torchvision.transforms as tfs
from torchvision.transforms import functional as FF
import os,sys
from tqdm import tqdm
sys.path.append('.')
sys.path.append('..')
import numpy as np
import torch
import random , glob
from PIL import Image
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from torchvision.utils import make_grid
from metrics import *
from option import opt,cwd
from tools import pad_pil
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
def tensorShow(tensors,titles=None):
        '''
        t:BCWH
        '''
        fig=plt.figure(figsize=(20,20))
        for tensor,tit,i in zip(tensors,titles,range(len(tensors))):
            img = make_grid(tensor)
            npimg = img.numpy()
            ax = fig.add_subplot(221+i)
            ax.imshow(np.transpose(npimg, (1, 2, 0)))
            ax.set_title(tit)
        plt.show()
class EHDataset(data.Dataset):
    def __init__(self,path,mode,format):
        super(EHDataset,self).__init__()
        self.mode=mode
        ins=glob.glob(os.path.join(path,mode,'low','*.'+format))
        self.lows=[]
        self.highs=[]
        for im in tqdm(ins):
            low=Image.open(im);self.lows.append(low)
            high=Image.open(im.replace('low','high'));self.highs.append(high)
    def __getitem__(self, index):
        low=self.lows[index]
        high=self.highs[index]
        if self.mode=='train':
            i,j,h,w=tfs.RandomCrop.get_params(low,output_size=(opt.crop_size,opt.crop_size))
            low=FF.crop(low,i,j,h,w)
            high=FF.crop(high,i,j,h,w)
        if self.mode!='train':#must can be divisible by opt.divisor
            low=pad_pil(low,opt.divisor)
            high=pad_pil(high,opt.divisor)
        low,high=self.augData(low.convert('RGB'),high.convert('RGB'))
        return low,high
    def augData(self,data,target):
        if self.mode=='train':
            rand_hor=random.randint(0,1)
            rand_ver=random.randint(0,1)
            rand_rot=random.randint(0,3)
            data=tfs.RandomHorizontalFlip(rand_hor)(data)
            target=tfs.RandomHorizontalFlip(rand_hor)(target)
            data=tfs.RandomVerticalFlip(rand_ver)(data)
            target=tfs.RandomVerticalFlip(rand_ver)(target)
            if rand_rot:
                data=FF.rotate(data,90*rand_rot)
                target=FF.rotate(target,90*rand_rot)
        data=tfs.ToTensor()(data)
        target=tfs.ToTensor()(target)
        data=tfs.Normalize(mean=[0.0629,0.0606,0.0558],std=[0.0430,0.0412,0.0425])(data)
        return  data ,target
    def __len__(self):
        return len(self.lows)

class AttentionGuidedDataset(data.Dataset):#dir:dataset/test/(enhance|dark|lowlight)/*.png
    def __init__(self,path,mode,subset,format):#subset:dark/lowlight
        super(AttentionGuidedDataset,self).__init__()
        self.mode=mode
        ins=glob.glob(os.path.join(path,mode,subset,'*.'+format))
        self.lows=[]
        self.highs=[]
        for im in tqdm(ins):
            self.lows.append(im)
            self.highs.append(im.replace(subset,'enhance'))
            # low=Image.open(im);self.lows.append(low)
            # high=Image.open(im.replace(subset,'enhance'));self.highs.append(high)
    def __getitem__(self, index):
        low=Image.open(self.lows[index])
        high=Image.open(self.highs[index])
        minWid=min(low.size)
        if self.mode=='train':
            if opt.crop_size>minWid:
                crop_size=minWid-minWid%opt.divisor
                i,j,h,w=tfs.RandomCrop.get_params(low,output_size=(crop_size,crop_size))#不够crop的话，就用稍小的size来crop
            else :
                i,j,h,w=tfs.RandomCrop.get_params(low,output_size=(opt.crop_size,opt.crop_size))
            low=FF.crop(low,i,j,h,w)
            high=FF.crop(high,i,j,h,w)
        if self.mode!='train':#must can be divisible by opt.divisor
            low=pad_pil(low,opt.divisor)
            high=pad_pil(high,opt.divisor)
        low,high=self.augData(low.convert('RGB'),high.convert('RGB'))
        return low,high
    def augData(self,data,target):
        if self.mode=='train':
            rand_hor=random.randint(0,1)
            rand_ver=random.randint(0,1)
            rand_rot=random.randint(0,3)
            data=tfs.RandomHorizontalFlip(rand_hor)(data)
            target=tfs.RandomHorizontalFlip(rand_hor)(target)
            data=tfs.RandomVerticalFlip(rand_ver)(data)
            target=tfs.RandomVerticalFlip(rand_ver)(target)
            if rand_rot:
                data=FF.rotate(data,90*rand_rot)
                target=FF.rotate(target,90*rand_rot)
        data=tfs.ToTensor()(data)
        target=tfs.ToTensor()(target)
        data=tfs.Normalize(mean=[0.0629,0.0606,0.0558],std=[0.0430,0.0412,0.0425])(data)
        return  data ,target
    def __len__(self):
        return len(self.lows)


def get_train_loader(trainset=opt.trainset):
    path=os.path.join(opt.data,trainset)
    print(path)
    if trainset=='LOL':
        loader=DataLoader(EHDataset(path,'train','png'),batch_size=opt.bs,shuffle=True)
    if trainset=='AttentionGuided':
        loader=DataLoader(AttentionGuidedDataset(path,'train',opt.subset,'png'),batch_size=opt.bs,shuffle=True)
    return loader
def get_eval_loader(trainset=opt.trainset):
    path=os.path.join(opt.data,trainset)
    if trainset=='LOL':
        loader=DataLoader(EHDataset(path,'eval','png'),batch_size=1,shuffle=False)
    if trainset=='AttentionGuided':
        loader=DataLoader(AttentionGuidedDataset(path,'test',opt.subset,'png'),batch_size=1,shuffle=False)
    return loader
def get_eval_train_loader(trainset=opt.trainset):#查看是否overfit，和eval数据集一样有15张，从train集合的子集
    path=os.path.join(opt.data,trainset)
    if trainset=='LOL':
        loader=DataLoader(EHDataset(path,'eval_train','png'),batch_size=1,shuffle=False)
    if trainset=='AttentionGuided':
        loader=DataLoader(AttentionGuidedDataset(path,'eval_train',opt.subset,'png'),batch_size=1,shuffle=False)
    return loader

if __name__ == "__main__":
    #python data_utils.py --trainset=AttentionGuided --subset=dark lowlight
    from tools import get_illumination
    t_loader=get_train_loader()
    # t_loader=get_eval_loader()
    # t_loader=get_eval_train_loader()
    for _,(input,gt) in enumerate(t_loader):
        # ssim1=ssim(input,gt)
        i1=get_illumination(input)
        i=get_illumination(gt)
        tensorShow([input,gt],[f'{i1}',f'{i}'])
    # path='/Users/wangzhilin/Downloads/data/LightEnchancement/LOL'
    # da=EHDataset(path,'eval','png')
    pass

