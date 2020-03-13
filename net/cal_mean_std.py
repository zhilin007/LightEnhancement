from PIL import Image
import numpy as np
import os,tools
path='/Users/wangzhilin/Downloads/data/LightEnchancement/LOL/eval/high'
from torchvision.transforms import transforms as tfs
# print(imgs)
#cal image mean and stddev
def cal_mean_stddev(path,format):
    width=1600
    height=1200
    files=os.listdir(path)
    imgs=[]
    for file in files :
        if file.find(format)!=-1:
            # img=tfs.Resize((height,width))(Image.open(os.path.join(path,file)))
            img=Image.open(os.path.join(path,file))
          #  img.show(
          #  import time
          #  time.sleep(100)
            img=np.array(img)
            imgs.append(img)
    print('total images :%d'%len(imgs))
    imgs=np.array(imgs)/255.0
    print('batch,width,height,channels(RGB)',imgs.shape)
    m=np.mean(imgs,axis=(0,1,2))
    print('mean :R G B',m)
# for std : std=std/len(imgs) 
    std=np.zeros(3)
    for i in range(len(imgs)):
        std+=np.std(imgs[i],axis=(0,1))
    stddev=std/len(imgs)
    print('stddev:R G B',stddev)

def cal_mean_stddev_YUV(path,format):
    width=1600
    height=1200
    files=os.listdir(path)
    imgs=[]
    for file in files :
        if file.find(format)!=-1:
            # img=tfs.Resize((height,width))(Image.open(os.path.join(path,file)))
            img=Image.open(os.path.join(path,file))
          #  img.show(
          #  import time
          #  time.sleep(100)
            img=np.array(img)
            imgs.append(img)
    print('total images :%d'%len(imgs))
    imgs=np.array(imgs)/255.0
    imgs=tools.rgb2yuv(imgs)
    print('batch,width,height,channels(RGB)',imgs.shape)
    m=np.mean(imgs,axis=(0,1,2))
    print('mean :Y U V',m)
# for std : std=std/len(imgs) 
    std=np.zeros(3)
    for i in range(len(imgs)):
        std+=np.std(imgs[i],axis=(0,1))
    stddev=std/len(imgs)
    print('stddev:Y U V',stddev)
if __name__ =='__main__':
    print(path)
    cal_mean_stddev(path,'png')
