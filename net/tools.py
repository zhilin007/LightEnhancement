import torch,os,cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.transforms as tfs
import torch.nn.functional as F
def rgb2yuv(ts):
	#ts :ndarry :[N,H,W,C] 
	R=ts[:,:,:,0];G=ts[:,:,:,1];B=ts[:,:,:,2]
	Y=0.299*R+0.587*G+0.114*B
	U=0.564*(B-Y)
	V=0.713*(R-Y)
	Y=Y[:,:,:,None];U=U[:,:,:,None];V=V[:,:,:,None]
	ts=np.concatenate([Y,U,V],axis=-1)
	print(ts.shape)
	return ts
def tensor_rgb2yuv(ts):
	#ts :tensor :[N,C,H,W] 
	R=ts[:,0,:,:];G=ts[:,1,:,:];B=ts[:,2,:,:]
	Y=0.299*R+0.587*G+0.114*B
	U=0.564*(B-Y)
	V=0.713*(R-Y)
	Y=Y[:,None,:,:];U=U[:,None,:,:];V=V[:,None,:,:]
	ts=torch.cat([Y,U,V],dim=1)
	return ts
def tensor_yuv2rgb(ts):
	#ts :tensor :[N,C,H,W] 
	Y=ts[:,0,:,:];U=ts[:,1,:,:];V=ts[:,2,:,:]
	R=Y+1.403*V
	G = Y - 0.344*U - 0.714*V
	B = Y + 1.770*U
	R=R[:,None,:,:];G=G[:,None,:,:];B=B[:,None,:,:]
	ts=torch.cat([R,G,B],dim=1)
	return ts
def pad_pil(pil,divisor,pad=0):
	img=np.asarray(pil)
	H,W,C=img.shape
	Wk=0;Hk=0
	if W%divisor:
		Wk = divisor - W % divisor
	if H%divisor:
		Hk = divisor - H % divisor
	ap=np.pad(img,((divisor*pad, Hk+divisor*pad), (divisor*pad, Wk+divisor*pad),[0,0]),'reflect')
	pil=Image.fromarray(ap.astype('uint8')).convert('RGB')
	return pil
def get_illumination(ts):
	#ts:[N,C,H,W]->[N,1,1,1]
	# y=tensor_rgb2yuv(ts)[:,:1,:,:]
	R=ts[:,0,:,:];G=ts[:,1,:,:];B=ts[:,2,:,:]
	Y=0.299*R+0.587*G+0.114*B
	Y=Y[:,None,:,:]
	i=torch.mean(Y,(2,3),keepdim=True)
	return i
def def_illumination(arr,shape):
	#arr:list of illumination range 0-1
	#shape:list of [1,1,H,W]
	ts=[]
	for a in arr:
		t=torch.zeros(shape)+a
		ts.append(t)
	return ts
def unNorm(tensor,mean=[0.0629,0.0606,0.0558],std=[0.0430,0.0412,0.0425]):
	for i in range(len(mean)):
		tensor[:,i,:,:]=tensor[:,i,:,:]*std[i]+mean[i]
	return tensor
if __name__ == "__main__":
	ts=torch.ones([1,1,1,1])
	print(ts.item())
	
	
