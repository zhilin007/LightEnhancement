import torch,os,cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.transforms as tfs
import torch.nn.functional as F
from torchvision.utils import make_grid

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
def get_illumination(ts,m=True):
	#ts:[N,C,H,W]->[N,1,1,1]
	# y=tensor_rgb2yuv(ts)[:,:1,:,:]
	R=ts[:,0,:,:];G=ts[:,1,:,:];B=ts[:,2,:,:]
	Y=0.299*R+0.587*G+0.114*B
	Y=Y[:,None,:,:]
	if m:
		i=torch.mean(Y,(2,3),keepdim=True)
		return i
	else :
		return Y
def def_illumination(arr,shape):
	#arr:list of illumination range 0-1
	#shape:list of [1,1,H,W]
	ts=[]
	for a in arr:
		t=torch.zeros(shape)+a
		ts.append(t)
	return ts
def unNorm(tensor,mean=[0.0629,0.0606,0.0558],std=[0.0430,0.0412,0.0425]):
	t=tensor.clone()
	for i in range(len(mean)):
		t[:,i,:,:]=t[:,i,:,:]*std[i]+mean[i]
	return t
def get_att_ill():
	from data_utils import get_eval_loader,tensorShow
	loader=get_eval_loader()
	for i ,(x,y) in enumerate(loader):
		# x_i=get_illumination(x,False)
		x=unNorm(x)
		y_Y=get_illumination(y,False)
		maxy=torch.max(y,dim=1,keepdim=True)[0]
		maxx=torch.max(x,dim=1,keepdim=True)[0]
		a=(torch.abs(maxy-maxx)/maxy)
		tensorShow([x,a,y,y_Y],['x','a','y','yY'])
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
def grid_sample_bilateral():
	input=os.getcwd()+'/data/LOL/eval/low/1.png';input=Image.open(input)
	gt=os.getcwd()+'/data/LOL/eval/high/1.png';gt=Image.open(gt)
	# gt.show()
	gt=gt.resize((150,100))#x4
	# gt.resize((600,400),Image.BILINEAR).show()#不管用哪种方式，其信息损失都很大
	input=tfs.ToTensor()(input)[None,::];gt=tfs.ToTensor()(gt)[None,::]
	# grid=tfs.Normalize(mean=[0.0629,0.0606,0.0558],std=[0.0430,0.0412,0.0425])(input)[None,::]
	# grid=get_illumination(grid,m=False).permute(0,2,3,1)
	# out=F.grid_sample(gt,torch.cat([grid,grid],dim=-1)) 和想的一样，和联合双边滤波不是一个东西

	# N,C,H,W=input.size()
	# gh,gw=torch.meshgrid([torch.range(0,H),torch.range(0,W)])
	# gh = gh.float().repeat(N, 1, 1).unsqueeze(3) / (H-1) * 2 - 1 # norm to [-1,1] NxHxWx1
	# gw = gw.float().repeat(N, 1, 1).unsqueeze(3) / (W-1) * 2 - 1 # norm to [-1,1] NxHxWx1
	# out=F.grid_sample(gt,torch.cat([gw,gh],dim=-1)) 就是bilinear
	

	



	
	
	

if __name__ == "__main__":
	# ts=torch.ones([1,1,1,1])
	# print(ts.item())
	# get_att_ill()
	grid_sample_bilateral()
	
