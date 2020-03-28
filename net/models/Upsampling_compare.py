import torch, torch.nn.functional as F
from torch import nn

class BilinearUp(nn.Module):
	def __init__(self):
		super(BilinearUp,self).__init__()
	def forward(self,x):
		x=F.interpolate(x,scale_factor=2,mode='bilinear',align_corners=False)
		x=F.interpolate(x,scale_factor=2,mode='bilinear',align_corners=False)
		x=F.interpolate(x,scale_factor=2,mode='bilinear',align_corners=False)
		x=F.interpolate(x,scale_factor=2,mode='bilinear',align_corners=False)
		return x
class DeconvUp(nn.Module):
	def __init__(self):
		super(DeconvUp,self).__init__()
		self.up=nn.Sequential(
			nn.ConvTranspose2d(3,3,3,2,1,1),
			nn.ConvTranspose2d(3,3,3,2,1,1),
			nn.ConvTranspose2d(3,3,3,2,1,1),
			nn.ConvTranspose2d(3,3,3,2,1,1),
		)
	def forward(self,x):
		return self.up(x)
class SubPixel(nn.Module):
	def __init__(self):
		super(SubPixel,self).__init__()
		self.up=nn.Sequential(
			nn.Conv2d(3,12,1),
			nn.PixelShuffle(2),
			nn.Conv2d(3,12,1),
			nn.PixelShuffle(2),
			nn.Conv2d(3,12,1),
			nn.PixelShuffle(2),
			nn.Conv2d(3,12,1),
			nn.PixelShuffle(2)
		)
	def forward(self,x):
		return self.up(x)
if __name__ == "__main__":
	import time
	steps=1000
	x=torch.ones([1,3,256,256]).to('cuda:1')
	nets=[BilinearUp().to('cuda:1'),DeconvUp().to('cuda:1'),SubPixel().to('cuda:1')]
	for net in nets:
		sep_time=time.time()
		for i in range(steps):
			print(f'\r  {i}/{steps}',end='',flush=True)
			with torch.no_grad():
				_=net(x)
		time_interval=time.time()-sep_time
		print(time_interval)
