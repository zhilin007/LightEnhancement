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
class ConvGuidedFilter(nn.Module):#为了对比时效修改过
	def __init__(self, radius=1, norm=nn.BatchNorm2d):
		super(ConvGuidedFilter, self).__init__()
		self.box_filter = nn.Conv2d(3, 3, kernel_size=3, padding=radius, dilation=radius, bias=False, groups=3)
		self.conv_a = nn.Sequential(nn.Conv2d(6, 32, kernel_size=1, bias=False),
									norm(32),
									nn.ReLU(inplace=True),
									nn.Conv2d(32, 32, kernel_size=1, bias=False),
									norm(32),
									nn.ReLU(inplace=True),
									nn.Conv2d(32, 3, kernel_size=1, bias=False))
		self.box_filter.weight.data[...] = 1.0
		self.x_hr=torch.zeros([1,3,256*16,256*16]).to('cuda:2')
	def forward(self, x_lr):
		x_hr=self.x_hr
		y_lr=x_lr
		_, _, h_lrx, w_lrx = x_lr.size()
		_, _, h_hrx, w_hrx = x_hr.size()
		N = self.box_filter(x_lr.data.new().resize_((1, 3, h_lrx, w_lrx)).fill_(1.0))
		## mean_x
		mean_x = self.box_filter(x_lr)/N
		## mean_y
		mean_y = self.box_filter(y_lr)/N
		## cov_xy
		cov_xy = self.box_filter(x_lr * y_lr)/N - mean_x * mean_y
		## var_x
		var_x  = self.box_filter(x_lr * x_lr)/N - mean_x * mean_x
		## A
		A = self.conv_a(torch.cat([cov_xy, var_x], dim=1))
		## b
		b = mean_y - A * mean_x
		## mean_A; mean_b
		mean_A = F.interpolate(A, (h_hrx, w_hrx), mode='bilinear', align_corners=True)
		mean_b = F.interpolate(b, (h_hrx, w_hrx), mode='bilinear', align_corners=True)
		return mean_A * x_hr + mean_b

if __name__ == "__main__":
	import time
	steps=1000
	x=torch.ones([1,3,256,256]).to('cuda:2')
	# nets=[BilinearUp().to('cuda:1'),SubPixel().to('cuda:1'),DeconvUp().to('cuda:1')]
	#3.6s | 19.24s | 非常慢x100
	#不需要运算的还是很快的
	nets=[BilinearUp().to('cuda:2'),ConvGuidedFilter().to('cuda:2')]
	for net in nets:
		torch.cuda.empty_cache()
		sep_time=time.time()
		for i in range(steps):
			print(f'\r  {i}/{steps}',end='',flush=True)
			with torch.no_grad():
				_=net(x)
		time_interval=time.time()-sep_time
		print(time_interval)
