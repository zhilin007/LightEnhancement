'''
Efficient upsampling unet
downsampling使用conv2d
先融合再进行上采样
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
	def __init__(self,in_dim,out_dim,bn):
		super(Block,self).__init__()
		self.bn=bn
		self.conv1=nn.Conv2d(in_dim, out_dim, kernel_size=3,stride=1,padding=1)
		self.bn1=nn.BatchNorm2d(out_dim)
		self.relu1=nn.ReLU(inplace=True)
		self.conv2=nn.Conv2d(out_dim, out_dim, kernel_size=3,stride=1,padding=1)
		self.bn2=nn.BatchNorm2d(out_dim)
		self.relu2=nn.ReLU(inplace=True)
	def forward(self,x):
		if self.bn:
			out=self.relu1(self.bn1(self.conv1(x)))
			out=self.relu2(self.bn2(self.conv2(out)))
		else:
			out=self.relu1(self.conv1(x))
			out=self.relu2(self.conv2(out))
		return out
class Up(nn.Module):
	def __init__(self,in_dim,out_dim,bn):
		super(Up,self).__init__()
		self.bn=bn
		self.up=nn.Upsample(scale_factor=2)
		self.conv1=nn.Conv2d(in_dim,out_dim,kernel_size=3,stride=1,padding=1)
		self.bn1=nn.BatchNorm2d(out_dim)
		self.relu1=nn.ReLU(inplace=True)
	def forward(self,x):
		if self.bn:
			out=self.relu1(self.bn1(self.conv1(self.up(x))))
		else :
			out=self.relu1(self.conv1(self.up(x)))
		return out

class EUUNet(nn.Module):
	def __init__(self,color=4,out_color=3,feature=64,bn=False):
		#depth=4 2^4=16
		super(EUUNet,self).__init__()

		self.conv1=Block(color,feature,bn)
		self.conv1_down=nn.Conv2d(feature,feature,3,2,1)
		self.conv2=Block(feature,feature,bn)
		self.conv2_down=nn.Conv2d(feature,feature,3,2,1)
		self.conv3=Block(feature,feature,bn)
		self.conv3_down=nn.Conv2d(feature,feature,3,2,1)
		self.conv4=Block(feature,feature,bn)
		self.conv4_down=nn.Conv2d(feature,feature,3,2,1)
		self.conv5=Block(feature,feature,bn)
		
		self.up5=Up(feature,feature,bn)
		self.up5_conv=Block(feature,feature,bn)
		self.up4=Up(feature*2,feature,bn)
		self.up4_conv=Block(feature,feature,bn)
		self.up3=Up(feature*2,feature,bn)
		self.up3_conv=Block(feature,feature,bn)
		self.up2=Up(feature*2,feature,bn)
		self.up2_conv=Block(feature*2,feature,bn)
		self.conv1x1=nn.Conv2d(feature,out_color,kernel_size=1,stride=1,padding=0)
	def forward(self,x):
		#encoder
		x1=self.conv1(x)#64
		x2=self.conv1_down(x1)#32
		x2=self.conv2(x2)#32
		x3=self.conv2_down(x2)#16
		x3=self.conv3(x3)#16
		x4=self.conv3_down(x3)#8
		x4=self.conv4(x4)#8
		x5=self.conv4_down(x4)#4
		x5=self.conv5(x5)#4
		#decoder
		'''diff from Unet '''
		d5=self.up5(x5)#8
		d5=self.up5_conv(d5)#8
		d4=self.up4(torch.cat([x4,d5],dim=1))#16
		d4=self.up4_conv(d4)#16
		d3=self.up3(torch.cat([x3,d4],dim=1))#32
		d3=self.up3_conv(d3)#32
		d2=self.up2(torch.cat([x2,d3],dim=1))#64
		d2=self.up2_conv(torch.cat([x1,d2],dim=1))
		d1=self.conv1x1(d2)
		return d1

if __name__ == "__main__":
	x=torch.ones([1,4,64,64])
	net=EUUNet()
	net(x)