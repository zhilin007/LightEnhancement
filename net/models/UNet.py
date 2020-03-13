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

class UNet(nn.Module):
	def __init__(self,color=3,feature=64,bn=False):
		#depth=4 2^4=16
		super(UNet,self).__init__()
		self.maxpool=nn.MaxPool2d(kernel_size=2,stride=2)
		self.conv1=Block(4,feature,bn)
		self.conv2=Block(feature,feature*2,bn)
		self.conv3=Block(feature*2,feature*4,bn)
		self.conv4=Block(feature*4,feature*8,bn)
		self.conv5=Block(feature*8,feature*16,bn)
		
		self.up5=Up(feature*16,feature*8,bn)
		self.up5_conv=Block(feature*16,feature*8,bn)
		self.up4=Up(feature*8,feature*4,bn)
		self.up4_conv=Block(feature*8,feature*4,bn)
		self.up3=Up(feature*4,feature*2,bn)
		self.up3_conv=Block(feature*4,feature*2,bn)
		self.up2=Up(feature*2,feature*1,bn)
		self.up2_conv=Block(feature*2,feature*1,bn)

		self.conv1x1=nn.Conv2d(feature,color,kernel_size=1,stride=1,padding=0)
	def forward(self,x):
		#encoder
		x1=self.conv1(x)
		
		x2=self.maxpool(x1)
		x2=self.conv2(x2)

		x3=self.maxpool(x2)
		x3=self.conv3(x3)

		x4=self.maxpool(x3)
		x4=self.conv4(x4)

		x5=self.maxpool(x4)
		x5=self.conv5(x5)
		#decoder
		d5=self.up5(x5)
		d5=self.up5_conv(torch.cat([x4,d5],dim=1))

		d4=self.up4(d5)
		d4=self.up4_conv(torch.cat([x3,d4],dim=1))

		d3=self.up3(d4)
		d3=self.up3_conv(torch.cat([x2,d3],dim=1))

		d2=self.up2(d3)
		d2=self.up2_conv(torch.cat([x1,d2],dim=1))

		d1=self.conv1x1(d2)
		return d1

if __name__ == "__main__":
	net=UNet()
	print(sum([p.numel() for p in net.parameters()]))
