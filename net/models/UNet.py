import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
	def __init__(self,in_dim,out_dim,bn=False):
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
	def __init__(self,in_dim,out_dim,bn=False):
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
		x1=self.conv1(x)#64
		
		x2=self.maxpool(x1)#32
		x2=self.conv2(x2)#32

		x3=self.maxpool(x2)#16
		x3=self.conv3(x3)#16

		x4=self.maxpool(x3)#8
		x4=self.conv4(x4)#8

		x5=self.maxpool(x4)#4
		x5=self.conv5(x5)#4
		#decoder
		d5=self.up5(x5)#8
		d5=self.up5_conv(torch.cat([x4,d5],dim=1))#8

		d4=self.up4(d5)#16
		d4=self.up4_conv(torch.cat([x3,d4],dim=1))

		d3=self.up3(d4)
		d3=self.up3_conv(torch.cat([x2,d3],dim=1))

		d2=self.up2(d3)
		d2=self.up2_conv(torch.cat([x1,d2],dim=1))

		d1=self.conv1x1(d2)
		return d1

class UNet64(nn.Module):#feature 不变
	def __init__(self,color=4,out_color=3,feature=64,bn=False):
		#depth=5 2^(5-1)=16
		super(UNet64,self).__init__()
		self.maxpool=nn.MaxPool2d(kernel_size=2,stride=2)
		self.conv1=Block(color,feature,bn)
		self.conv2=Block(feature,feature,bn)
		self.conv3=Block(feature,feature,bn)
		self.conv4=Block(feature,feature,bn)
		self.conv5=Block(feature,feature,bn)
		self.up5=Up(feature,feature,bn)
		self.up5_conv=Block(feature*2,feature,bn)
		self.up4=Up(feature,feature,bn)
		self.up4_conv=Block(feature*2,feature,bn)
		self.up3=Up(feature,feature,bn)
		self.up3_conv=Block(feature*2,feature,bn)
		self.up2=Up(feature,feature,bn)
		self.up2_conv=Block(feature*2,feature,bn)
		self.conv1x1=nn.Conv2d(feature,out_color,kernel_size=1,stride=1,padding=0)
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
class UNet_Depth(nn.Module):
	def __init__(self,in_color=4,out_color=3,feature=64,depth=5):
		#downsample :depth-1
		#receptive field of encoder:
		#depth=5 :140
		#depth=6 :284
		#depth=7 :572
		#depth=8 :1148
		super(UNet_Depth,self).__init__()
		self.depth=depth
		self.maxpool=nn.MaxPool2d(kernel_size=2,stride=2)

		self.encoder=nn.ModuleList()
		self.encoder.append(Block(in_color,feature))
		for _ in range(1,depth):
			self.encoder.append(Block(feature,feature))

		self.up=nn.ModuleList()
		for _ in range(1,depth):
			self.up.append(Up(feature,feature))

		self.decoder=nn.ModuleList()
		for _ in range(1,depth):
			self.decoder.append(Block(feature*2,feature))

		self.conv1x1=nn.Conv2d(feature,out_color,kernel_size=1,stride=1,padding=0)
	def forward(self,x):
		features=[]
		x=self.encoder[0](x)
		features.append(x)
		for i in range(1,self.depth):
			x=self.maxpool(x)
			x=self.encoder[i](x)
			features.append(x)
		print(len(features))
		features=features[::-1]
		x=features[0]
		for i in range(1,self.depth):
			x=self.up[i-1](x)
			x=self.decoder[i-1](torch.cat([features[i],x],dim=1))
		out=self.conv1x1(x)
		return out



if __name__ == "__main__":
	x=torch.zeros([1,4,256,256])
	net=UNet_Depth(depth=9)
	y=net(x)
	print(y.size())
	print(sum([p.numel() for p in net.parameters()]))
