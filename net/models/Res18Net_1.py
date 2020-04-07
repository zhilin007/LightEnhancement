import torch,torch.nn as nn
import torch.nn.functional as F
'''
dims=[16,24,32,48,64],最后一直上采样+conv到x 不能收敛

dims=[32,64,64,64,64]

如果可以的话，用input替代x1

'''

class BasicBlock(nn.Module):
	expansion = 1
	def __init__(self,in_dim,out_dim,stride=1,downsample=None):
		super(BasicBlock,self).__init__()
		self.conv1=nn.Conv2d(in_dim,out_dim,3,stride,1)
		self.relu=nn.ReLU(inplace=True)
		self.conv2=nn.Conv2d(out_dim,out_dim,3,1,1)
		self.downsample=downsample
	def forward(self,x):
		residual=x
		out=self.relu(self.conv1(x))
		out=self.conv2(out)
		if self.downsample is not None:
			residual=self.downsample(residual)
		out=out+residual
		out=self.relu(out)
		return out
class ResNet18(nn.Module):#depth=5
	def __init__(self,dims,color=4,layers=[2,2,2,2]):
		self.inplanes=dims[0]
		super(ResNet18,self).__init__()
		self.conv1 = nn.Conv2d(color, self.inplanes, kernel_size=3, stride=1, padding=1,
							bias=True)
		self.relu = nn.ReLU(inplace=True)
		block=BasicBlock
		self.layer1 = self._make_layer(block, dims[1], layers[0],stride=2)
		self.layer2 = self._make_layer(block, dims[2], layers[1], stride=2)
		self.layer3 = self._make_layer(block, dims[3], layers[2], stride=2)
		self.layer4 = self._make_layer(block, dims[4], layers[3], stride=2)
	def _make_layer(self, block, planes, blocks, stride=1):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				nn.Conv2d(self.inplanes, planes * block.expansion,
						kernel_size=1, stride=stride, bias=False))
		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample))
		self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.inplanes, planes))
		return nn.Sequential(*layers)
	def forward(self,x1):
		x1=self.relu(self.conv1(x1))
		x2=self.layer1(x1)
		x4=self.layer2(x2)
		x8=self.layer3(x4)
		x16=self.layer4(x8)
		return x1,x2,x4,x8,x16
class _Upsample(nn.Module):
	def __init__(self, in_dim, skip_in_dim, out_dim):
		super(_Upsample, self).__init__()
		self.skip_conv  = nn.Conv2d(skip_in_dim,in_dim,1)
		self.blend_conv = nn.Conv2d(in_dim,out_dim,3,1,1)
		self.relu=nn.ReLU()
		self.upsampling_method = F.interpolate
	def forward(self,skip,x):
		x=self.upsampling_method(x,scale_factor=2)
		skip=self.skip_conv(skip)
		out=self.relu(self.blend_conv(skip+x))
		return out
class Decoder(nn.Module):
	def __init__(self,in_dims,out_dims):
		super(Decoder,self).__init__()
		self.up1=_Upsample(in_dims[-1],in_dims[-2],out_dims[-1])
		self.up2=_Upsample(out_dims[-1],in_dims[-3],out_dims[-2])
		self.up3=_Upsample(out_dims[-2],in_dims[-4],out_dims[-3])
		self.up4=_Upsample(out_dims[-3],in_dims[-5],out_dims[-4])
	def forward(self,x16,x8,x4,x2,x1):
		out=self.up1(x8,x16)#x8
		out=self.up2(x4,out)#x4
		out=self.up3(x2,out)#x2
		out=self.up4(x1,out)#x1
		return out
class Res18Net1(nn.Module):
	def __init__(self,incolor=4,outcolor=3,dims=[32,64,64,64,64]):
		super(Res18Net1,self).__init__()
		self.encoder=ResNet18(dims=dims,color=incolor)
		self.decoder=Decoder(in_dims=dims,out_dims=dims[:-1])
		self.post=nn.Sequential(
			nn.Conv2d(dims[0],outcolor,1))
	def forward(self,x):
		x1,x2,x4,x8,x16=self.encoder(x)
		out=self.decoder(x16,x8,x4,x2,x1)
		out=self.post(out)
		return out+x[:,:3,:,:]
if __name__ == "__main__":
	x=torch.zeros([1,4,256,256])
	net=Res18Net1()
	y=net(x)
	print(y.size())
	print(sum([p.numel() for p in net.parameters()]))

	
