#从主干网络中选出来最好的
import torch,torch.nn as nn
import torch.nn.functional as F
from .GuidedFilter.guided_filter import ConvGuidedFilter
'''
特征变为[16,32,64,64]原来[64,128,256,512]
decoder使用bn而不是instanceNorm
Adaptive Norm改动
'''

class BasicBlock(nn.Module):
	expansion = 1
	def __init__(self,in_dim,out_dim,stride=1,downsample=None,norm=False):
		super(BasicBlock,self).__init__()
		self.norm=norm
		self.conv1=nn.Conv2d(in_dim,out_dim,3,stride,1)
		self.norm1 = nn.InstanceNorm2d(out_dim) if self.norm else None
		self.relu=nn.ReLU(inplace=True)
		self.conv2=nn.Conv2d(out_dim,out_dim,3,1,1)
		self.norm2 = nn.InstanceNorm2d(out_dim) if self.norm else None
		self.downsample=downsample
	def forward(self,x):
		residual=x
		if self.norm:
			out=self.relu(self.norm1(self.conv1(x)))
			out=self.norm2(self.conv2(out))
		else :
			out=self.relu(self.conv1(x))
			out=self.conv2(out)
		if self.downsample is not None:
			residual=self.downsample(residual)
		skip=out+residual
		out=self.relu(skip)#改了。。
		return skip,out
class ResNet18(nn.Module):
	def __init__(self,color=4,layers=[2,2,2,2],features=[16,32,64,64],norm=False):
		self.inplanes=features[0]
		self.norm=norm
		super(ResNet18,self).__init__()
		self.conv1 = nn.Conv2d(color, features[0], kernel_size=7, stride=2, padding=3,
							bias=True)
		self.norm1=nn.InstanceNorm2d(features[0]) if self.norm else None
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		block=BasicBlock
		self.layer1 = self._make_layer(block, features[0], layers[0])
		self.layer2 = self._make_layer(block, features[1], layers[1], stride=2)
		self.layer3 = self._make_layer(block, features[2], layers[2], stride=2)
		self.layer4 = self._make_layer(block, features[3], layers[3], stride=2)
	def _make_layer(self, block, planes, blocks, stride=1):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				nn.Conv2d(self.inplanes, planes * block.expansion,
						kernel_size=1, stride=stride, bias=False))
		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample,norm=self.norm))
		self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.inplanes, planes,norm=self.norm))
		return nn.Sequential(*layers)
	def forward_layer(self, x, layer):
		skip = None
		for l in layer:
			x = l(x)
			if isinstance(x, tuple):
				skip,x = x
		return skip,x
	def forward(self,x1):
		x2=self.relu(self.conv1(x1))
		x4=self.maxpool(x2)
		x4,x=self.forward_layer(x4,self.layer1)
		x8,x=self.forward_layer(x,self.layer2)
		x16,x=self.forward_layer(x,self.layer3)
		x32,x=self.forward_layer(x,self.layer4)
		return x1,x2,x4,x8,x16,x32


upsample = lambda x, size: F.interpolate(x, size, mode='bilinear', align_corners=False)
class ReluConv(nn.Sequential):
	def __init__(self, in_dim, out_dim,k=3,norm=False):
		super(ReluConv, self).__init__()
		if norm:
			self.add_module('norm', nn.BatchNorm2d(in_dim))#使用BN
		self.add_module('relu', nn.ReLU(inplace=False))
		self.add_module('conv', nn.Conv2d(in_dim, out_dim, kernel_size=k, padding=k//2))
class SpatialPyramidPooling(nn.Module):
	def __init__(self, num_maps_in, num_levels=3, bt_size=128, out_size=128,grids=(8, 4, 2, 1), square_grid=False,fixed_size=None,norm=False):
		super(SpatialPyramidPooling, self).__init__()
		level_size = out_size // num_levels
		self.fixed_size = fixed_size
		self.grids = grids
		if self.fixed_size:
			ref = min(self.fixed_size)
			self.grids = list(filter(lambda x: x <= ref, self.grids))
		self.square_grid = square_grid
		self.upsampling_method = upsample
		if self.fixed_size is not None:
			self.upsampling_method = lambda x, size: F.interpolate(x, mode='nearest', size=fixed_size)
		self.spp = nn.Sequential()
		self.spp.add_module('spp_bn', ReluConv(num_maps_in, bt_size, k=1,norm=norm))
		num_features = bt_size
		final_size = num_features
		for i in range(num_levels):
			final_size += level_size
			self.spp.add_module('spp' + str(i),ReluConv(num_features, level_size, k=1,norm=norm))
		self.spp.add_module('spp_fuse',ReluConv(final_size, out_size, k=1,norm=norm))
	def forward(self, x):
		levels = []
		target_size = self.fixed_size if self.fixed_size is not None else x.size()[2:4]
		ar = target_size[1] / target_size[0]
		x = self.spp[0].forward(x)
		levels.append(x)
		num = len(self.spp) - 1
		for i in range(1, num):
			if not self.square_grid:
				grid_size = (self.grids[i - 1], max(1, round(ar * self.grids[i - 1])))
				x_pooled = F.adaptive_avg_pool2d(x, grid_size)#input,output_size
			else:
				x_pooled = F.adaptive_avg_pool2d(x, self.grids[i - 1])
			level = self.spp[i].forward(x_pooled)
			level = self.upsampling_method(level, target_size)
			levels.append(level)
		x = torch.cat(levels, 1)
		x = self.spp[-1].forward(x)
		return x
class _Upsample(nn.Module):
	def __init__(self, num_maps_in, skip_maps_in, num_maps_out, k=3,norm=False):
		super(_Upsample, self).__init__()
		self.skip_conv = ReluConv(skip_maps_in, num_maps_in, k=1,norm=norm)
		self.blend_conv = ReluConv(num_maps_in, num_maps_out, k=k,norm=norm)
		self.upsampling_method = upsample
	def forward(self,skip,x):
		skip = self.skip_conv.forward(skip)
		skip_size = skip.size()[2:4]
		x = self.upsampling_method(x, skip_size)#x.size()->skip.size()!!
		x = x + skip #element add
		x = self.blend_conv.forward(x)
		return x
class Decoder(nn.Module):#out_dim=128 decoder to x4
	def __init__(self,features=[16,32,64,64],in_dim=64,norm=False):
		super(Decoder,self).__init__()
		self.up1=_Upsample(in_dim,features[-2],features[-2],norm=norm)#original outdim=128
		self.up2=_Upsample(features[-2],features[-3],features[-3],norm=norm)
		self.up3=_Upsample(features[-3],features[-4],features[-4],norm=norm)
	def forward(self,x32,x16,x8,x4):
		out=self.up1(x16,x32)#x16
		out=self.up2(x8,out)#x8
		out=self.up3(x4,out)#x4
		return out
class AdaptiveNorm(nn.Module):
	def __init__(self, n):
		super(AdaptiveNorm, self).__init__()
		self.w_0 = nn.Parameter(torch.Tensor([0.0]))#original 1
		self.w_1 = nn.Parameter(torch.Tensor([1.0]))#original 0
		self.bn  = nn.BatchNorm2d(n, momentum=0.999, eps=0.001)
	def forward(self, x):
		return self.w_0 * x + self.w_1 * self.bn(x)

class Backbone7x7(nn.Module):
	def __init__(self,incolor=4,outcolor=3,features=[16,32,64,64],norm=False):
		super(Backbone7x7,self).__init__()
		self.sppdim=64 #original 128
		self.encoder=ResNet18(incolor,features=features,norm=norm)#original features:[64,128,256,512]
		self.spp=SpatialPyramidPooling(features[-1],out_size=self.sppdim,norm=norm)
		self.decoder=Decoder(features,in_dim=self.sppdim,norm=norm)
		self.post=ReluConv(features[-4],outcolor,1,norm=norm)
		self.filter = ConvGuidedFilter(1, norm=AdaptiveNorm)
		self.guided_map = nn.Sequential(
			nn.Conv2d(3, 16, 1, bias=False),
			AdaptiveNorm(16),
			nn.ReLU(inplace=True),#改动！！！不是lrelu了
			nn.Conv2d(16, 3, 1)
		)
	def forward(self,x):
		image_size=x.size()[2:4]
		x1,x2,x4,x8,x16,x32=self.encoder(x)
		x32=self.spp(x32)
		out_x4=self.decoder(x32,x16,x8,x4)
		out_x4=self.post(out_x4)
		# out=upsample(out_x4,image_size)
		x_h=x[:,:3,::];x_l=F.interpolate(x_h,scale_factor=0.25)
		out=self.filter(self.guided_map(x_l),out_x4,self.guided_map(x_h))
		return out_x4,out

if __name__ == "__main__":

	#devisor=32
	x=torch.zeros([1,4,160,160])
	net=Backbone7x7(norm=True)
	print(sum([p.numel() for p in net.parameters()]))
	# net(x)
	print(net)

	
