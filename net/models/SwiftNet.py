import torch,torch.nn as nn
import torch.nn.functional as F

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
		skip=out+residual
		out=self.relu(out)### bug!!!!!!!!!!!!!!!!! 应该是skip！！！！！！
		return skip,out
class ResNet18(nn.Module):
	def __init__(self,color=3,layers=[2,2,2,2]):
		self.inplanes=64
		super(ResNet18,self).__init__()
		self.conv1 = nn.Conv2d(color, 64, kernel_size=7, stride=2, padding=3,
							bias=True)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		block=BasicBlock
		self.layer1 = self._make_layer(block, 64, layers[0])
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
		self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
		self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
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
	def __init__(self, in_dim, out_dim,k=3):
		super(ReluConv, self).__init__()
		self.add_module('relu', nn.ReLU(inplace=True))
		self.add_module('conv', nn.Conv2d(in_dim, out_dim, kernel_size=k, padding=k//2))
class SpatialPyramidPooling(nn.Module):
	def __init__(self, num_maps_in, num_levels=3, bt_size=128, out_size=128,grids=(8, 4, 2, 1), square_grid=False,fixed_size=None):
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
		self.spp.add_module('spp_bn', ReluConv(num_maps_in, bt_size, k=1))
		num_features = bt_size
		final_size = num_features
		for i in range(num_levels):
			final_size += level_size
			self.spp.add_module('spp' + str(i),ReluConv(num_features, level_size, k=1))
		self.spp.add_module('spp_fuse',ReluConv(final_size, out_size, k=1))
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
	def __init__(self, num_maps_in, skip_maps_in, num_maps_out, k=3):
		super(_Upsample, self).__init__()
		self.skip_conv = ReluConv(skip_maps_in, num_maps_in, k=1)
		self.blend_conv = ReluConv(num_maps_in, num_maps_out, k=k)
		self.upsampling_method = upsample
	def forward(self,skip,x):
		skip = self.skip_conv.forward(skip)
		skip_size = skip.size()[2:4]
		x = self.upsampling_method(x, skip_size)#x.size()->skip.size()!!
		x = x + skip #element add
		x = self.blend_conv.forward(x)
		return x
class Decoder(nn.Module):#out_dim=128
	def __init__(self):
		super(Decoder,self).__init__()
		self.up1=_Upsample(128,256,128)
		self.up2=_Upsample(128,128,128)
		self.up3=_Upsample(128,64,128)
	def forward(self,x32,x16,x8,x4):
		out=self.up1(x16,x32)#x16
		out=self.up2(x8,out)#x8
		out=self.up3(x4,out)#x4
		return out
class SwiftNet(nn.Module):
	def __init__(self,incolor=4,outcolor=3):
		super(SwiftNet,self).__init__()
		self.encoder=ResNet18(incolor)
		self.spp=SpatialPyramidPooling(512)
		self.decoder=Decoder()
		self.post=ReluConv(128,outcolor,1)
	def forward(self,x):
		image_size=x.size()[2:4]
		x1,x2,x4,x8,x16,x32=self.encoder(x)
		x32=self.spp(x32)
		out_x4=self.decoder(x32,x16,x8,x4)
		out_x4=self.post(out_x4)
		out=upsample(out_x4,image_size)
		return out

if __name__ == "__main__":
	#最后output是1/4 上采样回去的，这对我们pixel-wise的评测标准非常不好!!!!
	#devisor=32
	x=torch.zeros([1,4,160,160])
	net=SwiftNet()
	# print(sum([p.numel() for p in net.parameters()]))
	net(x)

	
