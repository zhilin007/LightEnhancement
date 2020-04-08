import torch
import torch.nn as nn
from torch.nn import init
from .guided_filter import FastGuidedFilter, ConvGuidedFilter

def weights_init_identity(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		n_out, n_in, h, w = m.weight.data.size()
		# Last Layer
		if n_out < n_in:
			init.xavier_uniform_(m.weight.data)
			return
		# Except Last Layer
		m.weight.data.zero_()
		ch, cw = h // 2, w // 2
		for i in range(n_in):
			m.weight.data[i, i, ch, cw] = 1.0
	elif classname.find('BatchNorm2d') != -1:
		init.constant_(m.weight.data, 1.0)
		init.constant_(m.bias.data,   0.0)

class AdaptiveNorm(nn.Module):
	def __init__(self, n):
		super(AdaptiveNorm, self).__init__()
		self.w_0 = nn.Parameter(torch.Tensor([1.0]))
		self.w_1 = nn.Parameter(torch.Tensor([0.0]))
		self.bn  = nn.BatchNorm2d(n, momentum=0.999, eps=0.001)
	def forward(self, x):
		return self.w_0 * x + self.w_1 * self.bn(x)

def build_lr_net(norm=AdaptiveNorm, layer=5):
	layers = [
		nn.Conv2d(4, 24, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
		norm(24),
		nn.LeakyReLU(0.2, inplace=True),
	]
	for l in range(1, layer):
		layers += [nn.Conv2d(24,  24, kernel_size=3, stride=1, padding=2**l,  dilation=2**l,  bias=False),
				norm(24),
				nn.LeakyReLU(0.2, inplace=True)]
	layers += [
		nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
		norm(24),
		nn.LeakyReLU(0.2, inplace=True),
		nn.Conv2d(24,  3, kernel_size=1, stride=1, padding=0, dilation=1)
	]
	net = nn.Sequential(*layers)
	net.apply(weights_init_identity)
	return net

class DeepGuidedFilter(nn.Module):#引导滤波
	def __init__(self, radius=1, eps=1e-8,scale=0.25):
		super(DeepGuidedFilter, self).__init__()
		self.lr = build_lr_net()
		self.gf = FastGuidedFilter(radius, eps)
		self.scale=0.25
	def forward(self, x_hr):
		x_lr=torch.nn.functional.interpolate(x_hr,scale_factor=self.scale)
		return self.gf(x_lr[:,:3,::], self.lr(x_lr), x_hr[:,:3,::])
	def init_lr(self, path):
		self.lr.load_state_dict(torch.load(path))
class DeepGuidedFilterAndMap(DeepGuidedFilter):#map+filter
	def __init__(self, radius=1, eps=1e-4,scale=0.25):
		super(DeepGuidedFilterAndMap, self).__init__(radius, eps)
		self.scale=0.25
		self.guided_map = nn.Sequential(
			nn.Conv2d(4, 15, 1, bias=False),
			AdaptiveNorm(15),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Conv2d(15, 3, 1)
		)
		self.guided_map.apply(weights_init_identity)
	def forward(self, x_hr):
		x_lr=torch.nn.functional.interpolate(x_hr,scale_factor=self.scale)
		return self.gf(self.guided_map(x_lr), self.lr(x_lr), self.guided_map(x_hr))

class DeepGuidedFilterLayer(nn.Module):#conv_filter
	def __init__(self, radius=1, layer=5,scale=0.25):
		super(DeepGuidedFilterLayer, self).__init__()
		self.scale=0.25
		self.lr = build_lr_net(layer=layer)
		self.gf = ConvGuidedFilter(radius, norm=AdaptiveNorm)
	def forward(self, x_hr):
		x_lr=torch.nn.functional.interpolate(x_hr,scale_factor=self.scale)
		return self.gf(x_lr[:,:3,::], self.lr(x_lr), x_hr[:,:3,::])
	def init_lr(self, path):
		self.lr.load_state_dict(torch.load(path))

class DeepGuidedFilterLayerAndMap(DeepGuidedFilterLayer):#conv_filter+map
	def __init__(self, radius=1, dilation=0, c=16, layer=5,scale=0.25):
		super(DeepGuidedFilterLayerAndMap, self).__init__(radius, layer)
		self.scale=0.25
		self.guided_map = nn.Sequential(
			nn.Conv2d(4, c, 1, bias=False) if dilation==0 else \
				nn.Conv2d(3, c, 3, padding=dilation, dilation=dilation, bias=False),
			AdaptiveNorm(c),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Conv2d(c, 3, 1)
		)
	def forward(self, x_hr):
		x_lr=torch.nn.functional.interpolate(x_hr,scale_factor=self.scale)
		return self.gf(self.guided_map(x_lr), self.lr(x_lr), self.guided_map(x_hr))
