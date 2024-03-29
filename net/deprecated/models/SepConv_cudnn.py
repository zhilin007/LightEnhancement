import torch,torch.nn as nn,time
from torchstat import stat

def conv(feats,groups,kernel_size=3):
	return nn.Conv2d(in_channels=feats,out_channels=feats,kernel_size=kernel_size,
			stride=1,padding=kernel_size//2,groups=groups)
class SepConvNet(nn.Module):
	def __init__(self,nums_layer=4,feats=16):
		super(SepConvNet,self).__init__()
		layers=[]
		for i in range(nums_layer):
			layers.append(conv(feats,groups=feats,kernel_size=3))
			layers.append(conv(feats,groups=1,kernel_size=1))
			layers.append(nn.ReLU(True))
		self.backbone=nn.Sequential(*layers)
	def forward(self,x):
		out=self.backbone(x)
		return out
class StandConvNet(nn.Module):
	def __init__(self,nums_layer=4,feats=16):
		super(StandConvNet,self).__init__()
		layers=[]
		for i in range(nums_layer):
			layers.append(conv(feats,groups=1,kernel_size=3))
			layers.append(nn.ReLU(True))
		self.backbone=nn.Sequential(*layers)
	def forward(self,x):
		out=self.backbone(x)
		return out
if __name__ == "__main__":
	nums_layer=4;feats=16;steps=100000
	sepnet=SepConvNet(nums_layer,feats)
	stanet=StandConvNet(nums_layer,feats)
	
	'''
	stat(sepnet,(16,256,256))
	params: 1,728
	memory: 32.00MB
	Flops: 113.25MFlops

	stat(stanet,(16,256,256))
	params: 9,280 (x5.4)
	memory: 16.00MB
	Flops: 608.17MFlops (x5.4)
	'''
	#same with paper
	#208s
	x=torch.ones([1,16,256,256]).to('cuda:1')
	sepnet=sepnet.to('cuda:1')
	sep_time=time.time()
	for i in range(steps):
		print(f'\r sepnet {i}/{steps}',end='',flush=True)
		with torch.no_grad():
			x=sepnet(x)
	time_interval=time.time()-sep_time
	print(time_interval)
	#109s
	x=torch.ones([1,16,256,256]).to('cuda:1')
	stanet=stanet.to('cuda:1')
	stand_time=time.time()
	for i in range(steps):
		print(f'\r stanet {i}/{steps}',end='',flush=True)
		with torch.no_grad():
			x=stanet(x)
	time_interval=time.time()-stand_time
	print(time_interval)
	