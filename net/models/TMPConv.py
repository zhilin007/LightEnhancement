import torch
import torch.nn as nn
import torch.nn.functional as F

class FULLCONV(nn.Module):
	def __init__(self):
		super(FULLCONV,self).__init__()
		layer=[]
		layer.append(nn.Conv2d(3,16,3,1,1))
		for i in range(1,10):
			layer.append(nn.Conv2d(16,16,3,3,1))
		self.net=nn.Sequential(*layer)
	def forward(self,x):
		return self.net(x)
	
if __name__ == "__main__":
	x=torch.ones([1,3,1024,1024])
	net=FULLCONV()
	net(x)