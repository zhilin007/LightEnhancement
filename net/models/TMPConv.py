import torch
import torch.nn as nn
import torch.nn.functional as F

class FULLCONV(nn.Module):
	def __init__(self):
		super(FULLCONV,self).__init__()
		layer=[]
		layer.append(nn.Conv2d(3,64,3,1,1))
		for i in range(1,40):
			layer.append(nn.Conv2d(64,64,3,3,1))
			layer.append(nn.ReLU())
		self.net=nn.Sequential(*layer)
	def forward(self,x):
		return self.net(x)
	
if __name__ == "__main__":
	#我操 非常快！！！
	x=torch.ones([1,3,1024,1024])
	net=FULLCONV()
	net(x)