import torch ,numpy as np
import torch.nn as nn

def edge_func(x):
    #   x    : tensor of shape [N,3,H,W]
    #return  : tensor of shape [N,3,H,W]
    assert len(x.size())==4
    x_diffx = torch.abs(x[:,:,:,1:] - x[:,:,:,:-1])
    x_diffy = torch.abs(x[:,:,1:,:] - x[:,:,:-1,:])
    y = x.new(x.size())
    y.fill_(0)
    y[:,:,:,1:] += x_diffx
    y[:,:,:,:-1] += x_diffx
    y[:,:,1:,:] += x_diffy
    y[:,:,:-1,:] += x_diffy
    y/=3
    return y
class EML1Loss(nn.Module):
	def __init__(self):
		super(EML1Loss,self).__init__()
	def forward(self,pred,gt):
		N,C,H,W=pred.size()
		# print(N,C,H,W)
		edge=edge_func(gt)
		m=torch.min(edge.view(N,C,-1),dim=-1,keepdim=True)[0][:,:,:,None]
		edge-=m
		v=torch.max(edge.view(N,C,-1),dim=-1,keepdim=True)[0][:,:,:,None]
		edge/=v
		w=1+torch.log2(edge+1)
		l1=torch.abs(pred-gt)
		loss=l1*w
		loss=torch.mean(loss)
		return loss


if __name__ == "__main__":
	a=np.random.random(size=(1,3,2,2))
	b=np.random.random(size=(1,3,2,2))
	pred=torch.tensor(a,requires_grad=True)
	gt=torch.tensor(b,requires_grad=False)
	# m=torch.min(gt.view(1,4,-1),dim=-1,keepdim=True)[0][:,:,:,None]
	# print(m.size())
	loss=EML1Loss()
	l1=loss(pred,gt)
	print(l1)
	l1.backward()
	pass
