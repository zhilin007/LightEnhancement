import torch,time
from models import *
from option import cwd,opt
x=torch.ones([1,4,1024,1024])
net=Unet()
net=torch.nn.DataParallel(net).to('cuda:0')
def netload(net):
	pth=os.path.join(cwd,net,'best_pth','unet_160p_1e5_l1.pth')
	pth=torch.load(pth)
	net.load_state_dict(pth['model'])
	return net
x=torch.ones([1,4,1024,1024]).to('cuda:0')
stime=time.time()
for i in range(100000):
	print(f'\r {i}/100000',end='',flush=True)
	with torch.no_grad():
		_=net(x)
time_interval=time.time()-stime
FPS=100000/time_interval
print('FPS of net':FPS)

net=netload(net)
x=torch.ones([1,4,1024,1024]).to('cuda:0')
stime=time.time()
for i in range(100000):
	print(f'\r {i}/100000',end='',flush=True)
	with torch.no_grad():
		_=net(x)
time_interval=time.time()-stime
FPS=100000/time_interval
print('FPS of trained_net':FPS)

