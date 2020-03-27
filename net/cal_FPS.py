import torch,time,os
from models import *
from option import cwd,opt
import numpy as np
# x=torch.ones([1,4,1024,1024])
# net=UNet64().to('cuda:0')
# def netload(net):
# 	pth=os.path.join(cwd,'net','best_pth','unet64_160p_1e5_l1.pth')
# 	pth=torch.load(pth)
# 	net.load_state_dict(pth['model'])
# 	return net
# steps=2000
# x=torch.ones([1,4,1024,1024]).to('cuda:0')
# stime=time.time()
# for i in range(steps):
# 	print(f'\r {i}/{steps}',end='',flush=True)
# 	with torch.no_grad():
# 		_=net(x)
# time_interval=time.time()-stime
# FPS=steps/time_interval
# print('FPS of net:',FPS)
# net=netload(net)
# x=torch.ones([1,4,1024,1024]).to('cuda:0')
# stime=time.time()
# for i in range(steps):
# 	print(f'\r {i}/{steps}',end='',flush=True)
# 	with torch.no_grad():
# 		_=net(x)
# time_interval=time.time()-stime
# FPS=steps/time_interval
# print('FPS of trained_net:',FPS)

if __name__ == "__main__":
	#python cal_FPS --net=
	net=models[opt.net].to('cuda:0')
	steps=opt.steps
	x=torch.ones([1,4,1024,1024]).to('cuda:0')
	times=[]
	stime=time.time()
	for i in range(steps):
		time1=time.time()
		with torch.no_grad():
			_=net(x)
		infer_time=time.time-time1()
		times.append(infer_time)
		print(f'{i}/{steps} infer_time:{infer_time}')
	time_interval=time.time()-stime
	FPS=steps/time_interval
	print(f'FPS of {opt.net}:',FPS)
	print(f'infer_time: avg{np.mean(times)},min{np.min(times)},max{np.max(times)}')

