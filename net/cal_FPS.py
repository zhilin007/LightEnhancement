import torch,time,os
# from train_tensorboard import models_
from gfl_train_tensorboard import models_
# from Gen_Y_train_tensorboard import models_
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
	#python cal_FPS.py --steps=1000 --device='cuda:2' --net=SwiftNetSlim_GFL_SN --norm
	net=models_[opt.net].to(opt.device)
	steps=opt.steps
	x=torch.ones([1,4,1024,1024]).to(opt.device)
	# x=torch.ones([1,3,1024,1024]).to(opt.device)
	# y=torch.ones([1,1,1024,1024]).to(opt.device)
	times=[]
	stime=time.time()
	net.eval()
	for i in range(steps):
		time1=time.time()
		with torch.no_grad():
			_=net(x)
			# _=net(x,y)
		infer_time=time.time()-time1
		times.append(infer_time)
		print(f'{i}/{steps} infer_time:{infer_time}')
	time_interval=time.time()-stime
	FPS=steps/time_interval
	print('current device :',torch.cuda.get_device_name(0))
	print(f'FPS of {opt.net}:',FPS)
	print(x.size())
	print(f'infer_time: avg{np.mean(times)},min{np.min(times)},max{np.max(times)}')

