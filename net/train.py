import torch,os,sys,torchvision,argparse
import torchvision.transforms as tfs
from metrics import psnr,ssim
from models import *
import time,math,warnings
import numpy as np
from torch.backends import cudnn
from radam import RAdam
import torch,warnings
from torch import nn
# from tensorboardX import SummaryWriter
import torchvision.utils as vutils
from option import opt,step_save_pth,model_name
from data_utils import *
from ssim_loss import SSIM as ssimloss
from torchvision.models import vgg16
import tools
warnings.filterwarnings('ignore')
models_={
	'unet':UNet(),
	'unet64':UNet64(),
	'euunet':EUUNet(),
	'gen_y_unet':GE_Y_Unet(),
	'gen_y_unet':GE_Y_Unet64(),
}

start_time=time.time()
T=opt.steps	

def lr_schedule_cosdecay(t,T,init_lr=opt.lr):
	lr=0.5*(1+math.cos(t*math.pi/T))*init_lr
	return lr

def train(net,loader_train,loader_test,optim,criterion):
	losses=[]
	start_step=0
	max_ssim=0
	max_psnr=0
	ssims=[]
	psnrs=[]
	if os.path.exists(step_save_pth):
		print(f'resume from {step_save_pth}')
		ckp=torch.load(step_save_pth)
		losses=ckp['losses']
		net.load_state_dict(ckp['model'])
		start_step=ckp['step']
		max_ssim=ckp['max_ssim']
		max_psnr=ckp['max_psnr']
		psnrs=ckp['psnrs']
		ssims=ckp['ssims']
		print(f'start_step:{start_step} start training ---')
	else :
		print('train from scratch *** ')
	for step in range(start_step+1,opt.steps+1):
		net.train()
		lr=lr_schedule_cosdecay(step,T)
		for param_group in optim.param_groups:
			param_group["lr"] = lr  
		x,y=next(iter(loader_train))
		x=x.to(opt.device);y=y.to(opt.device)
		N,C,H,W=y.size()
		i=tools.get_illumination(y)+torch.zeros([N,1,H,W]).to(opt.device)
		out=net(torch.cat([x,i],1))
		loss=0
		if opt.l1loss:
			loss=criterion['l1loss'](out,y)+loss
		if opt.mseloss:
			loss=criterion['mseloss'](out,y)+loss
		if opt.ssimloss:
			loss3=criterion['ssimloss'](out,y)
			loss=loss+(1-loss3)
		loss.backward()
		optim.step()
		optim.zero_grad()
		losses.append(loss.item())
		esti_time=((time.time()-start_time)/60)/(step-start_step)
		esti_time=(opt.steps-step)*esti_time
		print(f'\rtrain loss : {loss.item():.5f}| step :{step}/{opt.steps}|lr :{lr :.7f} |esti_time :{esti_time/60:.1f}h｜GPU:{torch.cuda.max_memory_allocated()/(1024**3) :.1f}G',end='',flush=True)
		
		if step % opt.eval_step ==0 :
			with torch.no_grad():
				ssim_eval,psnr_eval=test(net,loader_test)
			print(f'\nstep :{step} |ssim:{ssim_eval:.5f} | psnr:{psnr_eval:.4f}')
			ssims.append(ssim_eval)
			psnrs.append(psnr_eval)
			torch.save({'step':step,'max_psnr':max_psnr,'max_ssim':max_ssim,'ssims':ssims,'psnrs':psnrs,'losses':losses,'model':net.state_dict()},step_save_pth)
			if  psnr_eval > max_psnr and ssim_eval > max_ssim:
				max_ssim=max(max_ssim,ssim_eval)
				max_psnr=max(max_psnr,psnr_eval)
				torch.save({'step':step,'max_psnr':max_psnr,'max_ssim':max_ssim,'ssims':ssims,'psnrs':psnrs,'losses':losses,'model':net.state_dict()},opt.pth)
				print(f'\n best model saved at step :{step}| max_ssim:{max_ssim:.4f}| max_psnr:{max_psnr:.4f}|')

	np.save(f'{cwd}/net/np_files/{model_name}_{opt.steps}_losses.npy',losses)
	np.save(f'{cwd}/net/np_files/{model_name}_{opt.steps}_ssims.npy',ssims)
	np.save(f'{cwd}/net/np_files/{model_name}_{opt.steps}_psnrs.npy',psnrs)

def test(net,loader_test):
	net.eval()
	torch.cuda.empty_cache()
	ssims=[]
	psnrs=[]
	l=len(loader_test)
	for i ,(inputs,targets) in enumerate(loader_test):
		inputs=inputs.to(opt.device);targets=targets.to(opt.device)
		N,C,H,W=targets.size()
		i=tools.get_illumination(targets)+torch.zeros([N,1,H,W]).to(opt.device)
		pred=net(torch.cat([inputs,i],1))
		ssim1=ssim(pred,targets).item()
		psnr1=psnr(pred,targets)
		ssims.append(ssim1)
		psnrs.append(psnr1)
	return np.mean(ssims) ,np.mean(psnrs)

if __name__ == "__main__":
	loader_train=get_train_loader()
	loader_test=get_eval_loader()
	net=models_[opt.net]
	net=net.to(opt.device)
	if opt.device=='cuda':
		net=torch.nn.DataParallel(net)
		cudnn.benchmark=True
	criterion = {}
	if opt.l1loss:
		criterion.update({'l1loss':nn.L1Loss().to(opt.device)})
	if opt.mseloss:
		criterion.update({'mseloss':nn.MSELoss().to(opt.device)})
	if opt.ssimloss:
		criterion.update({'ssimloss':ssimloss().to(opt.device)})
	optimizer = RAdam(params=filter(lambda x: x.requires_grad, net.parameters()),lr=opt.lr, betas = (0.9, 0.999), eps=1e-08)
	optimizer.zero_grad()
	train(net,loader_train,loader_test,optimizer,criterion)
	

