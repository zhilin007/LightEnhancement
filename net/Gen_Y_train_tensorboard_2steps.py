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
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
from option import opt,step_save_pth,model_name
from data_utils import *
from ssim_loss import SSIM as ssimloss
from torchvision.models import vgg16
import tools
warnings.filterwarnings('ignore')
#test时候同时画train和eval的psnr,ssim,loss
models_={
	# 'gen_y_unet':GE_Y_Unet(),
	# 'gen_y_unet64':GE_Y_Unet64(),
	'Gen_Y_Swiftslim2_BN2':Gen_Y_Swiftslim2_BN2(norm=opt.norm,scale_factor=opt.scale_factor),
	'Gen_Y_Swiftslim2_BN2_Share':Gen_Y_Swiftslim2_BN2_Share(norm=opt.norm,scale_factor=opt.scale_factor),
	'Gen_Y_Swiftslim2_BN2_SAME':Gen_Y_Swiftslim2_BN2_SAME(norm=opt.norm),
	'Gen_Y_Swiftslim2_BN2_SAME_DownSample':Gen_Y_Swiftslim2_BN2_SAME_DownSample(norm=opt.norm,scale_factor=opt.scale_factor),
	'Gen_Y_Swiftslim_BN2':Gen_Y_Swiftslim_BN2(norm=opt.norm,scale_factor=opt.scale_factor),
	'Gen_Y_Backbone7x7':Gen_Y_Backbone7x7(norm=opt.norm,scale_factor=opt.scale_factor),
}

start_time=time.time()

def lr_schedule_cosdecay(t,T,init_lr=opt.lr):
	lr=0.5*(1+math.cos(t*math.pi/T))*init_lr
	return lr

def train_y(net,loader_train,loader_test,loader_eval_train,optim_y,criterion,steps):
	start_step=0
	if os.path.exists(step_save_pth):
		print(f'resume from {step_save_pth}')
		ckp=torch.load(step_save_pth)
		net.load_state_dict(ckp['model'])
		start_step=ckp['step']
		print(f'start_step:{start_step} start training ---')
	else :
		print('train from scratch *** ')
	for step in range(start_step+1,steps+1):
		net.train()
		lr=lr_schedule_cosdecay(step,steps)
		for param_group in optim_y.param_groups:
			param_group["lr"] = lr  
		x,y=next(iter(loader_train))
		x=x.to(opt.device);y=y.to(opt.device)
		N,C,H,W=y.size()
		Y_gt=tools.get_illumination(y,False).to(opt.device)

		if opt.incolor==4:
			Y_mean=tools.get_illumination(y)+torch.zeros([N,1,H,W]).to(opt.device)
		elif opt.incolor==3:
			Y_mean=torch.Tensor([0.458971]).to(opt.device)+torch.zeros([N,1,H,W]).to(opt.device)#ImageNet
		else :
			raise Exception('Runtime Error')

		Y_out,out=net(x,Y_mean)
		#l1lss as default for y
		loss_y=criterion['l1loss'](Y_out,Y_gt)

		optim_y.zero_grad()
		loss_y.backward()
		optim_y.step()
		
		esti_time=((time.time()-start_time)/60)/(step-start_step)
		esti_time=(steps-step)*esti_time
		print(f'\rtrain_Y loss : {loss_y.item():.5f}| step :{step}/{steps}|lr :{lr :.7f} |esti_time :{esti_time/60:.1f}h ',end='',flush=True)
		if step % opt.eval_step ==0 :
			with torch.no_grad():
				loss_eval=test_y(net,loader_test)#on eval_dataset
				loss_train=test_y(net,loader_eval_train)#on train_dataset
				print(f'\nstep :{step} on eval_dataset |loss:{loss_eval:.5f}')
				print(f'step :{step} on train_dataset |loss:{loss_train:.5f}')
			torch.save({'step':step,'max_psnr':0,'max_ssim':0,'ssims':[],'psnrs':[],'losses':[],'model':net.state_dict()},step_save_pth)
def test_y(net,loader_test):
	net.eval()
	torch.cuda.empty_cache()
	losses=[]
	for i ,(inputs,targets) in enumerate(loader_test):
		inputs=inputs.to(opt.device);targets=targets.to(opt.device)
		N,C,H,W=targets.size()
		if opt.incolor==4:
			i=tools.get_illumination(targets)+torch.zeros([N,1,H,W]).to(opt.device)
		elif opt.incolor==3:
			i=torch.Tensor([0.458971]).to(opt.device)+torch.zeros([N,1,H,W]).to(opt.device)#ImageNet
		else:
			raise Exception('Runtime Error')
		Y_gt=tools.get_illumination(targets,False).to(opt.device)
		y_pred,pred=net(inputs,i)
		loss_y=criterion['l1loss'](y_pred,Y_gt)#default for Y
		losses.append(loss_y.item())
	return np.mean(losses)
def train_out(net,loader_train,loader_test,loader_eval_train,optim_net,criterion,steps):
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
	for step in range(start_step+1,steps+1):
		net.train()
		lr=lr_schedule_cosdecay(step-(opt.steps//2),steps-(opt.steps//2))
		for param_group in optim_net.param_groups:
			param_group["lr"] = lr 
		x,y=next(iter(loader_train))
		x=x.to(opt.device);y=y.to(opt.device)
		N,C,H,W=y.size()
		Y_gt=tools.get_illumination(y,False).to(opt.device)

		if opt.incolor==4:
			Y_mean=tools.get_illumination(y)+torch.zeros([N,1,H,W]).to(opt.device)
		elif opt.incolor==3:
			Y_mean=torch.Tensor([0.458971]).to(opt.device)+torch.zeros([N,1,H,W]).to(opt.device)#ImageNet
		else :
			raise Exception('Runtime Error')

		Y_out,out=net(x,Y_mean)

		loss_out=0
		if opt.l1loss:
			loss_out=criterion['l1loss'](out,y)+loss_out
		if opt.mseloss:
			loss_out=criterion['mseloss'](out,y)+loss_out
		if opt.ssimloss:
			loss3=criterion['ssimloss'](out,y)
			loss_out=loss_out+(1-loss3)

		optim_net.zero_grad()
		loss_out.backward()
		optim_net.step()
		
		
		losses.append(loss_out.item())
		esti_time=((time.time()-start_time)/60)/(step-start_step)
		esti_time=(steps-step)*esti_time
		print(f'\rtrain_out loss : {loss_out.item():.5f}| step :{step}/{steps}|lr :{lr :.7f} |esti_time :{esti_time/60:.1f}h ',end='',flush=True)
		if step % opt.eval_step ==0 :
			with torch.no_grad():
				ssim_eval,psnr_eval,loss_eval=test_out(net,loader_test)#on eval_dataset
				ssim_train,psnr_train,loss_train=test_out(net,loader_eval_train)#on train_dataset
				print(f'\nstep :{step} on eval_dataset |ssim:{ssim_eval:.5f} | psnr:{psnr_eval:.4f} |loss:{loss_eval:.5f}')
				print(f'step :{step} on train_dataset |ssim:{ssim_train:.5f} | psnr:{psnr_train:.4f} |loss:{loss_train:.5f}')
				#plot tensorboard
				with SummaryWriter(log_dir=opt.log_dir) as writer:
					writer.add_scalars('ssim',{'eval':ssim_eval,'train':ssim_train},step)
					writer.add_scalars('psnr',{'eval':psnr_eval,'train':psnr_train},step)
					writer.add_scalars('loss',{'eval':loss_eval,'train':loss_train},step)

			ssims.append(ssim_eval);psnrs.append(psnr_eval)
			torch.save({'step':step,'max_psnr':max_psnr,'max_ssim':max_ssim,'ssims':ssims,'psnrs':psnrs,'losses':losses,'model':net.state_dict()},step_save_pth)
			if  psnr_eval > max_psnr and ssim_eval > max_ssim:
				max_ssim=max(max_ssim,ssim_eval)
				max_psnr=max(max_psnr,psnr_eval)
				torch.save({'step':step,'max_psnr':max_psnr,'max_ssim':max_ssim,'ssims':ssims,'psnrs':psnrs,'losses':losses,'model':net.state_dict()},opt.pth)
				print(f'\n best model saved at step :{step}| max_ssim:{max_ssim:.4f}| max_psnr:{max_psnr:.4f}|')

	np.save(f'{cwd}/net/np_files/{model_name}_losses.npy',losses)
	np.save(f'{cwd}/net/np_files/{model_name}_ssims.npy',ssims)
	np.save(f'{cwd}/net/np_files/{model_name}_psnrs.npy',psnrs)

def test_out(net,loader_test):
	net.eval()
	torch.cuda.empty_cache()
	ssims=[]
	psnrs=[]
	losses=[]
	for i ,(inputs,targets) in enumerate(loader_test):
		inputs=inputs.to(opt.device);targets=targets.to(opt.device)
		N,C,H,W=targets.size()
		if opt.incolor==4:
			i=tools.get_illumination(targets)+torch.zeros([N,1,H,W]).to(opt.device)
		elif opt.incolor==3:
			i=torch.Tensor([0.458971]).to(opt.device)+torch.zeros([N,1,H,W]).to(opt.device)#ImageNet
		else:
			raise Exception('Runtime Error')

		Y_gt=tools.get_illumination(targets,False).to(opt.device)
		y_pred,pred=net(inputs,i)

		loss_out=0
		if opt.l1loss:
			loss_out=criterion['l1loss'](pred,targets)+loss_out
		if opt.mseloss:
			loss_out=criterion['mseloss'](pred,targets)+loss_out
		if opt.ssimloss:
			loss3=criterion['ssimloss'](pred,targets)
			loss_out=loss_out+(1-loss3)

		ssim1=ssim(pred,targets).item()
		psnr1=psnr(pred,targets)
		ssims.append(ssim1)
		psnrs.append(psnr1)
		losses.append(loss_out.item())
	return np.mean(ssims),np.mean(psnrs),np.mean(losses)

if __name__ == "__main__":
	loader_train=get_train_loader()
	loader_test=get_eval_loader()
	loader_eval_train=get_eval_train_loader()

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
	optimizer_y = RAdam(params=filter(lambda x: x.requires_grad,net.genY.parameters()),lr=opt.lr, betas = (0.9, 0.999), eps=1e-08)
	optimizer_y.zero_grad()
	optimizer_net=RAdam(params=filter(lambda x: x.requires_grad,net.genO.parameters()),lr=opt.lr,betas = (0.9, 0.999), eps=1e-08)
	optimizer_net.zero_grad()
	# train(net,loader_train,loader_test,loader_eval_train,optimizer_y,optimizer_net,criterion)
	train_y(net,loader_train,loader_test,loader_eval_train,optimizer_y,criterion,opt.steps//2)
	train_out(net,loader_train,loader_test,loader_eval_train,optimizer_net,criterion,opt.steps)

	

