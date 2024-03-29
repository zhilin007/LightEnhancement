import torch,os,sys,torchvision,tools,glob,numpy as np
from torchvision import utils
from train_tensorboard import models_
from option import opt,cwd,model_name
from data_utils import get_eval_loader
from PIL import Image
from ssim_loss import SSIM as ssimloss
from metrics import psnr,ssim
import torchvision.transforms.transforms as tfs
def dircheck(path):
	if not os.path.exists(path):
		print('grids dir :',path)
		os.makedirs(path)
def getNet():
	net=models_[opt.net].to(opt.device)
	if opt.device=='cuda':
		net=torch.nn.DataParallel(net)
	ckp=torch.load(opt.pth,opt.device)
	net.load_state_dict(ckp['model'])
	print(f'psnr:',ckp['max_psnr'],'ssim:',ckp['max_ssim'])
	return net
def eval(net,loader):
	net.eval()
	illumination=[0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
	for ind ,(inputs,targets) in enumerate(loader):
		inputs=inputs.to(opt.device);targets=targets.to(opt.device)
		N,C,H,W=targets.size()
		i=tools.get_illumination(targets)+torch.zeros([N,1,H,W]).to(opt.device)
		with torch.no_grad():
			pred1=net(torch.cat([inputs,i],1))
		tensorgrid=torch.cat([tools.unNorm(inputs),targets,pred1],dim=0)
		i_conditions=tools.def_illumination(illumination,[1,1,H,W])
		for i_c in i_conditions:
			i_c=i_c.to(opt.device)
			with torch.no_grad():
				pred=net(torch.cat([inputs,i_c],1))
				tensorgrid=torch.cat([tensorgrid,pred],0)
		grid=utils.make_grid(tensorgrid,4)
		i_gt=tools.get_illumination(targets).item()
		save_dir=os.path.join(cwd,'grids',model_name);dircheck(save_dir)
		save_dir=os.path.join(save_dir,f'{ind}_in_gt_{i_gt}_0.01_0.1_0.2_0.3_0.4_0.5_0.6_0.7_0.8_0.9_1.png')
		print(type(grid),grid.shape,ind)
		utils.save_image(grid,save_dir)
def eval_imgs(net,path):
	net.eval()
	imgs=glob.glob(os.path.join(path,'*.png'))
	illumination=[0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
	for im in imgs:
		id=im.split('/')[-1].split('.')[0]
		print(im)
		im=Image.open(im).convert('RGB')
		im=im.resize((256,256))
		im=tfs.ToTensor()(im)
		data=tfs.Normalize(mean=[0.0629,0.0606,0.0558],std=[0.0430,0.0412,0.0425])(im)[None,::]
		N,C,H,W=data.size()
		i_cs=tools.def_illumination(illumination,[1,1,H,W])
		grid=torch.cat([tools.unNorm(data)],0)
		for i_c in i_cs:
			with torch.no_grad():
				pred=net(torch.cat([data,i_c],1))
				grid=torch.cat([grid,pred.cpu()],0)
		grid=utils.make_grid(grid,4)
		save_dir=os.path.join(cwd,'grids_real',model_name);dircheck(save_dir)
		save_dir=os.path.join(save_dir,f'{id}_in_0.01_0.1_0.2_0.3_0.4_0.5_0.6_0.7_0.8_0.9_1.png')
		utils.save_image(grid,save_dir)
def test(net,loader_test):
	net.eval()
	torch.cuda.empty_cache()
	ssims=[]
	psnrs=[]
	losses=[]
	for i ,(inputs,targets) in enumerate(loader_test):
		print(i)
		inputs=inputs.to(opt.device);targets=targets.to(opt.device)
		N,C,H,W=targets.size()
		i=tools.get_illumination(targets)+torch.zeros([N,1,H,W]).to(opt.device)
		pred=net(torch.cat([inputs,i],1))
		loss1=0
		if opt.l1loss:
			loss1+=torch.nn.L1Loss()(pred,targets)
		if opt.mseloss:
			loss1+=torch.nn.MSELoss()(pred,targets)
		if opt.ssimloss:
			loss3=ssimloss()(pred,targets)
			loss1+=(1-loss3)
		ssim1=ssim(pred,targets).item()
		psnr1=psnr(pred,targets)
		ssims.append(ssim1)
		psnrs.append(psnr1)
		losses.append(loss1)
	print( np.mean(ssims),np.mean(psnrs),np.mean(losses))

if __name__ == "__main__":
	#python test.py --net=swiftnet --l1loss --pth=swiftnet_160p_1e5_l1  --divisor=32 --device=cuda:1
	# path='/data/code/LightEnhancement/figs'
	# eval_imgs(net,path)

	# net=getNet()
	loader=get_eval_loader()

	from models import SwiftNet_GuidedFilter
	net=SwiftNet_GuidedFilter().to(opt.device)
	if opt.device=='cuda':
		net=torch.nn.DataParallel(net)
	ckp=torch.load(opt.pth,opt.device)
	net.load_state_dict(ckp['model'])
	print(f'psnr:',ckp['max_psnr'],'ssim:',ckp['max_ssim'])
	with torch.no_grad():
		test(net,loader)
	
	
	
	

		


