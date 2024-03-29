import torch,os,sys,torchvision,tools,glob
from torchvision import utils
from Gen_Y_train_tensorboard import models_
# from Gen_Y_Share_train_tensorboard import models_
from metrics import psnr , ssim 
from option import opt,cwd,model_name
from data_utils import get_eval_loader
from PIL import Image
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
		print(ind)
		inputs=inputs.to(opt.device);targets=targets.to(opt.device)
		N,C,H,W=targets.size()
		i=tools.get_illumination(targets)+torch.zeros([N,1,H,W]).to(opt.device)
		with torch.no_grad():
			pred1_Y,pred1=net(inputs,i)
			psnr1=psnr(pred1,targets);ssim1=ssim(pred1,targets)
		tensorgrid=torch.cat([tools.unNorm(inputs),targets,pred1],dim=0)
		#暂时不用这么多Y
		i_conditions=tools.def_illumination(illumination,[1,1,H,W])
		for i_c in i_conditions:
			i_c=i_c.to(opt.device)
			with torch.no_grad():
				_,pred=net(inputs,i_c)
				tensorgrid=torch.cat([tensorgrid,pred],0)

		grid=utils.make_grid(tensorgrid,4)
		i_gt=tools.get_illumination(targets).item()
		save_dir=os.path.join(cwd,'grids',model_name);dircheck(save_dir)
		save_dir=os.path.join(save_dir,f'{ind}_in_gt_{i_gt}_0.01_0.1_0.2_0.3_0.4_0.5_0.6_0.7_0.8_0.9_1.png')
		# save_dir=os.path.join(save_dir,f'{ind}_in_gt_{i_gt}_{psnr1}_{ssim1}.png')

		print(type(grid),grid.shape,ind,psnr1,ssim1)
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
				_,pred=net(data,i_c)
				grid=torch.cat([grid,pred.cpu()],0)
		grid=utils.make_grid(grid,4)
		save_dir=os.path.join(cwd,'grids_real',model_name);dircheck(save_dir)
		save_dir=os.path.join(save_dir,f'{id}_in_0.01_0.1_0.2_0.3_0.4_0.5_0.6_0.7_0.8_0.9_1.png')
		utils.save_image(grid,save_dir)
if __name__ == "__main__":
	#python test_gen_Y.py --pth=Gen_Y_Backbone7x7_inC4_384p_2e5_l1_ssim --divisor=16 --device='cuda:2' --norm --net=Gen_Y_Backbone7x7  --scale_factor=0.25 --incolor=4
	# python test_gen_Y.py --pth=gen_y_unet_160p_2e5_l1 --divisor=16 --net=gen_y_unet
	net=getNet()
	loader=get_eval_loader()
	eval(net,loader)
	# path='/data/code/LightEnhancement/figs'
	# eval_imgs(net,path)

		


