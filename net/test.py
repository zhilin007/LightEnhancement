import torch,os,sys,torchvision,tools
from torchvision import utils
from train import models_
from option import opt,cwd
from data_utils import get_eval_loader

def getNet():
	net=models_[opt.net].to(opt.device)
	net=torch.nn.DataParallel(net)
	ckp=torch.load(opt.pth)
	net.load_state_dict(ckp['model'])
	print(f'psnr:',ckp['max_psnr'],'ssim:',ckp['max_ssim'])
	return net
def eval(net,loader):
	illumination=[0.1,0.2,0.3,0.4,0.5]
	for ind ,(inputs,targets) in enumerate(loader):
		inputs=inputs.to(opt.device);targets=targets.to(opt.device)
		N,C,H,W=targets.size()
		i=tools.get_illumination(targets)+torch.zeros([N,1,H,W]).to(opt.device)
		with torch.no_grad():
			pred1=net(torch.cat([inputs,i],1))
		tensorgrid=torch.cat([inputs,targets,pred1],dim=0)
		i_conditions=tools.def_illumination(illumination,[1,1,H,W])
		for i_c in i_conditions:
			i_c=i_c.to(opt.device)
			with torch.no_grad():
				pred=net(torch.cat([inputs,i_c],1))
				tensorgrid=torch.cat([tensorgrid,pred],0)
		grid=utils.make_grid(tensorgrid,8,0)
		i_gt=tools.get_illumination(targets).item()
		save_dir=os.path.join(cwd,'grids','unet',f'{ind}_in_gt_{i_gt}_0.1_0.2_0.3_0.4_0.5.png')
		print(type(grid),grid.shape,ind)
		utils.save_image(grid,save_dir)
if __name__ == "__main__":
	#rpython test.py --net='unet' --pth=unet_160p_1e5_l1 --divisor=16
	net=getNet()
	loader=get_eval_loader()
	eval(net,loader)

		


