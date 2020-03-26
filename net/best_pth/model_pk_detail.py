import torch
import os

def print_ckp(dir):
	ck=torch.load(dir,map_location='cpu')
	for key,value in ck.items():
		if key == 'max_ssim' or key =='max_psnr' or key=='step':
			print(f'{key} : {value}')
		else :
			pass
			

if __name__ == "__main__":
	dir='net/best_pth/'
	for i in os.listdir(dir):
		if i.find('.pth')!=-1:
			print(i)
			print_ckp(dir+i)