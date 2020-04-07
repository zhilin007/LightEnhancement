import matplotlib.pyplot as plt
import numpy as np
import os
abs='/'.join(os.path.realpath(__file__).split('/')[:-1])+'/'
def np_plot(losses_dir,acces_dir):
	losses=np.load(losses_dir)
	acces=np.load(acces_dir)
	print('maxacc :',np.max(acces))
	model_name=losses_dir.split('/')[-1].split('_')[0]
	epochs=losses_dir.split('/')[-1].split('_')[1]
	plt.figure(figsize=(12,6))
	plt.title(model_name+' '+epochs)
	
	ax1=plt.subplot(121)
	ax1.plot(losses)
	ax1.set_title('losses')
	ax2=plt.subplot(122)
	ax2.plot(acces)
	ax2.set_title('acces')
	plt.show()
def plots(nps,title):
	plt.title(title)
	for f in nps :
		print(f)
		a=np.load(abs+f)
		plt.plot(a,label=str(f))
	plt.legend()
	plt.show()
def plot_all_files(dir,keys=['MPID','psnr'],filters=['']):
	files=os.listdir(dir)
	def filter_key(str):
		for k in keys :
			if str.find(k) == -1:
				return False
		return True 
	def filter_filter(str):
		for k in filters:
			if str.find(k) !=-1 :
				return False
		return True
	files=filter(filter_key,files)
	files=filter(filter_filter,files)
	plots(files,'/'.join(keys))

if __name__ == "__main__":
	# x=np.load(abs+'unet_160p_1e5_l1_100000_losses.npy')
	plot_all_files(abs,keys=['losses','1e5'],filters=[])