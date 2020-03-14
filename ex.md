### LOL 

|net|step|psnr|ssim|time|line|
|-|-|-|-|-|-|
|unet|1e5|||4h|rpython train.py --net='unet' --step=100000 --pth=unet_160p_1e5_l1 --divisor=16 --bs=8 --l1loss --crop_size=160|
|gen_y_unet|1e5|||4h|rpython Gen_Y_train.py --net='gen_y_unet' --step=200000 --pth=gen_y_unet_160p_2e5_l1 --divisor=16 --bs=8 --l1loss --crop_size=160|