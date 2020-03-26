### LOL 

对于数据集的指标，使用没有光照scalar的模型做infer。和其他人的一样。

|net|step|psnr|ssim|time|line|
|-|-|-|-|-|-|
|unet|1e5|19.6919|0.8089|4h|python train.py --net='unet' --step=100000 --pth=unet_160p_1e5_l1 --divisor=16 --bs=8 --l1loss --crop_size=160|
|unet64|1e5|||4h|python train_tensorboard.py --net='unet64' --step=100000 --device=cuda:0 --pth=unet64_160p_1e5_l1 --divisor=16 --bs=8 --l1loss --crop_size=160|
|swiftnet|1e5|||h|python train.py --net='swiftnet' --device=cuda:0 --step=100000 --pth=swiftnet_160p_1e5_l1 --divisor=32 --bs=8 --l1loss --crop_size=160|
|euunet|1e5|||4h|python train.py --net='euunet' --step=100000 --pth=euunet_160p_1e5_l1 --divisor=16 --bs=8 --l1loss --crop_size=160|
|gen_y_unet|2e5|20.6982|0.8325|12h|rpython Gen_Y_train.py --net='gen_y_unet' --step=200000 --pth=gen_y_unet_160p_2e5_l1 --divisor=16 --bs=8 --l1loss --crop_size=160|
|gen_y_unet64|2e5|||12h|python Gen_Y_train.py --net='gen_y_unet64' --step=200000 --pth=gen_y_unet64_160p_2e5_l1 --divisor=16 --bs=8 --l1loss --crop_size=160|
