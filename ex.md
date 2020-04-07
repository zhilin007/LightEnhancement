### LOL 

对于数据集的指标，使用没有光照scalar的模型做infer。和其他人的一样。
ulimit -n 2048
nohup python >log.out

|net|step|psnr|ssim|time|line|
|-|-|-|-|-|-|
|unet`其结果模糊不清尽管能够做到denoise`|1e5|19.6919|0.8089|4h|python train.py --net='unet' --step=100000 --pth=unet_160p_1e5_l1 --divisor=16 --bs=8 --l1loss --crop_size=160|
|`unet64`|1e5|20.1542|0.7777|4h|python train_tensorboard.py --net='unet64' --step=100000 --device=cuda:0 --pth=unet64_160p_1e5_l1 --divisor=16 --bs=8 --l1loss --crop_size=160|
|unet64|3e5|19.7481|0.8190|12h|python train_tensorboard.py --net='unet64' --step=300000 --device=cuda:0 --pth=unet64_160p_3e5_l1 --divisor=16 --bs=8 --l1loss --crop_size=160|
|unet64|1e5|19.9966|0.7806|4h|python train_tensorboard.py --net='unet64' --step=100000 --device=cuda:1 --pth=unet64_160p_1e5_l1_8eN5 --lr=0.00008 --divisor=16 --bs=8 --l1loss --crop_size=160|
|`swiftnet`output是1/4 上采样|1e5|19.2726|0.7124|8h|python train_tensorboard.py --net='swiftnet' --device=cuda:0 --step=100000 --pth=swiftnet_160p_1e5_l1 --divisor=32 --bs=8 --l1loss --crop_size=160 --lr=0.0004|
|FullConv_SwiftNet|1e5|0.7805|19.5069|5h|python train_tensorboard.py --net='FullConv_SwiftNet' --device=cuda:0 --step=100000 --pth=FullConv_SwiftNet_160p_1e5_l1 --divisor=32 --bs=8 --l1loss --crop_size=160 --lr=0.0004|
|swiftnetslim约等于w=0.2|2e5|17.7897|0.6939|3h|python train_tensorboard.py --net='swiftnetslim' --device=cuda:0 --step=200000 --pth=swiftnetslim_160p_2e5_l1 --divisor=32 --bs=8 --l1loss --crop_size=160 --lr=0.0004|
|euunet|1e5|||4h|python train.py --net='euunet' --step=100000 --pth=euunet_160p_1e5_l1 --divisor=16 --bs=8 --l1loss --crop_size=160|
|gen_y_unet|2e5|20.6982|0.8325|12h|rpython Gen_Y_train.py --net='gen_y_unet' --step=200000 --pth=gen_y_unet_160p_2e5_l1 --divisor=16 --bs=8 --l1loss --crop_size=160|
|`hdr1`没有训练patch和整张图的过拟合（因为子分辨率变换算子与原分辨率算子相似），但是有train/test的过拟合，`grid_sample采用的是bilinear（x,y）,最关键的guided-map（z）没有采样！`|1e5|`25.6277结果较清晰`|`0.7534但存在很大的噪声`|19h|python train_tensorboard.py --net='hdr1' --step=100000 --device=cuda:0 --pth=hdr1_384p_1e5_l1 --divisor=1 --bs=8 --l1loss --crop_size=384|
|Res18Net1 `dims=[16,24,32,48,64]不能收敛 dims=[64,64,64,64,64]收敛更差`|1e5||||python train_tensorboard.py --net=Res18Net1 --step=100000 --device=cuda:0 --pth=Res18Net1_384p_1e5_l1 --divisor=16 --bs=8 --l1loss --crop_size=384|


### train_patch  | receptive field | test_image=600x400


在整张图上的train/test loss曲线都是先减小后变大 abnormal

`都是存在着对训练patch对过拟合`

downsample :2^(depth-1)
receptive field of `encoder`:
|depth|receptive field|
|-|-|
|5|140|
|6|284|
|7|572|
|8|1148|

`有过拟合风险`

`分析` pending tensorboard画不出来
相同depth下：
相同patch下：

|net|patch|step|psnr|ssim|time|line|
|-|-|-|-|-|-|-|
|UNet_Depth=5 验证代码没错|160|1e5|19.5771|0.8110|4h|python train_tensorboard.py --net='UNet_Depth' --depth=5 --step=100000 --device=cuda:0 --pth=UNet_Depth5_160p_1e5_l1 --divisor=16 --bs=8 --l1loss --crop_size=160|
|UNet_Depth=6|128|1e5|16.66|0.7402|4h|python train_tensorboard.py --net='UNet_Depth' --depth=6 --step=100000 --device=cuda:0 --pth=UNet_Depth6_128p_1e5_l1 --divisor=32 --bs=8 --l1loss --crop_size=128|
|UNet_Depth=6|256|1e5|20.56| 0.79|4h|python train_tensorboard.py --net='UNet_Depth' --depth=6 --step=100000 --device=cuda:0 --pth=UNet_Depth6_256p_1e5_l1 --divisor=32 --bs=8 --l1loss --crop_size=256|
|UNet_Depth=6|`384尽可能对接近test图像大小`|1e5|`23.82`|`0.841`|4h|python train_tensorboard.py --net='UNet_Depth' --depth=6 --step=100000 --device=cuda:0 --pth=UNet_Depth6_384p_1e5_l1 --divisor=32 --bs=8 --l1loss --crop_size=384|
|UNet_Depth=7|128|1e5|18.39|0.7235|4h|python train_tensorboard.py --net='UNet_Depth' --depth=7 --step=100000 --device=cuda:0 --pth=UNet_Depth7_128p_1e5_l1 --divisor=64 --bs=8 --l1loss --crop_size=128|
|UNet_Depth=7|256|1e5|17.9177|0.8016|4h|python train_tensorboard.py --net='UNet_Depth' --depth=7 --step=100000 --device=cuda:0 --pth=UNet_Depth7_256p_1e5_l1 --divisor=64 --bs=8 --l1loss --crop_size=256|
|UNet_Depth=7|`384`|1e5|`22.460`|`0.8258`|4h|python train_tensorboard.py --net='UNet_Depth' --depth=7 --step=100000 --device=cuda:0 --pth=UNet_Depth7_384p_1e5_l1 --divisor=64 --bs=8 --l1loss --crop_size=384|
|UNet_Depth=8|128|1e5|18.4298|0.7430|4h|python train_tensorboard.py --net='UNet_Depth' --depth=8 --step=100000 --device=cuda:0 --pth=UNet_Depth8_128p_1e5_l1 --divisor=128 --bs=8 --l1loss --crop_size=128|
|UNet_Depth=8|256|1e5|16.669|0.7882|4h|python train_tensorboard.py --net='UNet_Depth' --depth=8 --step=100000 --device=cuda:0 --pth=UNet_Depth8_256p_1e5_l1 --divisor=128 --bs=8 --l1loss --crop_size=256|

