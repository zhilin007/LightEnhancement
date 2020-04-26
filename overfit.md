
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

