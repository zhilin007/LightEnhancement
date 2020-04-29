### BackBone

|net||regressionstep|psnr|ssim|time|line|
|-|-|-|-|-|-|-|
|SwiftNetSlim_GFLAndMap_BN2 `Backbone7x7``,在out下进行回归,decode使用bn而不是in,encoder使用InstanceNorm,GFL使用AdaptiveNorm(改动)` `384P` SSIMLOSS |`patch_loss:0.14 train_loss:0.13 psnr:30. ssim:0.88 test_loss:0.20 psnr:26.2ssim:0.83`|1e5|`26.2817`|`0.8316`|18h|python gfl_train_tensorboard.py --net=SwiftNetSlim_GFLAndMap_BN2 --device=cuda:1 --step=100000 --pth=SwiftNetSlim_GFLAndMap_BN2_384p_1e5_l1_ssim_IN --divisor=16 --bs=8 --l1loss --crop_size=384 --lr=0.0004 --norm --ssimloss|
|SwiftNetSlim2_GFLAndMap_BN2 `Backbone3x3` 在前一个基础 `384P` SSIMLOSS |`patch_loss:0.15train_loss:0.13 psnr:29.4 ssim:0.88 test_loss:0.21 psnr:25.8ssim:0.82`|1e5|`25.9860`|0.8245|18h|python gfl_train_tensorboard.py --net=SwiftNetSlim2_GFLAndMap_BN2 --device=cuda:0 --step=100000 --pth=SwiftNetSlim2_GFLAndMap_BN2_384p_1e5_l1_ssim_IN --divisor=16 --bs=8 --l1loss --crop_size=384 --lr=0.0004 --norm --ssimloss|

### GEN Y BackBone3x3 l1loss

`bilinear实验结果比较好，在genY上不使用GFL`

|net|regression|step|psnr|ssim|time|line|
|-|-|-|-|-|-|-|
|Gen_Y_Swiftslim2_BN2`Y在下采样下处理，结果使用bilinear上采样`|`patch_loss:0.075train_loss:0.071 psnr:29.8 ssim:0.87 test_loss:0.083 psnr:26.4ssim:0.82`|2e5|`26.5037`|`0.8232`|18h|python Gen_Y_train_tensorboard.py --device='cuda:0' --steps=200000 --lr=0.0004 --pth=Gen_Y_Swiftslim2_BN2_384p_2e5_l1 --divisor=16 --bs=8 --l1loss --crop_size=384 --norm --net=Gen_Y_Swiftslim2_BN2 --scale_factor=0.25 |
|Gen_Y_Swiftslim2_BN2_Share`Y结果使用上采样 & 共享encoder SPP`|`patch_loss:0.075train_loss:0.075 psnr:29.28 ssim:0.87 test_loss:0.087 psnr:26.0ssim:0.82`|2e5|`26.1818`|`0.8220`|18h|python Gen_Y_Share_train_tensorboard.py --device='cuda:2' --steps=200000 --lr=0.0004 --pth=Gen_Y_Swiftslim2_BN2_Share_384p_2e5_l1 --divisor=16 --bs=8 --l1loss --crop_size=384 --norm --net=Gen_Y_Swiftslim2_BN2_Share --scale_factor=0.25 |
|Gen_Y_Swiftslim2_BN2_SAME`Y网络和主网络一样`|`patch_loss:0.047train_loss:0.041 psnr:29.8 ssim:0.87 test_loss:0.06 psnr:26.1ssim:0.81`|2e5|26.1502|0.8163|18h|python Gen_Y_train_tensorboard.py --device='cuda:1' --steps=200000 --lr=0.0004 --pth=Gen_Y_Swiftslim2_BN2_SAME_384p_2e5_l1 --divisor=16 --bs=8 --l1loss --crop_size=384 --norm --net=Gen_Y_Swiftslim2_BN2_SAME|
|Gen_Y_Swiftslim2_Bn2_SAME_share`共享encoder SPP`|`patch_loss:train_loss: psnr: ssim: test_loss: psnr:ssim:`|
|Gen_Y_Swiftslim2_BN2_SAME_DownSample`genY:在下采样分辨率下处理，结果使用GFL`|`patch_loss:0.052train_loss:0.076 psnr:27.86 ssim:0.867 test_loss:0.079 psnr:25.49ssim:0.814`|2e5|25.8214|0.8128|18h|python Gen_Y_train_tensorboard.py --device='cuda:0' --steps=200000 --lr=0.0004 --pth=Gen_Y_Swiftslim2_BN2_SAME_DownSample_384p_2e5_l1 --divisor=16 --bs=8 --l1loss --crop_size=384 --norm --net=Gen_Y_Swiftslim2_BN2_SAME_DownSample --scale_factor=0.25|

### GEN Y: BackBone7x7 Y:l1 0: l1+ssimloss 

|net|regression|step|psnr|ssim|time|line|
|-|-|-|-|-|-|-|
|Gen_Y_Swiftslim_BN2`Y在下采样下处理，结果使用bilinear上采样`|`patch_loss:train_loss: psnr: ssim: test_loss: psnr:ssim:`|2e5|||18h|python Gen_Y_train_tensorboard.py --device='cuda:0' --steps=200000 --lr=0.0004 --pth=Gen_Y_Swiftslim_BN2_384p_2e5_l1_ssim --divisor=16 --bs=8 --l1loss --crop_size=384 --norm --net=Gen_Y_Swiftslim_BN2 --scale_factor=0.25 --ssimloss|
|Gen_Y_Swiftslim_BN2_Share`Y结果使用上采样 & 共享encoder SPP`|`patch_loss:train_loss: psnr: ssim: test_loss: psnr:ssim:`|2e5|||18h|python Gen_Y_Share_train_tensorboard.py --device='cuda:2' --steps=200000 --lr=0.0004 --pth=Gen_Y_Swiftslim_BN2_Share_384p_2e5_l1 --divisor=16 --bs=8 --l1loss --crop_size=384 --norm --net=Gen_Y_Swiftslim_BN2_Share --scale_factor=0.25 --ssimloss|
