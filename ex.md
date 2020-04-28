### LOL 

对于数据集的指标，使用没有光照scalar的模型做infer。和其他人的一样。
ulimit -n 2048
nohup python >log.out

过拟合1:patch与total_image关于image_size的过拟合
过拟合2:train/test的过拟合
`
gen_y_unet_160p_2e5_l1.pth
step : 88000
max_psnr : 20.69818210016854(+1)
max_ssim : 0.8325258769386841(+0.03)
`

|net|step|psnr|ssim|time|line|
|-|-|-|-|-|-|
|unet`其结果模糊不清尽管能够做到denoise`|1e5|19.6919|0.8089|4h|python train.py --net='unet' --step=100000 --pth=unet_160p_1e5_l1 --divisor=16 --bs=8 --l1loss --crop_size=160|
|`unet64`|1e5|20.1542|0.7777|4h|python train_tensorboard.py --net='unet64' --step=100000 --device=cuda:0 --pth=unet64_160p_1e5_l1 --divisor=16 --bs=8 --l1loss --crop_size=160|
|FullConv_SwiftNet|1e5|0.7805|19.5069|5h|python train_tensorboard.py --net='FullConv_SwiftNet' --device=cuda:0 --step=100000 --pth=FullConv_SwiftNet_160p_1e5_l1 --divisor=32 --bs=8 --l1loss --crop_size=160 --lr=0.0004|
|`hdr1`没有训练patch和整张图的过拟合（因为子分辨率变换算子与原分辨率算子相似），但是有train/test的过拟合，`grid_sample采用的是bilinear（x,y）,最关键的guided-map（z）没有采样！`|1e5|`25.6277结果较清晰`|`0.7534但存在很大的噪声`|19h|python train_tensorboard.py --net='hdr1' --step=100000 --device=cuda:0 --pth=hdr1_384p_1e5_l1 --divisor=1 --bs=8 --l1loss --crop_size=384|
|Res18Net1 `dims=[16,24,32,48,64]不能收敛 dims=[64,64,64,64,64]收敛更差`|1e5|||10h|python train_tensorboard.py --net=Res18Net1 --step=100000 --device=cuda:0 --pth=Res18Net1_384p_1e5_l1 --divisor=16 --bs=8 --l1loss --crop_size=384|


### Swiftnet +guided filters

`收敛标准在0.02左右`
`InstanceNorm能够有效解决过拟合1`
`因为使用了Batch norm并不能够有效解决过拟合2`


|net|step|psnr|ssim|time|line|
|-|-|-|-|-|-|
|`swiftnet`output是1/4 上采样 `结果模糊的要死`|1e5|19.2726|0.7124|8h|python train_tensorboard.py --net='swiftnet' --device=cuda:0 --step=100000 --pth=swiftnet_160p_1e5_l1 --divisor=32 --bs=8 --l1loss --crop_size=160 --lr=0.0004|
|SwiftNet_GuidedFilter`output使用GuidedFilter,回归x4,loss都是在x4下，psnr和ssim在原分辨率下`回归到0.01非常好，过拟合1：train_loss:0.05很差，过拟合2也存在但较轻eval_loss:0.06|1e5|`22.3718`|0.7298|18h|python gf_train_tensorboard.py --net=SwiftNet_GuidedFilter --device=cuda:1 --step=100000 --pth=SwiftNet_GuidedFilter_384p_1e5_l1 --divisor=32 --bs=8 --l1loss --crop_size=384 --lr=0.0004|
|swiftnetslim `特征变为[16,32,64,64]原来[64,128,256,512] 约等于w=0.2也是能回归的0.02`|2e5|17.7897|0.6939|3h|python train_tensorboard.py --net='swiftnetslim' --device=cuda:0 --step=200000 --pth=swiftnetslim_160p_2e5_l1 --divisor=32 --bs=8 --l1loss --crop_size=160 --lr=0.0004|
|SwiftNetSlim_GuidedFilterLayerAndMap`在out下进行回归,swiftslim没有使用norm,GFL使用AdaptiveNorm``patch_loss:0.024 train_loss:0.056(过拟合1) psnr:23.7 ssim:0.86 test_loss:0.067 psnr：22.2（-1.5）ssim:0.81（-0.05）`|1e5|22.9125|0.8107|18h|python gfl_train_tensorboard.py --net=SwiftNetSlim_GuidedFilterLayerAndMap --device=cuda:0 --step=100000 --pth=SwiftNetSlim_GuidedFilterLayerAndMap_384p_1e5_l1 --divisor=32 --bs=8 --l1loss --crop_size=384 --lr=0.0004|
|SwiftNetSlim_GuidedFilterLayerAndMap`在out下进行回归,swiftslim使用InstanceNorm,GFL使用AdaptiveNorm``patch_loss:0.023 train_loss:0.023(没有过拟合1) psnr:29.57 ssim:0.872 test_loss:0.40psnr:25.8(-4过拟合2严重)ssim:0.815（-0.06严重）`|1e5|`25.8456`|0.8169|18h|python gfl_train_tensorboard.py --net=SwiftNetSlim_GuidedFilterLayerAndMap --device=cuda:1 --step=100000 --pth=SwiftNetSlim_GuidedFilterLayerAndMap_384p_1e5_l1_IN --divisor=32 --bs=8 --l1loss --crop_size=384 --lr=0.0004 --norm|
|SwiftNetSlim_GFLAndMap_BN `,在out下进行回归,decode使用bn而不是in,encoder使用InstanceNorm,GFL使用AdaptiveNorm``patch_loss:0.02-0.024 train_loss:0.024 psnr:29.7 ssim:0.87 test_loss:0.037(过拟合2）psnr:26.4（-3）ssim:0.819（-0.05）`|1e5|`26.4670`|0.8207|18h|python gfl_train_tensorboard.py --net=SwiftNetSlim_GFLAndMap_BN --device=cuda:0 --step=100000 --pth=SwiftNetSlim_GFLAndMap_BN_384p_1e5_l1_IN --divisor=32 --bs=8 --l1loss --crop_size=384 --lr=0.0004 --norm|
|SwiftNetSlim_GFLAndMap_BN `在out下进行回归,decode使用bn而不是in,encoder使用InstanceNorm,GFL使用AdaptiveNorm SSIMLOSS``patch_loss:0.15 train_loss:0.13 psnr:30.0 ssim:0.88 test_loss:0.20 psnr:26(-4)ssim:0.83(-0.05)`|1e5|26.0504|`0.8297`|18h|python gfl_train_tensorboard.py --net=SwiftNetSlim_GFLAndMap_BN --device=cuda:0 --step=100000 --pth=SwiftNetSlim_GFLAndMap_BN_384p_1e5_l1_ssim_IN --divisor=16 --bs=8 --l1loss --crop_size=384 --lr=0.0004 --norm --ssimloss|
|SwiftNetSlim_GFL_SN 全部替换为`SwitchableNorm保留adaptiveNorm形式，SSIMLOSS``patch_loss:0.14 train_loss:0.12 psnr:30.3 ssim:0.89 test_loss:0.20 psnr:26.3(-4)ssim:0.83(-0.06)`|1e5|26.4075|`0.8298`|18h|python gfl_train_tensorboard.py --net=SwiftNetSlim_GFL_SN --device=cuda:1 --step=100000 --pth=SwiftNetSlim_GFL_SN_384p_1e5_l1_ssim --divisor=16 --bs=8 --l1loss --crop_size=384 --lr=0.0004 --norm --ssimloss|
|SwiftNetSlim_GFLAndMap_BN2 `,在out下进行回归,decode使用bn而不是in,encoder使用InstanceNorm,GFL使用AdaptiveNorm(改动)` `256P` SSIMLOSS `patch_loss:0.13-0.15 train_loss:0.15 psnr:27.15 ssim:0.877 test_loss:0.212 psnr:25.36(-2)ssim:0.82(-0.05)`|1e5|25.6620|`0.8249`|18h|python gfl_train_tensorboard.py --net=SwiftNetSlim_GFLAndMap_BN2 --device=cuda:1 --step=100000 --pth=SwiftNetSlim_GFLAndMap_BN2_256p_1e5_l1_ssim_IN --divisor=1 --bs=8 --l1loss --crop_size=256 --lr=0.0004 --norm --ssimloss|
|SwiftNetSlim_GFLAndMap_BN2 `Backbone``,在out下进行回归,decode使用bn而不是in,encoder使用InstanceNorm,GFL使用AdaptiveNorm(改动)` `384P` SSIMLOSS `patch_loss:0.14 train_loss:0.13 psnr:30. ssim:0.88 test_loss:0.20 psnr:26.2ssim:0.83`|1e5|`26.2817`|`0.8316`|18h|python gfl_train_tensorboard.py --net=SwiftNetSlim_GFLAndMap_BN2 --device=cuda:1 --step=100000 --pth=SwiftNetSlim_GFLAndMap_BN2_384p_1e5_l1_ssim_IN --divisor=16 --bs=8 --l1loss --crop_size=384 --lr=0.0004 --norm --ssimloss|
|SwiftNetSlim2_GFLAndMap_BN2 `在前一个基础上conv改为3x3而不是7x7` `256P` SSIMLOSS  `patch_loss: 0.15 train_loss:0.145 psnr:28.47 ssim:0.875 test_loss:0.21 psnr:25.6(-3)ssim:0.822(-0.05)`|1e5|25.7234|0.8224|18h|python gfl_train_tensorboard.py --net=SwiftNetSlim2_GFLAndMap_BN2 --device=cuda:1 --step=100000 --pth=SwiftNetSlim2_GFLAndMap_BN2_256p_1e5_l1_ssim_IN --divisor=16 --bs=8 --l1loss --crop_size=256 --lr=0.0004 --norm --ssimloss|
|SwiftNetSlim2_GFLAndMap_BN2 `Backbone` 在前一个基础 `384P` SSIMLOSS  `patch_loss:0.15train_loss:0.13 psnr:29.4 ssim:0.88 test_loss:0.21 psnr:25.8ssim:0.82`|1e5|`25.9860`|0.8245|18h|python gfl_train_tensorboard.py --net=SwiftNetSlim2_GFLAndMap_BN2 --device=cuda:0 --step=100000 --pth=SwiftNetSlim2_GFLAndMap_BN2_384p_1e5_l1_ssim_IN --divisor=16 --bs=8 --l1loss --crop_size=384 --lr=0.0004 --norm --ssimloss|



### Guided Filter Network

`收敛标准在0.02左右`
`因为使用了norm所以有更好解决过拟合2`

|net|step|psnr|ssim|time|line|
|-|-|-|-|-|-|
|DeepGuidedFilter 回归到`0.05左右`，过拟合1:train_loss在0.07，过拟合2：存在但不稳定，有时候轻微有时候严重|1e5|21.8926|0.7401|20h|python train_tensorboard.py --net=DeepGuidedFilter --step=100000 --device=cuda:0 --pth=DeepGuidedFilter_384p_1e5_l1_025 --divisor=1 --bs=8 --l1loss --crop_size=384 --scale_factor=0.25|
|DeepGuidedFilterAndMap 回归到`0.05左右`，过拟合1效果同上，但过拟合2交情较轻|1e5|21.9899|0.7865`去躁了所以好了点`|20h|python train_tensorboard.py --net=DeepGuidedFilterAndMap --step=100000 --device=cuda:1 --pth=DeepGuidedFilterAndMap_384p_1e5_l1_025 --divisor=1 --bs=8 --l1loss --crop_size=384 --scale_factor=0.25|
|DeepGuidedFilterLayer回归到`0.05左右`，过拟合1:train_loss在0.07上，过拟合2：几乎没有甚至eval比train在loss上更低|1e5|`22.1607`|0.7719|20h|python train_tensorboard.py --net=DeepGuidedFilterLayer --step=100000 --device=cuda:0 --pth=DeepGuidedFilterLayer_384p_1e5_l1_025 --divisor=1 --bs=8 --l1loss --crop_size=384 --scale_factor=0.25|
|DeepGuidedFilterLayerAndMap回归到`0.05左右`过拟合1:train_loss在0.07上，过拟合2：几乎没有甚至eval比train在loss上更低|1e5|21.9514|`0.7808去躁了所以好了点`|20h|python train_tensorboard.py --net=DeepGuidedFilterLayerAndMap --step=100000 --device=cuda:1 --pth=DeepGuidedFilterLayerAndMap_384p_1e5_l1_025 --divisor=1 --bs=8 --l1loss --crop_size=384 --scale_factor=0.25|

