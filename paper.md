## 一些提醒：
 * `model放在了models/Defined_Network下`
 * [ ] SwiftNetSlim_GFLAndMap_BN2下guided_map是从RGB三个通道生成的！,之后Gen_Y_Swiftslim_BN2的guided_map是在RGBY四个通道下生成的！！！（未验证）
 * 存在scale_factor超参数，默认是0.25（4倍下采样）
 * [ ] 未知GT光亮程度下的结果差距
 * [ ] 更大的BS得到最终的结果




### 最后确定的BackBone 是7x7
`res18_Slim7x7 + SPP + Upsample(swiftnet)`

`SSIMLOSS InstanceNorm BatchNorm AdaptiveNorm(也许不是这么这么叫)`

|net|regressionstep|psnr|ssim|time|line|
|-|-|-|-|-|-|-|
|SwiftNetSlim_GFLAndMap_BN2 `Backbone7x7``,在out下进行回归,decode使用bn而不是in,encoder使用InstanceNorm,GFL使用AdaptiveNorm(改动)` `384P` SSIMLOSS |`patch_loss:0.14 train_loss:0.13 psnr:30. ssim:0.88 test_loss:0.20 psnr:26.2ssim:0.83`|1e5|`26.2817`|`0.8316`|18h|python gfl_train_tensorboard.py --net=SwiftNetSlim_GFLAndMap_BN2 --device=cuda:1 --step=100000 --pth=SwiftNetSlim_GFLAndMap_BN2_384p_1e5_l1_ssim_IN --divisor=16 --bs=8 --l1loss --crop_size=384 --lr=0.0004 --norm --ssimloss|
|Backbone7x7 `实验条件和上相同` `验证引导图incolor`|`patch_loss: train_loss: psnr: ssim: test_loss: psnr:ssim:`|1e5|||18h|python gfl_train_tensorboard.py --net=Backbone7x7 --device=cuda:0 --step=100000 --pth=Backbone7x7_inC4_384p_1e5_l1_ssim_IN --divisor=16 --bs=8 --l1loss --crop_size=384 --lr=0.0004 --norm --ssimloss --incolor=4|
|Backbone7x7 `实验条件和上相同` `验证未知GT光亮incolor=3下ImageNet光亮下性能`|`patch_loss: train_loss: psnr: ssim: test_loss: psnr:ssim:`|1e5|||18h|python gfl_train_tensorboard.py --net=Backbone7x7 --device=cuda:2 --step=100000 --pth=Backbone7x7_inC3_384p_1e5_l1_ssim_IN --divisor=16 --bs=8 --l1loss --crop_size=384 --lr=0.0004 --norm --ssimloss --incolor=3|

### GEN Y: BackBone7x7 

`Y:l1 0ut: l1+ssimloss `

`scale_factor超参！：Y在下采样下(0.25)处理，结果使用bilinear上采样`

`共享encoder+SPP或not`


|net|regression|step|psnr|ssim|time|line|
|-|-|-|-|-|-|-|
|Gen_Y_Swiftslim_BN2`Y在下采样下处理，结果使用bilinear上采样`|`patch_loss:train_loss: psnr: ssim: test_loss: psnr:ssim:`|2e5|||18h|python Gen_Y_train_tensorboard.py --device='cuda:0' --steps=200000 --lr=0.0004 --pth=Gen_Y_Swiftslim_BN2_384p_2e5_l1_ssim --divisor=16 --bs=8 --l1loss --crop_size=384 --norm --net=Gen_Y_Swiftslim_BN2 --scale_factor=0.25 --ssimloss|
|Gen_Y_Swiftslim_BN2_Share`Y结果使用上采样 & 共享encoder SPP`|`patch_loss:train_loss: psnr: ssim: test_loss: psnr:ssim:`|2e5|||18h|python Gen_Y_Share_train_tensorboard.py --device='cuda:2' --steps=200000 --lr=0.0004 --pth=Gen_Y_Swiftslim_BN2_Share_384p_2e5_l1 --divisor=16 --bs=8 --l1loss --crop_size=384 --norm --net=Gen_Y_Swiftslim_BN2_Share --scale_factor=0.25 --ssimloss|


### FPS with norm: on 1080p 以后替换为BackBone

`这种下采样genY的方式并没有增加多少运算量但是能够提升性能`

｜net|device|resolution|FPS|avg_infer_decay|
|-|-|-|-|-|
|`SwiftNetSlim_GFLAndMap_BN2`|GeForce GTX TITAN X|[1,4,1920,1080]|34.86|0.028|
|`Gen_Y_Swiftslim_BN2 `主干网络是SwiftNetSlim_GFLAndMap_BN2|GeForce GTX TITAN X|[1,3,1920,1080]|`31.5`|`0.031`|
|`Gen_Y_Swiftslim_BN2_Share` `FPS确实一样`|GeForce GTX TITAN X|[1,3,1920,1080]|`31.5`|`0.031`|

### FLOPS and Parameters on 1920x1080 以后替换为BackBone
`InstanceNorm不支持!!!!`
|net|InputSize|FLOPs|memory|paras|
|-|-|-|-|-|-|
|`SwiftNetSlim_GFLAndMap_BN2`|4x1920x1056|7.57G|864.06MB|427,534|
|`Gen_Y_Swiftslim_BN2`|3x1920x1056|||853,458|
|`Gen_Y_Swiftslim_BN2_Share`|3x1920x1056|||494,981|



