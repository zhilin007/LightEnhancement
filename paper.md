## 一些提醒：
 * `model放在了models/Defined_Network下`
 - [x] SwiftNetSlim_GFLAndMap_BN2下guided_map是从RGB三个通道生成的！,之后Gen_Y_Swiftslim_BN2的guided_map是在RGBY四个通道下生成的！！！基本一样
 - [ ] 存在scale_factor超参数，默认是0.25（4倍下采样）
 - [x] 未知GT光亮程度下作为回归问题的结果差距 较明显
 - [x] 更大的BS没有大的性能提升
 - [x] `2-steptraining strategy没用`
 




### 最后确定的BackBone7x7 BS=8
`res18_Slim7x7 + SPP + Upsample(swiftnet) 384p`

`SSIMLOSS L1Loss InstanceNorm BatchNorm AdaptiveNorm(也许不是这么这么叫)`

|net|regression|step|psnr|ssim|time|line|
|-|-|-|-|-|-|-|
|Backbone7x7 |`patch_loss:0.13 train_loss:0.13 psnr:30. ssim:0.88 test_loss:0.20 psnr::26.2ssim:0.83`|1e5|26.2374|0.8311|18h|python gfl_train_tensorboard.py --net=Backbone7x7 --device=cuda:0 --step=100000 --pth=Backbone7x7_inC4_384p_1e5_l1_ssim_IN --divisor=16 --bs=8 --l1loss --crop_size=384 --lr=0.0004 --norm --ssimloss --incolor=4|
|Backbone7x7过拟合严重，拟合能力也不强 `实验条件和上相同` `没有光照调节下incolor=3，ImageNet光亮替代`|`patch_loss:0.15 train_loss:0.15 psnr:27.4 ssim:0.88 test_loss:0.27 psnr:21.3ssim:0.80`|1e5|21.9252|0.8018|18h|python gfl_train_tensorboard.py --net=Backbone7x7 --device=cuda:2 --step=100000 --pth=Backbone7x7_inC3_384p_1e5_l1_ssim_IN --divisor=16 --bs=8 --l1loss --crop_size=384 --lr=0.0004 --norm --ssimloss --incolor=3|
|Backbone7x7`incolor=3 扩大BS=96` |BS=96没啥大作用|1e5|21.9674|0.8039|18h|python gfl_train_tensorboard.py --net=Backbone7x7 --device=cuda:0 --step=100000 --pth=Backbone7x7_inC3_384p_1e5_l1_ssim_IN_96bs --divisor=16 --bs=96 --l1loss --crop_size=384 --lr=0.0004 --norm --ssimloss --`incolor=3`|

<!-- |Backbone7x7 `EML1loss 替代l1loss`|patch_loss: train_loss: psnr: ssim: test_loss: psnr: ssim:|1e5|26.3232|0.8317|18h|python gfl_train_tensorboard.py --net=Backbone7x7 --device=cuda:0 --step=100000 --pth=Backbone7x7_inC4_384p_1e5_eml1_ssim_IN --divisor=16 --bs=8 --eml1loss --crop_size=384 --lr=0.0004 --norm --ssimloss --incolor=4| -->
<!-- |Backbone7x7 `太差``实验条件和上相同` `训练使用gt光亮,infer使用imagenet光亮`|`patch_loss:0.15 train_loss:0.23 psnr:21.2 ssim:0.85 test_loss:0.32 psnr:18.63ssim:0.78`|1e5|18.7321|0.7633|18h|python gfl_train_tensorboard`2`.py --net=Backbone7x7 --device=cuda:1 --step=100000 --pth=Backbone7x7_inC3_2_384p_1e5_l1_ssim_IN --divisor=16 --bs=8 --l1loss --crop_size=384 --lr=0.0004 --norm --ssimloss --incolor=3| -->





### GEN Y: BackBone7x7 `未知GT光亮,incolor=3下使用ImageNet光亮性能`

* Gen_Y_Swiftslim_BN2=Gen_Y_Backbone7x7
* Gen_Y_Swiftslim_BN2_Share=Gen_Y_Backbone7x7_Share

`Y:l1 0ut: l1+ssimloss `

`scale_factor超参！：Y在下采样下(0.25)处理，结果使用bilinear上采样`

`Share: 共享encoder+SPP`

|net|regression|step|psnr|ssim|time|line|
|-|-|-|-|-|-|-|
|Gen_Y_Backbone7x7|`patch_loss:train_loss: psnr: ssim: test_loss: psnr:ssim:`|2e5|`22.2285`|0.8036|18h|python Gen_Y_train_tensorboard.py --device='cuda:1' --steps=200000 --lr=0.0004 --pth=Gen_Y_Backbone7x7_inC3_384p_2e5_l1_ssim --divisor=16 --bs=8 --l1loss --crop_size=384 --norm --net=Gen_Y_Backbone7x7 --scale_factor=0.25 --ssimloss --incolor=3|
|Gen_Y_Backbone7x7_Share|`patch_loss:train_loss: psnr: ssim: test_loss: psnr:ssim:`|2e5|21.8304|`0.8054`|18h|python Gen_Y_Share_train_tensorboard.py --device='cuda:2' --steps=200000 --lr=0.0004 --pth=Gen_Y_Backbone7x7_Share_inC3_384p_2e5_l1_ssim --divisor=16 --bs=8 --l1loss --crop_size=384 --norm --net=Gen_Y_Backbone7x7_Share --scale_factor=0.25 --ssimloss --incolor=3|

<!-- |Gen_Y_Backbone7x7 `2-step train`|`patch_loss:train_loss: psnr: ssim: test_loss: psnr:ssim:`|2e5|21.5746|0.8084|18h|python Gen_Y_train_tensorboard_2steps.py --device='cuda:1' --steps=200000 --lr=0.0004 --pth=Gen_Y_Backbone7x7_2steps_inC3_384p_2e5_l1_ssim --divisor=16 --bs=8 --l1loss --crop_size=384 --norm --net=Gen_Y_Backbone7x7 --scale_factor=0.25 --ssimloss --incolor=3| -->
<!-- |Gen_Y_Backbone7x7_Share `2-step train`|`patch_loss:train_loss: psnr: ssim: test_loss: psnr:ssim:`|2e5|19.8739|0.7878|18h|python Gen_Y_Share_train_tensorboard_2steps.py --device='cuda:2' --steps=200000 --lr=0.0004 --pth=Gen_Y_Backbone7x7_Share_2steps_inC3_384p_2e5_l1_ssim --divisor=16 --bs=8 --l1loss --crop_size=384 --norm --net=Gen_Y_Backbone7x7_Share --scale_factor=0.25 --ssimloss --incolor=3| -->


### GEN Y: BackBone7x7 在光亮调节incolor=4下

* Gen_Y_Swiftslim_BN2=Gen_Y_Backbone7x7
* Gen_Y_Swiftslim_BN2_Share=Gen_Y_Backbone7x7_Share

`Y:l1 0ut: l1+ssimloss `

`scale_factor超参！：Y在下采样下(0.25)处理，结果使用bilinear上采样`

`共享encoder+SPP或not`

|net|regression|step|psnr|ssim|time|line|
|-|-|-|-|-|-|-|
|Gen_Y_Backbone7x7|`patch_loss:train_loss: psnr: ssim: test_loss: psnr:ssim:`|2e5|26.2966|0.8330|18h|python Gen_Y_train_tensorboard.py --device='cuda:1' --steps=200000 --lr=0.0004 --pth=Gen_Y_Backbone7x7_inC4_384p_2e5_l1_ssim --divisor=16 --bs=8 --l1loss --crop_size=384 --norm --net=Gen_Y_Backbone7x7 --scale_factor=0.25 --ssimloss --incolor=4|
|Gen_Y_Backbone7x7_Share|`patch_loss:train_loss: psnr: ssim: test_loss: psnr:ssim:`|2e5|`26.4421`|`0.8338`|18h|python Gen_Y_Share_train_tensorboard.py --device='cuda:2' --steps=200000 --lr=0.0004 --pth=Gen_Y_Backbone7x7_Share_inC4_384p_2e5_l1_ssim --divisor=16 --bs=8 --l1loss --crop_size=384 --norm --net=Gen_Y_Backbone7x7_Share --scale_factor=0.25 --ssimloss --incolor=4|
|Gen_Y_Backbone7x7 `EML1LOSS`||2e5|26.5438|0.8343|18h|python Gen_Y_train_tensorboard.py --device='cuda:0' --steps=200000 --lr=0.0004 --pth=Gen_Y_Backbone7x7_inC4_384p_2e5_eml1_ssim --divisor=16 --bs=8 --eml1loss --crop_size=384 --norm --net=Gen_Y_Backbone7x7 --scale_factor=0.25 --ssimloss --incolor=4|
|Gen_Y_Backbone7x7_Share`EML1LOSS`||2e5|26.1463|0.8321|18h|python Gen_Y_Share_train_tensorboard.py --device='cuda:1' --steps=200000 --lr=0.0004 --pth=Gen_Y_Backbone7x7_Share_inC4_384p_2e5_eml1_ssim --divisor=16 --bs=8 --eml1loss --crop_size=384 --norm --net=Gen_Y_Backbone7x7_Share --scale_factor=0.25 --ssimloss --incolor=4|

<!-- |Gen_Y_Backbone7x7 `2-step train`|`patch_loss:train_loss: psnr: ssim: test_loss: psnr:ssim:`|2e5|26.0745|0.8300|18h|python Gen_Y_train_tensorboard_2steps.py --device='cuda:1' --steps=200000 --lr=0.0004 --pth=Gen_Y_Backbone7x7_2steps_inC4_384p_2e5_l1_ssim --divisor=16 --bs=8 --l1loss --crop_size=384 --norm --net=Gen_Y_Backbone7x7 --scale_factor=0.25 --ssimloss --incolor=4| -->
<!-- |Gen_Y_Backbone7x7_Share `2-step train`|`patch_loss:train_loss: psnr: ssim: test_loss: psnr:ssim:`|2e5|25.6229|0.8279|18h|python Gen_Y_Share_train_tensorboard_2steps.py --device='cuda:2' --steps=200000 --lr=0.0004 --pth=Gen_Y_Backbone7x7_Share_2steps_inC4_384p_2e5_l1_ssim --divisor=16 --bs=8 --l1loss --crop_size=384 --norm --net=Gen_Y_Backbone7x7_Share --scale_factor=0.25 --ssimloss --incolor=4| -->


### FPS with norm: on 1920x1080 resolution

`这种下采样genY的方式并没有增加多少运算量但是能够提升性能`

｜net|device|resolution|FPS|avg_infer_decay|
|-|-|-|-|-|
|Backbone7x7|GeForce GTX TITAN X|[1,4,1920,1080]|34.86|0.028|
|Gen_Y_Backbone7x7主干网络是Backbone7x7|GeForce GTX TITAN X|[1,3,1920,1080]|`31.5`|`0.031`|
|Gen_Y_Backbone7x7_Share `FPS确实一样`|GeForce GTX TITAN X|[1,3,1920,1080]|`31.5`|`0.031`|

### FLOPS and Parameters on 1920x1080
`InstanceNorm不支持!!!!`
|net|InputSize|FLOPs|memory|paras|
|-|-|-|-|-|-|
|`Backbone7x7|4x1920x1056|7.57G|864.06MB|427,534|
|`Gen_Y_Backbone7x7|3x1920x1056|||853,458|
|`Gen_Y_Backbone7x7_Share`|3x1920x1056|||494,981|



