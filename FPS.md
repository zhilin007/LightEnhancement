### FPS粗略判断 on 1megapixel

pred_trained 和non-trained infer时间一样

`这测的不准，原来很快的过几天再测试就变得很慢了`

｜net|device|resolution|FPS|avg_infer_decay|
|-|-|-|-|-|
｜unet64|GeForce GTX TITAN X|[1,4,1024,1024]|9|0.12|
｜unet|GeForce GTX TITAN X|[1,4,1024,1024]|3|0.31|
|swiftnet|GeForce GTX TITAN X|[1,4,1024,1024]|42.358|0.0235|
|swiftnetslim|GeForce GTX TITAN X|[1,4,1024,1024]|119.63|0.00827|
|gen_y_unet|GeForce GTX TITAN X|[1,3,1024,1024 & 1,1,1024,1024]|1.61|0.62|
|FullConv_SwiftNet |GeForce GTX TITAN X|[1,4,1024,1024]|21.892|0.04565|
|EFFA|GeForce GTX TITAN X|[1,3,1024,1024]|0.3|2.91|
|hdr1 |GeForce GTX TITAN X|[1,4,1024,1024]|92.9170|0.0105|
|DeepGuidedFilterLayerAndMap|GeForce GTX TITAN X|[1,4,1024,1024]|83.81|0.0118|
|SwiftNet_GuidedFilter|GeForce GTX TITAN X|[1,4,1024,1024]|40.40|0.0246|
|SwiftNetSlim_GuidedFilterLayerAndMap|GeForce GTX TITAN X|[1,4,1024,1024]|75.94|0.013|
|SwiftNetSlim_GFLAndMap_BN|GeForce GTX TITAN X|[1,4,1024,1024]|58.08|0.0171|
|SwiftNetSlim_GFLAndMap_BN使用用NORM|GeForce GTX TITAN X|[1,4,1024,1024]|74.268|0.013|
|SwiftNetSlim2_GFLAndMap_BN2 使用norm|GeForce GTX TITAN X|[1,4,1024,1024]|71.9|0.013|

### FPS of BackBone with norm: on 1080p
｜net|device|resolution|FPS|avg_infer_decay|
|-|-|-|-|-|
|`SwiftNetSlim_GFLAndMap_BN2`|GeForce GTX TITAN X|[1,4,1920,1080]|34.86|0.028|
|SwiftNetSlim2_GFLAndMap_BN2`没有比7x7提速多少`|GeForce GTX TITAN X|[1,4,1920,1080]|36.61|0.0268|
 
### GENY with norm on 1080p!

`这种下采样genY的方式并没有增加多少运算量但是能够提升性能`

｜net|device|resolution|FPS|avg_infer_decay|
|-|-|-|-|-|
|Gen_Y_Swiftslim`2`_BN2`并没有比slim加快多少`|GeForce GTX TITAN X|[1,3,1920,1080]|31.87|0.030|
|Gen_Y_Swiftslim2_BN2_Share|GeForce GTX TITAN X|[1,3,1920,1080]|31.87|0.030|
|Gen_Y_Swiftslim2_BN2_SAME_DownSample|GeForce GTX TITAN X|[1,3,1920,1080]|21.88|0.044|
|Gen_Y_Swiftslim2_BN2_SAME`这样SAME确实慢`|GeForce GTX TITAN X|[1,3,1920,1080]|18.63|0.053|
|`Gen_Y_Swiftslim_BN2 `主干网络是SwiftNetSlim_GFLAndMap_BN2|GeForce GTX TITAN X|[1,3,1920,1080]|`31.5`|`0.031`|
|`Gen_Y_Swiftslim_BN2_Share` `FPS确实一样`|GeForce GTX TITAN X|[1,3,1920,1080]|`31.5`|`0.031`|




