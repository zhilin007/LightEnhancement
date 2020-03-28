### FPS粗略判断

pred_trained 和non-trained infer时间一样

｜net|device|resolution|FPS|avg_infer_decay|
|-|-|-|-|-|
｜unet64|GeForce GTX TITAN X|[1,4,1024,1024]|9|0.12|
｜unet|GeForce GTX TITAN X|[1,4,1024,1024]|3|0.31|
|swiftnet|GeForce GTX TITAN X|[1,4,1024,1024]|`181`|0.0055|
|swiftnetslim|GeForce GTX TITAN X|[1,4,1024,1024]|`42`|0.024|
|gen_y_unet|GeForce GTX TITAN X|[1,3,1024,1024 & 1,1,1024,1024]|1.61|0.62|
|FullConv_SwiftNet |GeForce GTX TITAN X|[1,4,1024,1024]|21.892|0.04565|
|EFFA|GeForce GTX TITAN X|[1,3,1024,1024]|0.3|2.91|
