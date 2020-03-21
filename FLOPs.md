
### RTX2080Ti Float32:13.45 TFLOPS

##### 1920x1080 目标: < 100GFLOPs
|net|InputSize|FLOPs|memory|paras|
|-|-|-|-|-|-|
|unet|4x1920x1072|2.06T|11,471.13MB|34,514,051|
|gen_y_unet|3x1920x1072|4.12T|x2|x2|
|unet64|4x1920x1072|536.25G|7374.58MB|925,507|
|euunet|4x1920x1072|636.17G|8198.99MB|1,073,219|
#### 224x224
|net|InputSize|FLOPs|memory|paras|
|-|-|-|-|-|-|
|unet|4x224x224|50.13G|279.64MB|34,514,051|
|unet64|4x224x224|13.07G|179.78MB|925,507|
|res18|3x224x224|1.82G|25.65MB|11,689,512|
|densenet121|-|2.88G|147.10MB|7,978,856|