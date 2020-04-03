python train_tensorboard.py --net='UNet_Depth' --depth=7 --step=100000 --device=cuda:0 --pth=UNet_Depth7_128p_1e5_l1 --divisor=64 --bs=8 --l1loss --crop_size=128
python train_tensorboard.py --net='UNet_Depth' --depth=7 --step=100000 --device=cuda:0 --pth=UNet_Depth7_256p_1e5_l1 --divisor=64 --bs=8 --l1loss --crop_size=256
python train_tensorboard.py --net='UNet_Depth' --depth=7 --step=100000 --device=cuda:0 --pth=UNet_Depth7_384p_1e5_l1 --divisor=64 --bs=8 --l1loss --crop_size=384
