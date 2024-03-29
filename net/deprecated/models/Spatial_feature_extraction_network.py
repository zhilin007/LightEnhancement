'''
paper :Exploiting temporal consistency for real-time video depth estimation

Spatial feature extraction network. This network consists of three parts, 
including an encoder, a decoder and a multi-scale feature fusion module (MFF). 
In this paper, we employ the relatively shallow model ResNet-18 as the encoder for fast processing.
'''

import torch,torchvision
import torch.nn.functional as F
import torch.nn as nn


class _UpProjection(nn.Sequential):
	def __init__(self, num_input_features, num_output_features):
		super(_UpProjection, self).__init__()
		self.conv1 = nn.Conv2d(num_input_features, num_output_features,
							kernel_size=5, stride=1, padding=2, bias=False)
		self.bn1 = nn.BatchNorm2d(num_output_features)
		self.relu = nn.ReLU(inplace=True)
		self.conv1_2 = nn.Conv2d(num_output_features, num_output_features,
								kernel_size=3, stride=1, padding=1, bias=False)
		self.bn1_2 = nn.BatchNorm2d(num_output_features)

		self.conv2 = nn.Conv2d(num_input_features, num_output_features,
							kernel_size=5, stride=1, padding=2, bias=False)
		self.bn2 = nn.BatchNorm2d(num_output_features)
	def forward(self, x, size):
		x = F.upsample(x, size=size, mode='bilinear', align_corners=True)
		x_conv1 = self.relu(self.bn1(self.conv1(x)))
		bran1 = self.bn1_2(self.conv1_2(x_conv1))
		bran2 = self.bn2(self.conv2(x))
		out = self.relu(bran1 + bran2)
		return out

class E_resnet(nn.Module):
	def __init__(self):
		super(E_resnet, self).__init__()        
		original_model=torchvision.models.resnet18(pretrained=True)
		self.conv1 = original_model.conv1
		self.bn1 = original_model.bn1
		self.relu = original_model.relu
		self.maxpool = original_model.maxpool
		self.layer1 = original_model.layer1
		self.layer2 = original_model.layer2
		self.layer3 = original_model.layer3
		self.layer4 = original_model.layer4
	
	def forward(self, x):
		x_block0 = self.conv1(x)
		x = self.bn1(x_block0)
		x = self.relu(x)
		x = self.maxpool(x)
		x_block1 = self.layer1(x)
		x_block2 = self.layer2(x_block1)
		x_block3 = self.layer3(x_block2)
		x_block4 = self.layer4(x_block3)
		return x_block0,x_block1, x_block2, x_block3, x_block4

class D(nn.Module):
	def __init__(self, num_features = 512):
		super(D, self).__init__()
		self.conv = nn.Conv2d(num_features, num_features //
							2, kernel_size=1, stride=1, bias=False)
		num_features = num_features // 2
		self.bn = nn.BatchNorm2d(num_features)
		self.up1 = _UpProjection(
			num_input_features=num_features, num_output_features=num_features // 2)
		num_features = num_features // 2
		self.up2 = _UpProjection(
			num_input_features=num_features, num_output_features=num_features // 2)
		num_features = num_features // 2
		self.up3 = _UpProjection(
			num_input_features=num_features, num_output_features=num_features // 2)
		num_features = num_features // 2
		self.up4 = _UpProjection(
			num_input_features=num_features, num_output_features=num_features // 2)
		num_features = num_features // 2
		self.up5 = _UpProjection(
			num_input_features=num_features, num_output_features=num_features // 2)
		num_features = num_features // 2


	def forward(self, x_block0,x_block1, x_block2, x_block3, x_block4):
		x_d0 = F.relu(self.bn(self.conv(x_block4)))
		x_d1 = self.up1(x_d0, [x_block3.size(2), x_block3.size(3)])
		x_d2 = self.up2(x_d1, [x_block2.size(2), x_block2.size(3)])
		x_d3 = self.up3(x_d2, [x_block1.size(2), x_block1.size(3)])
		x_d4 = self.up4(x_d3, [x_block0.size(2), x_block0.size(3)])
		x_d5 = self.up5(x_d4, [x_block0.size(2)*2, x_block0.size(3)*2])
		return x_d5

class MFF(nn.Module):
	def __init__(self, block_channel, num_features=80):
		super(MFF, self).__init__()
		self.up1 = _UpProjection(
			num_input_features=block_channel[1], num_output_features=16)
		self.up2 = _UpProjection(
			num_input_features=block_channel[2], num_output_features=16)
		self.up3 = _UpProjection(
			num_input_features=block_channel[3], num_output_features=16)
		self.up4 = _UpProjection(
			num_input_features=block_channel[4], num_output_features=16)
		self.up0 = _UpProjection(
			num_input_features=block_channel[0], num_output_features=16)
		self.conv = nn.Conv2d(
			num_features, num_features, kernel_size=5, stride=1, padding=2, bias=False)
		self.bn = nn.BatchNorm2d(num_features)
	def forward(self,x_block0, x_block1, x_block2, x_block3, x_block4, size):
		x_m0 = self.up0(x_block0, size)
		x_m1 = self.up1(x_block1, size)
		x_m2 = self.up2(x_block2, size)
		x_m3 = self.up3(x_block3, size)
		x_m4 = self.up4(x_block4, size)
		
		x = self.bn(self.conv(torch.cat((x_m0,x_m1, x_m2, x_m3, x_m4), 1)))
		x = F.relu(x)
		return x

class R(nn.Module):
	def __init__(self):
		super(R, self).__init__()
		num_features = 88
		self.conv0 = nn.Conv2d(num_features, num_features,
							kernel_size=5, stride=1, padding=2, bias=False)
		self.bn0 = nn.BatchNorm2d(num_features)
		self.conv1 = nn.Conv2d(num_features, num_features,
							kernel_size=5, stride=1, padding=2, bias=False)
		self.bn1 = nn.BatchNorm2d(num_features)
		self.conv2 = nn.Conv2d(
			num_features, 3, kernel_size=5, stride=1, padding=2, bias=True)
	def forward(self, x):
		x0 = self.conv0(x)
		x0 = self.bn0(x0)
		x0 = F.relu(x0)
		x1 = self.conv1(x0)
		x1 = self.bn1(x1)
		x1 = F.relu(x1)
		x2 = self.conv2(x1)

		return x2

class Spatial_Extraction_MFF(nn.Module):
	def __init__(self, num_features=512, block_channel=[64,64,128,256,512]):

		super(Spatial_Extraction_MFF, self).__init__()
		self.E = E_resnet()#64,128,256,512
		self.D = D(512)
		self.MFF = MFF(block_channel)
		self.R = R()


	def forward(self, x):
		x_block0,x_block1, x_block2, x_block3, x_block4 = self.E(x)
		x_decoder = self.D(x_block0,x_block1, x_block2, x_block3, x_block4)
		x_mff = self.MFF(x_block0,x_block1, x_block2, x_block3, x_block4,[x_decoder.size(2),x_decoder.size(3)])
		out = self.R(torch.cat((x_decoder, x_mff), 1))
		return out

if __name__ == "__main__":
	# net=E_resnet()
	net=Spatial_Extraction_MFF()
	x=torch.zeros([1,3,256,256])
	out=net(x)
	print(out.size())
	for i in out:
		print(i.size())
