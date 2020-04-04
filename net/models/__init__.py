import sys
sys.path.append('.')
from models.UNet import UNet,UNet64,UNet_Depth
from models.Gen_Y_Unet import GE_Y_Unet,GE_Y_Unet64
from models.EUUnet import EUUNet
from models.SwiftNet import SwiftNet
from models.Swiftslim import SwiftNetSlim
from models.FullConv_SwiftNet import FullConv_SwiftNet
from models.Spatial_feature_extraction_network import Spatial_Extraction_MFF as MFF
from models.HDRNETS.hdrnet1 import HDRNet as hdr1


