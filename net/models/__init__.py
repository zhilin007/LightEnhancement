import sys
sys.path.append('.')
from models.UNet import UNet,UNet64,UNet_Depth
from models.Gen_Y_Unet import GE_Y_Unet,GE_Y_Unet64
from models.EUUnet import EUUNet
from models.SwiftNet import SwiftNet
from models.Swiftslim import SwiftNetSlim
from models.FullConv_SwiftNet import FullConv_SwiftNet
from models.Spatial_feature_extraction_network import Spatial_Extraction_MFF as MFF
from models.HDRNETS.hdr1 import HDRPointwiseNN as hdr1
from models.Res18Net_1 import Res18Net1

from models.GuidedFilter.guided_filter import FastGuidedFilter, ConvGuidedFilter
from models.GuidedFilter.GuidedFilterNetwork import DeepGuidedFilter,DeepGuidedFilterAndMap,DeepGuidedFilterLayer,DeepGuidedFilterLayerAndMap
from models.SwiftNet_guidedfilter import SwiftNet_GuidedFilter
from models.SwiftNetSlim_GuidedFilterLayerAndMap import SwiftNetSlim_GuidedFilterLayerAndMap
from models.SwiftNetSlim_GFLAndMap_BN import SwiftNetSlim_GFLAndMap_BN
from models.SwiftNetSlim_GFLAndMap_BN2 import SwiftNetSlim_GFLAndMap_BN2
from models.SwiftNetSlim_GFL_SN import SwiftNetSlim_GFL_SN
from models.SwiftNetSlim2_GFLAndMap_BN2 import SwiftNetSlim2_GFLAndMap_BN2


from models.GENY.Gen_Y_Swiftslim2_BN2 import Gen_Y_Swiftslim2_BN2
from models.GENY.Gen_Y_Swiftslim2_BN2_Share import Gen_Y_Swiftslim2_BN2_Share
from models.GENY.Gen_Y_Swiftslim2_BN2_SAME import Gen_Y_Swiftslim2_BN2_SAME
from models.GENY.Gen_Y_Swiftslim2_BN2_SAME import Gen_Y_Swiftslim2_BN2_SAME_DownSample

from models.GENY.Gen_Y_Swiftslim_BN2 import Gen_Y_Swiftslim_BN2
from models.GENY.Gen_Y_Swiftslim_BN2_Share import Gen_Y_Swiftslim_BN2_Share

from models.Defined_Network.Backbone7x7 import Backbone7x7
from models.Defined_Network.Gen_Y_Backbone7x7 import Gen_Y_Backbone7x7
from models.Defined_Network.Gen_Y_Backbone7x7_Share import Gen_Y_Backbone7x7_Share



