from .base import Base3DDetector
# from .centerpoint import CenterPoint
# from .dynamic_voxelnet import DynamicVoxelNet
# from .h3dnet import H3DNet
# from .mvx_faster_rcnn import DynamicMVXFasterRCNN, MVXFasterRCNN
# from .mvx_two_stage import MVXTwoStageDetector
# from .parta2 import PartA2
# from .ssd3dnet import SSD3DNet
# from .votenet import VoteNet
# from .voxelnet import VoxelNet
from .mmda import MMDA
from .multi_sepvoxelization import MultiSensorMultiTaskSep
from .multi_univoxelization import MultiSensorMultiTaskUni
from .sep_seg_det import SepSegDet
from .fusion_disc import FusionDisc
from .fusion_disc_01 import FusionDisc01
from .fusion_disc_02 import FusionDisc02
from .single_seg_xyz import SingleSegXYZ
from .sep_disc import SepDisc
from .target_consistency import TargetConsistency
from .target_consistency_no_bg import TargetConsistencyNoBG

__all__ = [
    'Base3DDetector', 'MMDA', 'MultiSensorMultiTaskSep', 'MultiSensorMultiTaskUni',
    'SepSegDet', 'FusionDisc', 'FusionDisc01', 'FusionDisc02', 'SingleSegXYZ',
    'SepDisc', 'TargetConsistency', 'TargetConsistencyNoBG'
]

# __all__ = [
#     'Base3DDetector', 'VoxelNet', 'DynamicVoxelNet', 'MVXTwoStageDetector',
#     'DynamicMVXFasterRCNN', 'MVXFasterRCNN', 'PartA2', 'VoteNet', 'H3DNet',
#     'CenterPoint', 'SSD3DNet', 'MMDA', 'MultiSensorMultiTaskSep', 'MultiSensorMultiTaskUni',
#     'SepSegDet', 'FusionDisc', 'FusionDisc01', 'FusionDisc02'
# ]
