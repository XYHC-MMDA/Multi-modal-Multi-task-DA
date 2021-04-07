from mmdet.datasets.pipelines import Compose
from .dbsampler import DataBaseSampler
from .formating import Collect3D, DefaultFormatBundle, DefaultFormatBundle3D, SegDetFormatBundle
from .loading import (LoadAnnotations3D, LoadMultiViewImageFromFiles,
                      LoadPointsFromFile, LoadPointsFromMultiSweeps,
                      NormalizePointsColor, PointSegClassMapping,
                      LoadSegDetPointsFromFile, LoadFrontImage, LoadImgSegLabel, LoadImgSegLabelVer2,
                      LoadMaskedMultiSweeps, LoadPointsFromFileVer2)
from .test_time_aug import MultiScaleFlipAug3D
from .transforms_3d import (BackgroundPointsFilter, GlobalRotScaleTrans,
                            IndoorPointSample, ObjectNoise, ObjectRangeFilter,
                            ObjectSample, PointShuffle, PointsRangeFilter,
                            RandomFlip3D, VoxelBasedPointSampler,
                            SegDetPointsRangeFilter, PointsSensorFilter, MergeCat, DetLabelFilter,
                            FrontImageFilter, PointsRangeFilterVer2, PointsSensorFilterVer2,
                            GetSegFromPoints, XmudaAug3D)

# __all__ = [
#     'ObjectSample', 'RandomFlip3D', 'ObjectNoise', 'GlobalRotScaleTrans',
#     'PointShuffle', 'ObjectRangeFilter', 'PointsRangeFilter', 'Collect3D',
#     'Compose', 'LoadMultiViewImageFromFiles', 'LoadPointsFromFile',
#     'DefaultFormatBundle', 'DefaultFormatBundle3D', 'DataBaseSampler',
#     'NormalizePointsColor', 'LoadAnnotations3D', 'IndoorPointSample',
#     'PointSegClassMapping', 'MultiScaleFlipAug3D', 'LoadPointsFromMultiSweeps',
#     'BackgroundPointsFilter', 'VoxelBasedPointSampler',
#     # added
#     'LoadSegDetPointsFromFile', 'LoadFrontImage', 'SegDetPointsRangeFilter',
#     'SegDetFormatBundle', 'PointsSensorFilter', 'MergeCat', 'DetLabelFilter',
#     'FrontImageFilter', 'LoadImgSegLabel', 'LoadMaskedMultiSweeps'
# ]
