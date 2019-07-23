from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .sample.ddd import DddDataset
from .sample.exdet import EXDetDataset
from .sample.ctdet import CTDetDataset
from .sample.multi_pose import MultiPoseDataset
from .sample.bbox_sample import BBoxTaskDataset

from .dataset.coco import COCO
from .dataset.pascal import PascalVOC
from .dataset.kitti import KITTI
from .dataset.coco_hp import COCOHP
from .dataset.bbox_dataset import BBOX_DATA


dataset_factory = {
  'bbox_dataset': BBOX_DATA,
  'coco': COCO,
  'pascal': PascalVOC,
  'kitti': KITTI,
  'coco_hp': COCOHP
}

_sample_factory = {
  'exdet': EXDetDataset,
  'ctdet': CTDetDataset,
  'bbox_sample': BBoxTaskDataset,
  'ddd': DddDataset,
  'multi_pose': MultiPoseDataset
}


def get_dataset(dataset, task):
  class Dataset(dataset_factory[dataset], _sample_factory[task]):
    pass
  return Dataset
  
