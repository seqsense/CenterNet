from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
import os

import torch.utils.data as data

class BBOX_DATA(data.Dataset):
  num_classes = 80
  default_resolution = [512, 512]
  mean = np.array([0.40789654, 0.44719302, 0.47026115],
                   dtype=np.float32).reshape(1, 1, 3)
  std  = np.array([0.28863828, 0.27408164, 0.27809835],
                   dtype=np.float32).reshape(1, 1, 3)

  def __init__(self, opt, classes_path, data_label_path, image_dir_path):
    # super(COCO, self).__init__()

    self.max_objs = 128
    self.image_dir = image_dir_path
    self.class_name = []
    with open(classes_path, 'r') as f:
      class_name = f.readline()
      while class_name:
        self.class_name.append(class_name.strip())
        class_name = f.readline()
  
    self._valid_ids = list(range(len(class_name)))
    self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}

    self.num_classes = len(self.class_name)
    self.voc_color = [(v // 32 * 64 + 64, (v // 8) % 4 * 64, v % 8 * 32) \
                      for v in range(1, self.num_classes + 1)]
    self._data_rng = np.random.RandomState(123)
    self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                             dtype=np.float32)
    self._eig_vec = np.array([
        [-0.58752847, -0.69563484, 0.41340352],
        [-0.5832747, 0.00994535, -0.81221408],
        [-0.56089297, 0.71832671, 0.41158938]
    ], dtype=np.float32)
    # self.mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
    # self.std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)

    self.opt = opt

    self.image_datas = []
    '''
    [
      {
        file_name: str,
        objects: [
          bbox: [sx, sy, ex, ey],
          class_id: int
        ]
      }
    ]
    '''
    with open(data_label_path, 'r') as f:
      data_str = f.readline()
      while data_str:
        data_list = data_str.strip().split()
        data_dict = {'file_name': data_list[0], 'objects':[]}
        for data in data_list[1:]:
          data = list(map(int, data.split(',')))
          object_dict = {'bbox': np.array(data[:4]), 'class_id': data[4]}
          data_dict['objects'].append(object_dict)
        self.image_datas.append(data_dict)
        data_str = f.readline()

    # self.image_dir = os.path.join() os.path.dirname(self.image_datas[0]['file_name'])
    self.num_samples = len(self.image_datas)

  def _to_float(self, x):
    return float("{:.2f}".format(x))

  def convert_eval_format(self, all_bboxes):
    # TODO
    '''
    # import pdb; pdb.set_trace()
    detections = []
    for image_id in all_bboxes:
      for cls_ind in all_bboxes[image_id]:
        category_id = self._valid_ids[cls_ind - 1]
        for bbox in all_bboxes[image_id][cls_ind]:
          bbox[2] -= bbox[0]
          bbox[3] -= bbox[1]
          score = bbox[4]
          bbox_out  = list(map(self._to_float, bbox[0:4]))

          detection = {
              "image_id": int(image_id),
              "category_id": int(category_id),
              "bbox": bbox_out,
              "score": float("{:.2f}".format(score))
          }
          if len(bbox) > 5:
              extreme_points = list(map(self._to_float, bbox[5:13]))
              detection["extreme_points"] = extreme_points
          detections.append(detection)
    return detections
    '''

  def __len__(self):
    return self.num_samples

  def save_results(self, results, save_dir):
    # TODO
    '''
    json.dump(self.convert_eval_format(results), 
                open('{}/results.json'.format(save_dir), 'w'))
    '''
  
  def run_eval(self, results, save_dir):
    # TODO
    pass
    '''
    # result_json = os.path.join(save_dir, "results.json")
    # detections  = self.convert_eval_format(results)
    # json.dump(detections, open(result_json, "w"))
    self.save_results(results, save_dir)
    coco_dets = self.coco.loadRes('{}/results.json'.format(save_dir))
    coco_eval = COCOeval(self.coco, coco_dets, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    '''
