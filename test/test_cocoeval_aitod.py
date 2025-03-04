import unittest
import numpy as np
import torch
from unittest.mock import patch
# from coco_eval_aitod_slow import COCO
# from coco_eval_aitod_slow import AitodCocoEvaluatorSlow as AitodCocoEvaluator

from faster_coco_eval_aitod import COCO, COCOeval_faster
from coco_eval_aitod import AitodCocoEvaluator
from gt_data import gt_data as GT_DATA, predictions as PRED

class TestAitodCocoEvaluator(unittest.TestCase):
    def setUp(self):
        # 创建一个简单的标注数据
        self.gt_data = GT_DATA
        
        # 创建COCO对象
        self.coco_gt = COCO()
        self.coco_gt.dataset = self.gt_data
        self.coco_gt.createIndex()
        
        # 创建评估器
        self.evaluator = AitodCocoEvaluator(
            coco_gt=self.coco_gt,
            iou_types=['bbox']
        )
        self.evaluator.cleanup()
    

    def test_summarize(self):
        predictions = PRED
        self.evaluator.update(predictions)

        self.evaluator.synchronize_between_processes()
        self.evaluator.accumulate()
        
        # 执行汇总
        self.evaluator.summarize()
        
        # 验证评估指标
        for iou_type in self.evaluator.iou_types:
            coco_eval = self.evaluator.coco_eval[iou_type]
            self.assertTrue(hasattr(coco_eval, 'stats'))
            self.assertIsInstance(coco_eval.stats, np.ndarray)

if __name__ == '__main__':
    unittest.main()