import sys

import faster_coco_eval_aitod.core.coco as coco
import faster_coco_eval_aitod.core.faster_eval_api as cocoeval
import faster_coco_eval_aitod.core.mask as mask
from faster_coco_eval_aitod.core import COCO, COCOeval_faster
from faster_coco_eval_aitod.version import __author__, __version__


def init_as_pycocotools():
    import faster_coco_eval_aitod

    sys.modules["pycocotools"] = faster_coco_eval_aitod
    sys.modules["pycocotools.coco"] = coco
    sys.modules["pycocotools.cocoeval"] = cocoeval
    sys.modules["pycocotools.mask"] = mask


__all__ = [
    "init_as_pycocotools",
    "mask",
    "coco",
    "cocoeval",
    "COCO",
    "COCOeval_faster",
    "__author__",
    "__version__",
]
