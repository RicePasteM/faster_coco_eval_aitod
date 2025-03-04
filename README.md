<div align="center">
    <h1>Faster-COCO-Eval-AITOD</h1>
</div>

<div align="center">

[![PyPI](https://img.shields.io/pypi/v/faster-coco-eval-aitod)](https://pypi.org/project/faster-coco-eval-aitod)
[![PyPI Downloads](https://img.shields.io/pypi/dm/faster-coco-eval-aitod.svg?label=PyPI%20downloads)](https://pypi.org/project/faster-coco-eval-aitod/)

<!-- [![Conda Version](https://img.shields.io/conda/vn/conda-forge/faster-coco-eval.svg)](https://anaconda.org/conda-forge/faster-coco-eval)
[![Conda Platforms](https://img.shields.io/conda/pn/conda-forge/faster-coco-eval.svg)](https://anaconda.org/conda-forge/faster-coco-eval) -->

[![license](https://img.shields.io/github/license/RicePasteM/faster_coco_eval_aitod.svg)](https://github.com/MiXaiLL76/faster_coco_eval_aitod/blob/main/LICENSE)

</div>


## Disclaimer

This project is a fork of [Faster-COCO-Eval](https://github.com/MiXaiLL76/faster_coco_eval) modified specifically for the AITOD (ATiny Object Detection in Aerial Images) dataset.

The main modifications include adapting evaluation parameters for tiny object detection and adding LRP (Localization Recall Precision) metric calculation, maintaining compatibility with [cocoapi-aitod](https://github.com/jwwangchn/cocoapi-aitod) while significantly improving computation speed.

## Key Features

- Optimized evaluation parameters for tiny object detection scenarios
- Added LRP metric calculation consistent with aitodpycocotools
- Significantly faster computation compared to original pycocotools
- Maintains all original Faster-COCO-Eval functionality
- Compatible with AITOD dataset evaluation requirements

## Install

### Basic implementation identical to pycocotools

```bash
pip install faster-coco-eval-aitod
```

### Conda install

```bash
conda install conda-forge::faster-coco-eval-aitod
```

### Basic usage

```py
import faster_coco_eval_aitod

# Replace aitodpycocotools with faster_coco_eval_aitod
faster_coco_eval_aitod.init_as_aitodpycocotools()

from faster_coco_eval_aitod import COCO, COCOeval_faster

anno = COCO(str(anno_json))
pred = anno.loadRes(str(pred_json))

val = COCOeval_faster(anno, pred, "bbox")
val.evaluate()
val.accumulate()
val.summarize()
# Access LRP metrics
lrp_metrics = val.stats_lrp
```

## Performance Comparison

For AITOD dataset evaluation, our implementation shows significant speed improvements while maintaining identical results with aitodpycocotools (tested using `/test` in this project):

| Image Counts | faster-coco-eval-aitod | aitodpycocotools | Speed Improvement |
| ---: | ---: | ---: | ---: |
| 5000 | 31.7s | 57.7s | **+45%** |


## Feautures

This library provides not only validation functions, but also error visualization functions. Including visualization of errors in the image.
You can study in more detail in the [test](https://github.com/RicePasteM/faster_coco_eval_aitod/blob/main/test).


## Update history

Available via link [history.md](https://github.com/RicePasteM/faster_coco_eval_aitod/blob/main/history.md)

<!-- ## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=MiXaiLL76/faster_coco_eval_aitod&type=Date)](https://star-history.com/#MiXaiLL76/faster_coco_eval_aitod&Date) -->

## License

The original module was licensed with apache 2, so I will continue with the same license.
Distributed under the apache version 2.0 license, see [license](LICENSE) for more information.

## Citation

If you use this fork in your research, please cite both this project and original Faster-COCO-Eval:

```
@article{faster-coco-eval-aitod,
  title   = {{Faster-COCO-Eval-AITOD}: Faster interpretation of the original aitodpycocotools},
  author  = {ZhangchiHu},
  year    = {2025}
}

@article{faster-coco-eval,
  title   = {{Faster-COCO-Eval}: Faster interpretation of the original COCOEval},
  author  = {MiXaiLL76},
  year    = {2024}
}
```