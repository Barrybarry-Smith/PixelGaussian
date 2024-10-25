# PixelGaussian: Generalizable 3D Gaussian Reconstruction From Arbitrary Views

### [Paper](https://arxiv.org/abs/2410.18979) | [Prject Page](https://wzzheng.net/PixelGaussian)

> [Xin Fei](https://scholar.google.com/citations?hl=zh-CN&user=r9rsD_0AAAAJ)*, [Wenzhao Zheng](https://wzzheng.net/)$\dagger$, [Yueqi Duan](https://duanyueqi.github.io/), [Wei Zhan](https://zhanwei.site/), [Masayoshi Tomizuka](https://me.berkeley.edu/people/masayoshi-tomizuka/), [Kurt Keutzer](https://people.eecs.berkeley.edu/~keutzer/), [Jiwen Lu](http://ivg.au.tsinghua.edu.cn/Jiwen_Lu/)

Tsinghua University, UC Berkeley

\*Work doen during an internship at UC Berkeley, $\dagger$ Project leader

Most existing generalizable 3D Gaussian splatting methods (e.g., pixelSplat, MVSplat) assign a fixed number of Gaussians to each pixel, leading to inefficiency in capturing local geometry and overlap across views.  Differently, **our PixelGaussian dynamically adjusts the Gaussian distributions based on geometric complexity in a feed-forward framework.**  With comparable efficiency, PixelGaussian (trained using 2 views) successfully generalizes to various numbers of input views with adaptive Gaussian densities.

![teaser](./figs/teaser.png)

## News
- **[2024/10/25]** Code release.
- **[2024/10/25]** Paper released on [arXiv](https://arxiv.org/abs/2410.18979).

## Visualizations
![pipeline](./figs/visualizations.png)

## Overview

![pipeline](./figs/pipeline.png)

Given multi-view input images, we initialize 3D Gaussians using a lightweight image encoder and cost volume. Cascade Gaussian Adapter (CGA) then dynamically adapts both the distribution and quantity of Gaussians. By leveraging local image features, Iterative Gaussian Refiner (IGR) further refines Gaussian representations via deformable attention. Finally, novel views are rendered from the refined 3D Gaussians using rasterization-based rendering.

## Results

![pipeline](./figs/results.png)

PixelGaussian achieves the best performance on the two representative datasets. Trained with 2 reference views, PixelGaussian can generalize to more views.

## Getting Started

### Installation

1. Please clone this project, create a conda virtual environment and install the requirements in `requirement.txt`

2. Download RealEstate10K, ACID datasets and corresponding assets following the instructions of [pixelSplat](https://github.com/dcharatan/pixelsplat/tree/main)

3. Running the code by
```bash
python -m src.main +experiment=[re10k/acid] data_loader.train.batch_size=[batch_size]
```

## Related Projects

Our code is based [MVSplat](https://github.com/donydchen/mvsplat) and [GaussianFormer](https://github.com/huang-yh/GaussianFormer) and is also inspired by [pixelSplat](https://github.com/dcharatan/pixelsplat) and  [SelfOcc](https://github.com/huang-yh/SelfOcc).

## Citation

If you find this project helpful, please consider citing the following paper:
```
@article{fei2024pixel,
    title={PixelGaussian: Generalizable 3D Gaussian Reconstruction From Arbitrary Views},
    author={Fei, Xin and Zheng, Wenzhao and Duan, Yueqi and Zhan, Wei and Tomizuka, Masayoshi and Keutzer, Kurt and Lu, Jiwen},
    journal={arXiv preprint arXiv:2410.18979},
    year={2024}
}
```
