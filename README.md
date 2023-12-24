# LCPS: First LiDAR-Camera Panoptic Segmentation Framework for 3D Perception

## Description

<div align="center">
   <img src="https://s2.loli.net/2023/08/24/1bZL8DSlWctEFNG.png"  height=160>          <img src="https://s2.loli.net/2023/08/24/yRazBCwPvVZqKO9.png" height=160>          <img src="https://s2.loli.net/2023/08/24/GJUPziVsu9SIj2X.png" height=160>
</div>

> Title: LiDAR-Camera Panoptic Segmentation via Geometry-Consistent and Semantic-Aware Alignment (ICCV 2023) 
>
> Download: ICCV Formal Version is [here](https://openaccess.thecvf.com/content/ICCV2023/html/Zhang_LiDAR-Camera_Panoptic_Segmentation_via_Geometry-Consistent_and_Semantic-Aware_Alignment_ICCV_2023_paper.html)
> 
> Download: Arxiv preprint paper is [here](https://arxiv.org/abs/2308.01686)
>
> Authors: [Zhiwei Zhang](https://scholar.google.com/citations?hl=en&user=GjkRn78AAAAJ), [Zhizhong Zhang](https://scholar.google.com/citations?user=CXZciFAAAAAJ&hl=en), Qian Yu, [Ran Yi](https://scholar.google.com/citations?user=y68DLo4AAAAJ&hl=en), [Yuan Xie\*](https://scholar.google.com/citations?user=RN1QMPgAAAAJ&hl=en), [Lizhuang Ma\*](https://scholar.google.com/citations?user=yd58y_0AAAAJ&hl=en)
>
> $\dagger$ Equal Contribution   *Corresponding author

## Demo

![](https://github.com/zhangzw12319/SharingDemo/blob/main/video_view_compressed.gif)
<div align="center">
    The Visualization video is Captured on NuScenes Dataset, compressed and converted to gif for efficient playting. Left: Semantic Segmentation,  Right: Instance Segmentation. Original MP4 video can be downloaded at <a href="https://www.dropbox.com/scl/fi/lujxp6ejliny9sceg64ao/video_view.mp4?rlkey=t9fypl4uptuyzmhgroni9f3re&dl=0">Dropbox</a> or <a href="https://www.aliyundrive.com/s/JGx2X4hpmEt">Aliyun</a> (Code: q5h7).
</div>

## News

- **[2023/08/24]** Demo release.
- **[2023/08/04]** Arxiv preprint released. :paperclip:
- **[2023/07/15]**  Accepted to ICCV 2023!  :fire::fire::fire:

## Introduction

![](https://s2.loli.net/2023/08/24/3ljWIEHhkpSF17c.png)

3D panoptic segmentation is a challenging perception task that requires both semantic segmentation and instance segmentation. In this task, we notice that images could provide rich texture, color, and discriminative information, which can complement LiDAR data for evident performance improvement, but their fusion remains a challenging problem. To this end, we propose LCPS, the first LiDAR-Camera Panoptic Segmentation network. In our approach, we conduct LiDAR-Camera fusion in three stages: 1) an Asynchronous Compensation Pixel Alignment (ACPA) module that calibrates the coordinate misalignment caused by asynchronous problems between sensors; 2) a Semantic-Aware Region Alignment (SARA) module that extends the oneto-one point-pixel mapping to one-to-many semantic relations; 3) a Point-to-Voxel feature Propagation (PVP) module that integrates both geometric and semantic fusion information for the entire point cloud. Our fusion strategy improves about 6.9% PQ performance over the LiDAR-only baseline on NuScenes dataset. Extensive quantitative and qualitative experiments further demonstrate the effectiveness of our novel framework.

## Getting Started

Code Structure (Full Projects will be updated soon):

```
Current:
LCPS/
├── docs                    # Detailed Documentations
│   ├── prepare_nusc.md		# Prepare NuScenes Dataset
│   ├── prepare_kitti.md	# Prepare SemanticKITTI Dataset
│   ├── env_install.md		# Prepare Environment
│   ├── train_test_repro.md	# Start training, validation, testing and reproducing
│   └── other_toolkits.md	# Get some visualization pictures & statistic results
├── requirements.txt		# Package Install
└── README.md 				# Quick Start
```

- [Step-1 Preparing NuScenes](docs/prepare_nusc.md)
- [Step-2 Preparing SemanticKITTI](docs/prepare_kitti.md)
- [Step-3 Environment Installation](docs/env_install.md)
- [Step-4 Training, Testing and Reproducing](docs/train_test_repro.md)
- [Step-5 Other Toolkits (Will be Updated Soon)](docs/other_toolkits.md)
- [Step-6 (Optional) Further Issues? Find QA](docs/QA.md)

## More Visualizations

Full visualizations and quick explainations can be seen [here](docs/more_vis.md).

![](https://s2.loli.net/2023/08/19/TV39IbcEpK1oA2n.png)

<img src="https://s2.loli.net/2023/08/19/LHNdsikFKDUtEy7.png" style="zoom: 25%;" />

## License

All assets and code are under the [Apache 2.0 license](./LICENSE) unless specified otherwise.

## Citation

If you find this project helpful, please consider **citing** the following paper:

```bitex
@InProceedings{Zhang_2023_ICCV,
    author    = {Zhang, Zhiwei and Zhang, Zhizhong and Yu, Qian and Yi, Ran and Xie, Yuan and Ma, Lizhuang},
    title     = {LiDAR-Camera Panoptic Segmentation via Geometry-Consistent and Semantic-Aware Alignment},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {3662-3671}
}
```

## Acknowledgement [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

Many thanks to the following awesome open-source projects, which provide helpful guidance for us！

Closely Relevant Project:

- [Panoptic-PolarNet](https://github.com/edwardzhou130/Panoptic-PolarNet)
- [CAM](https://github.com/zhoubolei/CAM)

Other Tools:

- Visualization Tools Del:
  - [NuScenes-devkit](https://github.com/nutonomy/nuscenes-devkit)

  - [SemanticKiTTI Toolkit](https://github.com/PRBonn/semantic-kitti-api)

- QR-Code Generation: https://www.hlcode.cn
