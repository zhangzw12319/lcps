# Step-4 Training, Testing and Reproducing

## Quick Start

Use NuScenes for example.

### Download Checkpoints for Image Branch

|            Name             |                            Link                            |
| :-------------------------: | :--------------------------------------------------------: |
| `resnet18-5c106cde.pth` (√) | https://download.pytorch.org/models/resnet18-5c106cde.pth  |
|   `resnet34-333f7ec4.pth`   | https://download.pytorch.org/models/resnet34-333f7ec4.pth  |
|   `resnet50-19c8e357.pth`   | https://download.pytorch.org/models/resnet50-19c8e357.pth  |
|  `resnet101-5d3b4d8f.pth`   | https://download.pytorch.org/models/resnet101-5d3b4d8f.pth |
|  `resnet152-b121ed2d`.pth   | https://download.pytorch.org/models/resnet152-b121ed2d.pth |

Download these weights and link their direct parent folder to `../checkpoints`.

### Train+Val

```bash
# Single GPU
python train.py \
-c configs/nusc_train.yaml \
-l test_nusc_20230101_pix_4k2bs_val_a6000.log

## DDP
export CUDA_VISIBLE_DEVICES=4,5,6,7

nohup \
python -m torch.distributed.launch --nproc_per_node=4 train.py \
-c configs/nusc_train.yaml \
-l nusc_20230101_pix_4k2bs_a6000.log \
> nohup_nusc_20230101_pix_4k2bs_a6000.log 2>&1 &
```



### Val Only

```bash
# Single GPU
python val.py \
-c configs/nusc_val.yaml \
-l test_nusc_20230101_pix_4k2bs_val_a6000.log

## DDP
export CUDA_VISIBLE_DEVICES=4,5,6,7

nohup \
python -m torch.distributed.launch --nproc_per_node=4 val.py \
-c configs/nusc_val.yaml \
-l nusc_20230101_pix_4k2bs_val_a6000.log \
> nohup_nusc_20230101_pix_4k2bs_val_a6000.log 2>&1 &
```



### Test

```bash
# Single GPU
python test.py \
-c configs/nusc_test.yaml \
-l test_nusc_20230101_pix_4k2bs_test_a6000.log \
--resume

## DDP
export CUDA_VISIBLE_DEVICES=4,5,6,7

nohup \
python -m torch.distributed.launch --nproc_per_node=4 test.py \
-c configs/nusc_test.yaml \
-l test_nusc_20230101_pix_4k2bs_test_a6000.log \
> nohup_nusc_20230101_pix_4k2bs_test_a6000.log 2>&1 &
```



Create Submission File 

- Step-1 Run `test.py`

- Step-2 Use `utils/nusc_validate_submission.py`or `kitti_validate_submission.py` to check submission format.

### Reproducing

#### Checkpoint Files

|    Dataset    |                 Checkpoint Link                  | YAML Settings for Validation |               Logs                |
| :-----------: | :----------------------------------------------: | :--------------------------: | :-------------------------------: |
|   NuScenes    | [Hugging Face](https://huggingface.co/im99/lcps) |   pa_po_nuscenes_val.yaml    |  [nusc](../logs/nohup_nusc_val.log)  |
| SemanticKITTI | [Hugging Face](https://huggingface.co/im99/lcps) |     pa_po_kitti_val.yaml     | [kitti](../logs/nohup_kitti_val.log) |



#### Leaderboard

|                           Dataset                            | PQ (test) | mIoU (test) |
| :----------------------------------------------------------: | :-------: | :---------: |
| [NuScenes](https://eval.ai/web/challenges/challenge-page/1243/leaderboard/3127) (Team: Auto-Perception) |   79.8%   |    78.9%    |
| [SemanticKITTI](https://codalab.lisn.upsaclay.fr/competitions/7092#results) (Team: AutoPerception) |   58.8%   |    62.8%    |



## Pre-checklist for Custom Adjustment (Advanced)

### Check Config Files

:white_check_mark: Check whether `path` and `instance_path)` (See [here](https://github.com/zhangzw12319/lcps/blob/main/docs/prepare_nusc.md)) variables are correctly set in `configs/XXXXX.yaml`, where `path` is set to `../data/nuscenes` or `../data/SemanticKitti` as default (via soft link) and instance_path is set to the location of `pkl` files. If NuScenes, check whether `version` is correct as shown in the following table.

|     Config Names      |  Correct Split  |
| :-------------------: | :-------------: |
| `XXX_nusc_mini.yaml`  |   `v1.0-mini`   |
| `XXX_nusc_train.yaml` | `v1.0-trainval` |
|  `XXX_nusc_val.yaml`  | `v1.0-trainval` |
| `XXX_nusc_test.yaml`  |   `v1.0-test`   |

:white_check_mark: Check whether `pix-fusion` is set on. If true `pix_fusion_path` is set to `resnet18-5c106cde.pth`.

:white_check_mark: Check whether data augmentations is set on. This may introduce performance fluctuations.

:white_check_mark: Check `min_points: 15` for NuScenes and `min_points: 50` for SemanticKITTI.

:white_check_mark: Check `model_save_path` carefully, since the checkpoint file in the next training period will overwrite the last one if filename unchanged. Therefore, when last training is over, it is better to `mv` the checkpoint files and rename. We recommend that the name of checkpoint and log file format be `[nusc|kitti]_20230101_modules_data-augmentation_8k4bs_3090.log`, where  `modules` means the proposed modules used in your code like `pix` representing `ACPA fusion`, and `data-augmentation` represents the augmentation adopted such as `flip` representing `RandomFlip`.

:white_check_mark: Check whether `model_load_path` is correct when run `val.py` and `test.py`.

:white_check_mark: Check important hyper-parameters, such as `learning_rate`, `LR_MILESTONES`, `max_epoch`, `XXX_batch_size`. Especially, the `num_workers` is better to be set to 4-8, since lower value may cause CPU preprocessing data as the bottleneck, while too high value will paralyze the server and slow down all GPU programs (severely slow down other users).

### Check Bash Scripts

:white_check_mark: Important args: `-c` specifies which config to use, `-l` set path to log files and `--resume` set whether loading pretrained weight from `model_load_path` in configs (must be on when running `val.py` or `test.py`). Check whether the path will overwrite previous log files.

:white_check_mark: Verify the correctness of LiDAR Semantic Segmentations

```bash
python utils/nusc_validate_submission.py --result_path semantic_test_preds/ --eval_set test --dataroot path_to_test_set --version v1.0-test --verbose True --zip_out ./ 
```













