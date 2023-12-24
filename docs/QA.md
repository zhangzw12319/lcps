# Questions & Answers (To be continued...)

This is the common QA list. Further analysis and statements for issues will be updated here...

## Ⅰ. YAML Config QA
### Q1. What about the computation resources needed?

| No.  |    Dataset    |   Split    | Num_Workers | Batch_Size | GPU Memory(GB) |     Recommended Devices     |
| :--: | :-----------: | :--------: | :---------: | :--------: | :------------: | :-------------------------: |
|  1   |   NuScenes    |   train    |      1      |     2      |    20 - 22     |  3090(x4, x8)/4090(x4, x8)  |
|  2   |   NuScenes    |   train    |      4      |     2      |    30 - 38     | V100,  A6000, A40 (x4, x8)  |
|  3   |   NuScenes    |  val/test  |      4      |     2      |    10 - 12     |  3090(x4, x8)/4090(x4, x8)  |
|  4   | SemanticKITTI |   train    |      1      |     2      |    38 - 40     |     A6000, A40 (x4, x8)     |
|  5   | SemanticKITTI | val / test |      1      |     4      |       20       | 3090(x1, x4), 4090 (x1, x4) |
|      |               |            |             |            |                |                             |

Notes:

- However, when `num_worker` is set larger, the program will consume more GPUs and cause heavier workload to CPU. For example on NuScenes, when `batch_size=2`,if `num_workers` is set =2 the `CUDA Out of Memory` will be reported on 24G GPU devices; else if `batch_size=1` , memory usage may be keep  just below 24G. Also you can lower down the batch_size to 1 and set gradient accumulation to 2 (`grad_accumu`=2) to keep larger `dataloader` worker numbers.
- When using Pytorch DDP for multi GPU training, there is usually at least one device (not always cuda:0, but also others) eating much more GPU than others from our observations, (e.g. bs=2, num_workers=4, 30G, 38G, 30G, ...). And I have tuned `torch.load(XXX.pt, map_location="cpu")` but find no obvious effect. So maybe A6000 or A40 is safer than V100(32G) to avoid OOM.



### Q2 Where should I pay attention to in config files?

- For NuScenes, `min_points=15`; For SemanticKITTI, `min_points=50`.
- `inst_aug` (copypaste or contmix) is default open for SemanticKITTI to improve rare categories such as terrain, other_ground and many foreground objects; but for NuScenes we find no obvious improvement, leaving it for future exploration. Further, instance augmentation is used in training set only.
- `max_epoch` is set to 300 for convenient watching. Actually, NuScenes is stopped before 120 Epochs and SemanticKITTI is stopped before 30 Epochs in practice (Maybe you can leave it to occupy shared GPUs or stop it when you just remember). Since`LR_MILESTONES=[100, 150, 200]`(we're lazy to change it frequently), the LR changes at 100 Epoch may benefits a small performance jump for NuScenes. It is recommended to set the first milestones earlier (100 → 40-70 ) since after 40 Epochs the performance will converged to 77 PQ around. Our codebase is not detailedly finetuned (e.g. no LR warm-up or exponential LR decays), leaving room for fine-grained LR adjustments. If you have better LR tuning suggestions, please feel free to leave comments in issues.
- to be continued ...


## Ⅱ. Performance QA

- We find that the performance fluctuated across different computing architectures with different number of  GPU devices. For example, if checkpoint file is given and fixed, the validation/test results may fluctuate at around $\pm 0.2\% \text{PQ}$ . 

## Ⅲ. Project Structure QA

- 

## Ⅳ. Detailed Implementation QA

- To be honest, the interface implementation of our codebase is ugly and untidy. I've delayed the release day for a long time attempting to elegantly refactor my code into some structure (like mmsegmentation, etc.). Accompanied with illness and other heavy workloads in recent months, I fail to do so at last, since the implementation is rather complicated with lots of  subtle logical implementations. And my past attempts diminish the final performance a lot during re-testing periods, and thus I give up temporarily and release the untrimmed code. Sorry for the delay.

- Copy-paste augmentation may be contradictory to multi-modal fusion, since the panoptic labeling of extra instances should be carefully handled, and the image area should also be added with fake 2D instance features. There are already research work (e.g. PointAugmenting) tackling with this issue, but for code simplicity (it is already so complicated), I adopt zero vectors for such image features.

- Noise points are supervised for extra noise class. Although it make no affect for semantic segmentation, we still find that it may interfere with post-processing module for instance segmentation and panoptic merging. The implementation is also kind of tricky.

- `nb_process_label`, `nb_process_inst ` is carefully adjusted to generate panoptic labels with `numba` acceleration. It is added the filtering function to separate the empty voxel area and noise voxel area.

- to be continued ...

  

## Ⅴ. Others