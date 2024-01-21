export CUDA_VISIBLE_DEVICES=4,5,6,7

cd ..
nohup python -m torch.distributed.launch --nproc_per_node=4 --use_env train.py \
-c configs/pa_po_kitti_trainval.yaml \
-l kitti_20230730_4k2bs_a6000.log \
> nohup_kitti_20230730_4k2bs_a6000.log 2>&1 &
