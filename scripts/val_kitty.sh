cd ..

nohup python val.py \
-c configs/pa_po_kitti_val.yaml \
-l kitti_val.log \
--resume \
> nohup_kitti_val.log 2>&1 &


