cd ..

nohup python val.py \
-c configs/pa_po_nuscenes_val.yaml \
-l nusc_val.log \
--resume \
> nohup_nusc_val.log 2>&1 &