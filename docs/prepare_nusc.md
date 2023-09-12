# Step-1 Preparing NuScenes

## Dataset Introduction

See Official [Website](https://www.nuscenes.org/panoptic)

## Dataset Structure

```
nuscenes/
├── lidarseg					# Semantic GT, Format: XXXX_lidarseg.bin
│   ├── v1.0-mini
│   ├── v1.0-trainval
│   └── v1.0-test				# Empty folder, brought by extracting compressed files
├── maps						# Map Image Folders
├── panoptic					# Panoptic GT, Format: XXXX_panoptic.bin
│   ├── v1.0-mini
│   ├── v1.0-trainval
│   └── v1.0-test
├── pkl_files					# Generated index files, can be downloaded (see the following)
├── inst_database_train			# Filtered Foreground objects for instance augmentation (e.g. Copy Paste)
│   ├── inst_database_train_info.pkl
│   ├── barrier
│   	├── XXX.bin
│   	└── ...
│   ├── barrier
│   └── ...
├── samples						# Key Frames
│   ├── CAM_BACK
│   ├── CAM_BACK_LEFT
│   ├── CAM_BACK_RIGHT
│   ├── CAM_FRONT
│   ├── CAM_FRONT_LEFT
│   ├── CAM_FRONT_RIGHT
│   ├── LIDAR_TOP				# Format: XXX_LIDAR_TOP_XXX.pcd.bin		
│   ├── RADAR_BACK_LEFT 		# Format: XXX_RADAR_BACK_LEFT_XXX.pcd
│   ├── RADAR_BACK_RIGHT
│   ├── RADAR_FRONT
│   ├── RADAR_FRONT_LEFT
│   └── RADAR_FRONT_RIGHT
├── sweeps						# Non key frames, structure similar as samples folder
├── v1.0-mini`					# Metadata folder with `json` files
│   ├── attribute.json
│   ├── ego_pose.json
│   ├── map.json
│   ├── sample_data.json
│   ├── sensor.json
│   ├── calibrated_sensor.json
│   ├── instance.json
│   ├── lidarseg.json
│   ├── panoptic.json
│   ├── sample.json
│   ├── visibility.json
│   ├── category.json
│   ├── log.json
│   ├── sample_annotation.json
│   └── scene.json
├── v1.0-trainval				# Similar as v1.0-mini
└── v1.0-test					# Similar as v1.0-mini
```

## Access

### Download 

You can download the original dataset either from NuScenes [website](https://www.nuscenes.org/nuscenes) or [OpenDataLab](https://opendatalab.com/nuScenes).

For NuScenes website, you need to download `Full dataset (v1.0)`(including mini, trainval and test splits), `NuScenes-lidareseg` ，and`NuScenes-panoptic`. For OpenDataLab, it seems that only `Full dataset (v1.0)`is provided and you need to download the other two from NuScenes website.

PKL files for NuScenes for our project can be downloaded [here](https://pan.baidu.com/s/1a94BcZAYb0rWMJZL_uZayw?pwd=posk).

(Optional) Support for Instance Augmentation. The compressed files for `inst_database_train` can be downloaded [here](https://pan.baidu.com/s/1h-IyvHWz3oD1P6KACBq9cg?pwd=ccm2).

### Extraction

Please note that first extract `lidarseg` GT compressed files, move` json` files such as `attribute.json` to `v1.0-mini`, `v1.0-trainval` and `v1.0-test`, and **confirm overwrite original files from `v1.0-Fulll metadata`if conflict**; then extract `panoptic` GT compressed files, and do the same things. The meta file overwrite priority order: `panoptic` > `lidarseg` > `V1.0 Original Meta Files`.