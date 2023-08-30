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
├── v1.0-mini
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

You can download either from NuScenes website or [OpenDatalab](https://opendatalab.com/nuScenes).

PKL files for NuScenes for our project can be downloaded [here](https://pan.baidu.com/s/1a94BcZAYb0rWMJZL_uZayw?pwd=posk).

(Optional) Support for CopyPaste (or CutMix) Augmentation. (Will be Updated.)

### Extraction

Please note that first extract `lidarseg` GT compressed files, move such as `attribute.json` to `v1.0-mini`, `v1.0-trainval` and `v1.0-test`, and **confirm overwrite original files if conflict**; then extract `panoptic` GT compressed files, and do the same things. The meta file overwrite priority order: `panoptic` > `lidarseg` > `V1.0 Original Meta Files`.