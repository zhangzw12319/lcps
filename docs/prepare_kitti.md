# Preparing SemanticKITTI

## Dataset Introduction

See Official [Website](http://www.semantic-kitti.org/tasks.html#panseg)



## Dataset Structure

```
SemanticKitti/
├── dataset
│   └── sequences					# 21 sequences
│   		├── 00					# 00~07 + 09~10: training split; 08: val split; Other：test split. For test split, there is no labels folder.
│   		│	├── image_2			# Left Camrea
│   		│	├── image_3			# Right Camera
│   		│	├── instance		# Download as shown in the following. Format: XXX.bin
│   		│	├── labels			# Format: XXX.label
│   		│	├── velodyne		# Format: XXX.bin
│   		│	├── calib.txt
│   		│	├── poses.txt
│   		│	└── times.txt
│   		├── 01
│   		├── 02
│   		├── ...
│   		└── 21
└── instance_path.pkl				# Download pkl file as shown in the following.
```



## Access

### Download 

You can download either from SemanticKITTI website or [OpenDatalab](https://opendatalab.com/SemanticKITTI/download) (Lack camera images `data_odometry_color.zip`. Please download it from SemanticKITTI website.)

PKL files for SemanticKITTI for our project can be downloaded [here](https://pan.baidu.com/s/1qt-Xlh5IDFLrs1yha0TwfA?pwd=6uoi).

(Optional) Support for CopyPaste (or CutMix) Augmentation. Download files [here](https://pan.baidu.com/s/1ogJB0d4sB7syGjr6MqwxEQ?pwd=0op5 ) and extract each compressed file such as `00/instance.tar` to under sequence `00` folder.

### Extraction

When extracting SemanticKITTI, please extract `calib.zip` at last,  since we need to overwrite `calib.txt` in `calib.zip` with that in `data_odometry_velodyne.zip`.

