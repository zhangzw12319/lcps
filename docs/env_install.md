# Step-3 Environment Installation



## Env Requirements

- Ubuntu 20.04, 22.04 or later (recommended)
- Python: it is suggested that the python version is compatible with torch version. It can be checked referred by `--index-url` in [PyTorch Previous versions](https://download.pytorch.org/whl/torch/). For example, if you install torch v1.12.1 with CUDA 11.3 by wheels, you can at most install python<=3.9.X.
- PyTorch >= 1.11 (1.11.X, 1.12.X, 1.13.X, 2.0.X, 2.1.X tested, but not compulsory)
- CUDA Tookit (tested on cu-113, cu-117 and cu-120, but not compulsory)
- Spconv (Require >= 2.2.3, compulsory. Please follow [official readme](https://github.com/traveller59/spconv).)
- Pytorch_scatter (Please follow [official readme](https://github.com/rusty1s/pytorch_scatter).)

Our code is robust at multiple version combination of `PyTorch`, `spconv 2.X` , `CUDA` . The higher version is, the fast running speed and more utilization of latest NVIDIA GPU can be achieved. For now, there is an example envrionment setting as follows: 

> - CUDA 11.7
> - Python==3.10.0
> - PyTorch==1.13.1
> - spconv-cu117==2.3.6



For other packages, you can install by `pip install -r requirements.txt`