# Title

## Description

This repository contains the code for our paper: **Temporal and Spatial Context Aware Voxel Transformer for Semantic Scene Completion**

## Environment setup

This project is built upon the following environment:
* Python 3.7
* CUDA 11.3
* PyTorch 1.10.0

Please refer to the [Inastall.md](docs/Install.md) in the docs folder for details.

## Dataset

* [SemanticKITTI](https://semantic-kitti.org/)
* [SSCBench-kitti-360](https://github.com/ai4ce/SSCBench)

####   Prepare Dataset

​	Please refer to the [Dataset.md](docs/Dataset.md) in the docs folder for details.

## Train on single GPU

Train a model on the SemanticKITTI dataset by
```
CUDA_VISIBLE_DEVICES=0 python main.py \
--config_path configs/semantickitti_TSCFormer.py \
--log_folder semantickitti_TSCFormer \
--seed 7240 \
--log_every_n_steps 100
```

Train a model on the SSCBench-kitti-360 dataset by
```
CUDA_VISIBLE_DEVICES=0 python main.py \
--config_path configs/kitti360_TSCFormer.py \
--log_folder kitti360_TSCFormer \
--seed 7240 \
--log_every_n_steps 100
```

## Train on multiple GPUs

Train a model on the SemanticKITTI dataset by

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
--config_path configs/semantickitti_TSCFormer.py \
--log_folder semantickitti_TSCFormer \
--seed 7240 \
--log_every_n_steps 100
```

Train a model on the SSCBench-kitti-360 dataset by

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
--config_path configs/kitti360_TSCFormer.py \
--log_folder kitti360_TSCFormer \
--seed 7240 \
--log_every_n_steps 100
```



## Evaluate

Evaluate a model on the SemanticKITTI dataset by
```
CUDA_VISIBLE_DEVICES=0 python main.py \
--eval --ckpt_path .ckpts/best.ckpt \
--config_path configs/semantickitti_TSCFormer.py \
--log_folder semantickitti_TSCFormer_eval --seed 7240 \
--log_every_n_steps 100
```

Evaluate a model on the SSCBench-kitti-360 dataset by
```
CUDA_VISIBLE_DEVICES=0 python main.py \
--eval --ckpt_path .ckpts/best.ckpt \
--config_path configs/kitti360_TSCFormer.py \
--log_folder kitti360_TSCFormer_eval --seed 7240 \
--log_every_n_steps 100
```

## Evaluation with Saving the Results

The results will be saved into the save_path.

Evaluate a model on the SemanticKITTI dataset by

```
CUDA_VISIBLE_DEVICES=0 python main.py \
--eval --ckpt_path .ckpts/best.ckpt \
--config_path configs/semantickitti_TSCFormer.py \
--log_folder semantickitti_TSCFormer_eval --seed 7240 \
--log_every_n_steps 100 --save_path pred
```

Evaluate a model on the SSCBench-kitti-360 dataset by

```
CUDA_VISIBLE_DEVICES=0 python main.py \
--eval --ckpt_path .ckpts/best.ckpt \
--config_path configs/kitti360_TSCFormer.py \
--log_folder kitti360_TSCFormer_eval --seed 7240 \
--log_every_n_steps 100 --save_path pred
```

## Submission

SemanticKITTI dataset should change the configuration of the test data in the config file

```
CUDA_VISIBLE_DEVICES=0 python main.py \
--eval --ckpt_path .ckpts/best.ckpt \
--config_path configs/semantickitti_TSCFormer.py \
--log_folder semantickitti_TSCFormer_eval --seed 7240 \
--log_every_n_steps 100 --save_path pred --test_mapping
```

## Visualization

We used the official visual code, see the [semantic-kitti-api](https://github.com/PRBonn/semantic-kitti-api) for details.

## Acknowledgment

Our implementation is mainly based on the following codebase. We gratefully thank the authors for their wonderful works.

- [BEVFormer](https://github.com/fundamentalvision/BEVFormer)
- [mmdet3d](https://github.com/open-mmlab/mmdetection3d)
- [MonoScene](https://github.com/astra-vision/MonoScene)
- [semantic-kitti-api](https://github.com/PRBonn/semantic-kitti-api)
- [MobileStereoNet](https://github.com/cogsys-tuebingen/mobilestereonet)
- [Symphonize](https://github.com/hustvl/Symphonies.git)
- [DFA3D](https://github.com/IDEA-Research/3D-deformable-attention.git)
- [VoxFormer](https://github.com/NVlabs/VoxFormer.git)
- [OccFormer](https://github.com/zhangyp15/OccFormer.git)
- [CGFormer][pkqbajng/CGFormer (github.com)](https://github.com/pkqbajng/CGFormer)
- [SGN](https://github.com/Jieqianyu/SGN)
