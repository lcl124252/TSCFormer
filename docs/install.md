# Step-by-step installation instructions

**a. Create a conda virtual environment and activate**

python 3.8 may not be supported.

```shell
conda create -n environment python=3.7 -y
conda activate environment
```

**b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/get-started/previous-versions/)**

```shell
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
```

or 

```shell
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

We select this pytorch version because mmdet3d 0.17.1 do not supports pytorch >= 1.11 and our cuda version is 11.3.

**c. Install mmcv, mmdet, and mmseg**

```shell
pip install openmim
mim install mmcv-full==1.4.0
mim install mmdet==2.14.0
mim install mmsegmentation==0.14.1
```

**c. Install mmdet3d 0.17.1, DFA3D and dcnv3**

Compared with the offical version, the mmdetection3d provided by [OccFormer](https://github.com/zhangyp15/OccFormer) further includes operations like bev-pooling, voxel pooling. 

**Note that** :If the module is not found after executing the `setup.sh`, it is recommended to switch to the installation folder and manually run `python setup.py install`.

```shell
cd packages
bash setup.sh
cd ../
```

**d. Install other dependencies, like timm, einops, torchmetrics, spconv, pytorch-lightning, etc.**

```shell
pip install -r docs/requirements.txt
```

**e. Fix bugs (known now)**

```shell
pip install yapf==0.40.0
```

##### f. pretrain weights

Please create a `./pretrain` folder under the `./TSCFormer` folder and download the **Efficientnetb7** weights and **geodepth** weights we provided, then place them inside the `./pretrain` folder.

```
cd ./TSCFormer
mkdir pretrain
```

