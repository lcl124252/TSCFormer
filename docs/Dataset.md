# Dataset

## SemanticKITTI Dataset

### Prepare data

Download the [KITTI Odometry Dataset](https://www.cvlibs.net/datasets/kitti/eval_odometry.php) (including color, velodyne laser data, and calibration files) and the annotations for Semantic Scene Completion from [SemanticKITTI](http://www.semantic-kitti.org/dataset.html#download). Please follow the command image2depth of [VoxFormer](https://github.com/NVlabs/VoxFormer) to create depth maps and preprocess the annotations for semantic scene completion.

### Folder structure

```
/semantickittii/
          |-- sequences/
          │       |-- 00/
          │       │   |-- poses.txt
          │       │   |-- calib.txt
          │       │   |-- image_2/
          │       │   |-- image_3/
          │       |   |-- voxels/
          │       |         |- 000000.bin
          │       |         |- 000000.label
          │       |         |- 000000.occluded
          │       |         |- 000000.invalid
          │       |         |- 000005.bin
          │       |         |- 000005.label
          │       |         |- 000005.occluded
          │       |         |- 000005.invalid
          │       |-- 01/
          │       |-- 02/
          │       .
          │       |-- 21/
          |-- labels/
          │       |-- 00/
          │       │   |-- 000000_1_1.npy
          │       │   |-- 000000_1_2.npy
          │       │   |-- 000005_1_1.npy
          │       │   |-- 000005_1_2.npy
          │       |-- 01/
          │       .
          │       |-- 10/
          |-- lidarseg/
          |       |-- 00/
          |       │   |-- labels/
          |       |         ├ 000001.label
          |       |         ├ 000002.label
          |       |-- 01/
          |       |-- 02/
          |       .
          |       |-- 21/
          |-- depth/sequences/
          		  |-- 00/
          		  │   |-- 000000.npy
          		  |   |-- 000001.npy
          		  |-- 01/
                  |-- 02/
                  .
                  |-- 21/
```

## SSCBench-kitti-360 Dataset

### Prepare data

Refer to [SSCBench](https://github.com/ai4ce/SSCBench/tree/main/dataset/KITTI-360) to download the dataset. And download the [poses](https://drive.google.com/file/d/1nsZLa-X3fz14ZZxZgPUOCm3dY5MDZ5vZ/view?usp=drive_link) that [SGN](https://github.com/Jieqianyu/SGN/tree/master/preprocess) has processed acoording the matching index between SSCBench-KITTI-360 and KITTI-360. 

The data is organized in the following format:

```
./kitti360/
        └── data_2d_raw/
        │        ├── 2013_05_28_drive_0000_sync/ 
        │        │   ├── image_00/
        │        │   │   ├── data_rect/
        │        │   │   │	├── 000000.png
        │        │   │   │	├── 000001.png
        │        │   │   │	├── ...
        │        │   ├── image_01/
        │        │   │   ├── data_rect/
        │        │   │   │	├── 000000.png
        │        │   │   │	├── 000001.png
        │        │   │   │	├── ...
        │        │   ├── voxels/
        │        │   └── poses.txt
        │        ├── 2013_05_28_drive_0002_sync/
        │        ├── 2013_05_28_drive_0003_sync/
        │        .
        │        └── 2013_05_28_drive_0010_sync/
        └── preprocess/
                 ├── labels/ 
                 │   ├── 2013_05_28_drive_0000_sync/
                 │   │   ├── 000000_1_1.npy
                 │   │   ├── 000000_1_2.npy
                 │   │   ├── 000000_1_8.npy
                 │   │   ├── ...
                 │   ├── 2013_05_28_drive_0002_sync
                 │   ├── 2013_05_28_drive_0003_sync/
                 │   .
                 │   └── 2013_05_28_drive_0010_sync/
                 ├── labels_half/ 
                 └── unified/ 
```

Then please follow the command image2depth of [VoxFormer](https://github.com/NVlabs/VoxFormer) to create depth maps

The data is organized in the following format:

```
./kitti360/
        └── data_2d_raw/
        │        ├── 2013_05_28_drive_0000_sync/ 
        │        │   ├── image_00/
        │        │   │   ├── data_rect/
        │        │   │   │	├── 000000.png
        │        │   │   │	├── 000001.png
        │        │   │   │	├── ...
        │        │   ├── image_01/
        │        │   │   ├── data_rect/
        │        │   │   │	├── 000000.png
        │        │   │   │	├── 000001.png
        │        │   │   │	├── ...
        │        │   ├── voxels/
        │        │   └── poses.txt
        │        ├── 2013_05_28_drive_0002_sync/
        │        ├── 2013_05_28_drive_0003_sync/
        │        .
        │        └── 2013_05_28_drive_0010_sync/
        └── preprocess/
        │        ├── labels/ 
        │        │   ├── 2013_05_28_drive_0000_sync/
        │        │   │   ├── 000000_1_1.npy
        │        │   │   ├── 000000_1_2.npy
        │        │   │   ├── 000000_1_8.npy
        │        │   │   ├── ...
        │        │   ├── 2013_05_28_drive_0002_sync
        │        │   ├── 2013_05_28_drive_0003_sync/
        │        │   .
        │        │   └── 2013_05_28_drive_0010_sync/
        │        ├── labels_half/ 
        │        └── unified/ 
        |-- depth
                    |-- sequences
                        |-- 2013_05_28_drive_0000_sync
                        |	|-- 000000.npy
                        |	|-- 000001.npy
                        |-- ...
                        |-- 2013_05_28_drive_0010_sync
```

