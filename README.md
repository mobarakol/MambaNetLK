# PointNetLK_c3vd: Point Cloud Registration with Mamba3D

This project is an extended version of the original PointNetLK, with added support for Mamba3D architecture and C3VD dataset processing capabilities.

## Model Architecture

### Mamba3D Model Series

The project includes multiple versions of Mamba3D models optimized for 3D point cloud processing:

- **mamba3d_v1**: Pure Python implementation of Mamba architecture, slower but highly compatible
- **mamba3d_v2**: Implementation using official mamba-ssm library, fast but requires additional dependencies
- **mamba3d_v3**: Enhanced version of v2 with SE attention mechanism
- **mamba3d_v4**: Latest version with further optimized architecture

### Features

- **Point Cloud Permutation Invariance**: Achieved through positional encoding and spatially-aware SSM
- **Linear Complexity**: Mamba architecture's long sequence processing capability
- **Multi-level Features**: Captures both local and global geometric features
- **Compatibility**: Fully compatible with PointNet_features interface

## Dependencies

### Core Dependencies
```
PyTorch >= 1.8.0
torchvision
numpy
scipy
matplotlib
plyfile
```

### Optional Dependencies (Restricted)
**Note**: Due to hardware limitations, the following libraries are currently not supported but will be added in the future:
- `mamba-ssm`: Required for mamba3d_v2/v3/v4, but currently unavailable
- `open3d`: For voxelization optimization, currently using Python implementation (slower)

### Current Status
- **Recommended**: `mamba3d` (i.e., mamba3d_v1) - Pure Python implementation, stable and available
- **Available but requires mamba-ssm**: `mamba3d_v3/v4` - Requires mamba-ssm library, fully debugged
- **Experimental**: `mamba3d_v2` - Requires mamba-ssm, but not fully debugged

## C3VD Dataset Training Guide

### Dataset Preparation

The C3VD dataset should have the following structure:
```
C3VD_datasets/
├── C3VD_ply_source/          # Source point clouds (video data)
│   ├── class1_scene1/
│   │   ├── 0001_depth_pcd.ply
│   │   └── ...
│   └── class2_scene2/
└── visible_point_cloud_ply_depth/  # Target point clouds (MRI data)
    ├── class1_scene1/
    │   ├── points.ply
    │   └── ...
    └── class2_scene2/
```

### 1. Train Classifier (Stage 1)

First train a classifier for feature extraction, recommend using mamba3d_v1:

```bash
python experiments/train_classifier.py \
  -o ./results/mamba3d_classifier \
  -i /path/to/C3VD_datasets \
  -c ./experiments/sampledata/c3vd_classes.txt \
  --dataset-type c3vd \
  --model-type mamba3d \
  --num-points 1024 \
  --batch-size 16 \
  --epochs 200 \
  --device cuda:0 \
  --num-mamba-blocks 3 \
  --d-state 16 \
  --expand 2 \
  --symfn selective
```

#### Mamba3D Specific Parameters:
- `--num-mamba-blocks`: Number of Mamba blocks (default: 3)
- `--d-state`: Mamba state space dimension (default: 16)  
- `--expand`: Mamba expansion factor (default: 2)
- `--symfn`: Symmetry function (recommended: selective)

### 2. Train PointNet-LK Registration (Stage 2)

Use pre-trained classifier weights to initialize PointNet-LK:

```bash
python experiments/train_pointlk.py \
  -o ./results/mamba3d_pointlk \
  -i /path/to/C3VD_datasets \
  -c ./experiments/sampledata/c3vd_classes.txt \
  --dataset-type c3vd \
  --model-type mamba3d \
  --pretrained ./results/mamba3d_classifier_feat_best.pth \
  --num-points 1024 \
  --batch-size 8 \
  --epochs 200 \
  --device cuda:0 \
  --use-voxelization \
  --voxel-size 0.05
```

#### Voxelization Parameters (Python Implementation):
- `--use-voxelization`: Enable voxelization preprocessing
- `--voxel-size`: Voxel size (default: 0.05)
- `--voxel-grid-size`: Voxel grid size (default: 32)
- `--max-voxel-points`: Maximum points per voxel (default: 100)

**Note**: Current voxelization uses Python implementation. If open3d is available, you can replace it for better efficiency.

### 3. Test Registration Performance

```bash
python experiments/test_pointlk.py \
  -o ./results/test_results.csv \
  -i /path/to/C3VD_datasets \
  -c ./experiments/sampledata/c3vd_classes.txt \
  --dataset-type c3vd \
  --model-type mamba3d \
  --pretrained ./results/mamba3d_pointlk_model_best.pth \
  --num-points 1024 \
  --device cuda:0
```

## Performance Optimization Recommendations

### Current Limitations
1. **mamba-ssm library**: Not available, limiting the use of v2/v3/v4 versions
2. **open3d library**: Not available, voxelization uses Python implementation (slower)
3. **Memory usage**: Mamba models require more GPU memory compared to PointNet

### Optimization Strategies
- Use smaller batch_size (8-16)
- Appropriately reduce num_mamba_blocks (2-3)
- Enable gradient clipping to prevent gradient explosion
- Use mixed precision training (if supported)

## Main Files Description

### Training Scripts
- `experiments/train_classifier.py`: Classifier training
- `experiments/train_pointlk.py`: PointNet-LK registration training  
- `experiments/test_pointlk.py`: Registration performance testing

### Model Implementations
- `ptlk/mamba3d_v1.py`: Main Mamba3D implementation (recommended)
- `ptlk/mamba3d_v2.py`: mamba-ssm based implementation (requires additional dependencies)
- `ptlk/mamba3d_v3.py`: Enhanced version (requires additional dependencies)
- `ptlk/mamba3d_v4.py`: Latest version (requires additional dependencies)

### Data Processing
- `ptlk/data/datasets.py`: C3VD dataset class and voxelization functionality


## Citation

Most of the code in this project is based on the following original paper:

```bibtex
@InProceedings{yaoki2019pointnetlk,
       author = {Aoki, Yasuhiro and Goforth, Hunter and Arun Srivatsan, Rangaprasad and Lucey, Simon},
       title = {PointNetLK: Robust & Efficient Point Cloud Registration Using PointNet},
       booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
       month = {June},
       year = {2019}
}
```

## Future Development Plans

- [ ] Integrate open3d library to optimize voxelization performance
- [ ] Complete debugging work for mamba3d_v2
