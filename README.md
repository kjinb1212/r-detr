### Codes of Our AAAI2023 Submission “Rotated-DETR: an End-to-End Transformer-based Oriented Object Detector for Aerial Images”

This is the Pytorch implementation of our paper "Rotated-DETR: an End-to-End Transformer-based Oriented Object Detector for Aerial Images". Our codes are based on mmrotate framework.

### Step-1 Installation 
1. Create a conda virtual environment and activate it.

    ```shell
    conda create -n r-detr python=3.7 -y
    conda activate r-detr
    ```
2. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/)

3. Install MMRotate
    ```shell
    pip install openmim
    mim install mmrotate
    cd r_detr
    pip install -r requirements/build.txt
    pip install -v -e .
    ```
### Step-2 Data Preparation 
Prepare the dataset following the [Preparing DOTA Dataset](https://github.com/open-mmlab/mmrotate/blob/main/tools/data/dota/README.md)


### Step-3 Change the Codes of MMdetection2 to Add Our Method
#### Test a model

You can use the following commands to infer a dataset.

```shell
# single-gpu
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]

# multi-gpu
./tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} [optional arguments]
```
Examples:

```shell
# single-gpu
python ./tools/test.py  \
  configs/sparse_detr/dotav1_1024_512/sparse_detr_swint_10_dota1_1024_512.py \
  checkpoints/rotated_detr_swint_10_dota1_1024_512.pth --format-only \
  --eval-options submission_dir=work_dirs/Task1_results

# multi-gpu
./tools/dist_test.sh  \
  configs/sparse_detr/dotav1_1024_512/sparse_detr_swint_10_dota1_1024_512.py \
  checkpoints/rotated_detr_swint_10_dota1_1024_512.pth 1 --format-only \
  --eval-options submission_dir=work_dirs/Task1_results
```

#### Train a model
```shell
# single-gpu
python tools/train.py ${CONFIG_FILE} [optional arguments]

# multi-gpu
./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```
Examples:

```shell
# single-gpu
python ./tools/test.py  \
  configs/sparse_detr/dotav1_1024_512/sparse_detr_swint_10_dota1_1024_512.py

# multi-gpu
./tools/dist_test.sh  \
  configs/sparse_detr/dotav1_1024_512/sparse_detr_swint_10_dota1_1024_512.py 4
```
