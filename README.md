
<p align="center">
  <h1 align="center">RE0: Recognize Everything with 3D Zero-shot Instance Segmentation</h1>
  <h3 align="center"><a href="https://recognizeeverything.github.io/">Paper</a> | <a href="https://recognizeeverything.github.io/">Project Page</a></h3>
  <div align="center"></div>

</p>
<p align="center">
<p align="center">
  <a href="">
    <img src="https://recognizeeverything.github.io/src/ssr_teaser.png" alt="Logo" width="80%">
  </a>
</p>
<br>

## Installation

```shell
# clone repo
git clone https://github.com/RecognizeEverything/RE0.git

# add submodule
git submodule add https://github.com/facebookresearch/detectron2.git submodule/detectron2
git submodule add https://github.com/qqlu/Entity.git submodule/Entity
git submodule add https://github.com/openai/CLIP.git submodule/CLIP
git submodule add https://github.com/ScanNet/ScanNet.git submodule/ScanNet

# prepare environment
conda create -n re0 python=3.8.18
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -e submodule/detectron2
pip install -e submodule/CLIP

# Installation of Cropformer (https://github.com/qqlu/Entity/blob/main/Entityv2/CODE.md)
cp cropformer.py submodule/detectron2/projects/CropFormer/demo_cropformer/
cp -r submodule/Entity/Entityv2/CropFormer submodule/detectron2/projects
make -C submodule/detectron2/projects/CropFormer/entity_api/PythonAPI
cd submodule/detectron2/projects/CropFormer/mask2former/modeling/pixel_decoder/ops/
sh make.sh 

# Prepare the needed checkpoints
1.https://huggingface.co/datasets/qqlu1992/Adobe_EntitySeg/tree/main/CropFormer_model/Entity_Segmentation/Mask2Former_hornet_3x
2.https://github.com/qqlu/Entity/blob/main/Entityv2/README.md#model-zoo
```

## Data Preparation

### ScanNet200

1. Download the [ScanNet200 Dataset](https://kaldir.vc.in.tum.de/scannet_benchmark/documentation)
2. Run preprocessing code for raw ScanNet as follows:

```shell
cd data_preprocess/
python preprocess_2d_scannet.py --scannet_path="PATH/TO/YOUR/SCANS" --output_path="PATH/TO/OUTPUT" --frame_skip=10
```

### Your Personal Dataset

1. following the format of ScanNet
2. update the `setting.py`

## Getting Started

- updating configs in `setting.py`

- ```shell
  python main.py
  ```

## Citation

If you find *RE0* useful to your research, please cite our work

## Acknowledgement

RE0 is inspirited by the following repos: [SAM3D](https://github.com/Pointcept/SegmentAnything3D), [SAMPro3D](https://github.com/GAP-LAB-CUHK-SZ/SAMPro3D), [OpenMask3D](https://github.com/OpenMask3D/openmask3d), [Cropformer](https://github.com/qqlu/Entity/blob/main/Entityv2/README.md#model-zoo)



