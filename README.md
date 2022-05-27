# [CVPR 2022] ZeroWaste: Towards Deformable Object Segmentation in Cluttered Scenes
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5246013.svg)](https://doi.org/10.5281/zenodo.5246013) <img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/80x15.png" />
![Image](images/recycling_figure_1_v3.png)
This is the official repository of the ZeroWaste project [arxiv](https://arxiv.org/abs/2106.02740). Our ZeroWaste dataset distributed under 
<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"></a><a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">Creative Commons Attribution-NonCommercial 4.0 International License </a>  can be found [here](https://doi.org/10.5281/zenodo.4899926).

## Supervised experiments
### Requirements
* Python 3.8
* pytorch 1.8
* [Detectron2](https://github.com/facebookresearch/detectron2) (please see the [official installation guide](https://detectron2.readthedocs.io/en/latest/tutorials/install.html))

### Training
To train the supervised methods (DeeplabV3+ or Mask R-CNN), use the command below: 
```
# train deeplab on ZeroWaste data
python deeplab/train_net.py --config-file deeplab/configs/zerowaste_config.yaml --dataroot /path/to/zerowaste/data/ (optional) --resume OUTPUT_DIR /deeplab/outputs/*experiment_name* (optional) MODEL.WEIGHTS /path/to/checkpoint.pth

# train Mask R-CNN on ZeroWaste\TACO-zerowaste data
python maskrcnn/train_net.py --config-file maskrcnn/configs/*config*.yaml (optional, only use if trained on TACO-zerowaste) --taco --dataroot /path/to/zerowaste/data/ (optional) --resume OUTPUT_DIR /maskrcnn/outputs/*experiment_name* (optional) --MODEL.WEIGHTS /path/to/checkpoint.pth

# train ReCo on ZeroWasteAug data
python reco_aug/train_sup.py --dataset zerowaste --num_labels 0 --seed 1
```

### Evaluation
The checkpoints for the experiments reported in our paper can be found [here](http://csr.bu.edu/ftp/recycle/models/). Please use the following code to evaluate the model on our dataset:
```
# evaluate the pretrained deeplab ZeroWaste:
python deeplab/train_net.py --config-file deeplab/configs/zerowaste_config.yaml --dataroot /path/to/zerowaste-or-taco/data/  --eval-only OUTPUT_DIR /deeplab/outputs/results/ --MODEL.WEIGHTS path/to/checkpoint.pth

# evaluate the pretrained Mask R-CNN on ZeroWaste\TACO-zerowaste:
python deeplab/train_net.py --config-file deeplab/configs/*config*.yaml (optional, only use if evaluated on TACO-zerowaste) --taco --dataroot /path/to/zerowaste-or-taco/data/  --eval-only OUTPUT_DIR /maskrcnn/outputs/*ex

# evaluate the pretrained ReCo-sup on ZeroWasteAug
python reco_aug/test_sup.py --dataset zerowaste --num_labels 0 --seed 1 --checkpoint path/to/checkpoint.pth
```

## Semi-supervised experiments
We used the [official implementation](https://github.com/lorenmt/reco) of [ReCo](https://arxiv.org/abs/2104.04465)  with minor modification in data loading for our experiments. 

### Requirements
* Python 3.8
* pytorch 1.8

### Data
Please download and unzip the ZeroWaste-f, ZeroWasteAug, and ZeroWaste-s (in reco_org/dataset and reco_aug/dataset) for the semi-zupervised experiments. 

### Training
To train the model from scratch with the hyperparameters used in our experiments:

```
python reco_aug/train_semisup.py --dataset zerowaste --num_labels 60 --apply_aug classmix --apply_reco
```

### Evaluation
The trained model checkpoints can be found [here](http://csr.bu.edu/ftp/recycle/models/reco/reco_aug/). The following command runs inference on the given data: 

```
python reco_aug/test_sup.py --dataset zerowaste --num_labels 0 --apply_aug classmix --apply_reco --checkpoint path/to/checkpoint.pth
```

## Weakly-supervised experiments
We used the [official implementation](https://github.com/OFRIN/PuzzleCAM) of [Puzzle-Cam](https://arxiv.org/abs/2101.11253)
### Requirements
* Python 3.8, PyTorch 1.7.0, and more in requirements.txt
* CUDA 10.1, cuDNN 7.6.5

Please download the ZeroWaste-w dataset for binary classification. A pretrained binary classifier used in our experiments can be found [here](http://csr.bu.edu/ftp/recycle/models/binary_classification/).

### For Puzzle-Cam trained with 4-class image-level labels

```
cd puzzlecam_4_classes
bash run.sh
```
### For Puzzle-Cam trained with binary before/after image-level labels

```
cd puzzlecam_binary
bash run.sh
```

## Citation
Please cite our paper: 
```
@article{zerowaste,
  author =       {Dina Bashkirova, Mohamed Abdelfattah, Ziliang Zhu, James Akl,    Fadi Alladkani, Ping Hu, Vitaly Ablavsky, Berk Calli, Sarah Adel Bargal and Kate Saenko},
  title =        {ZeroWaste Dataset: Towards Deformable Object Segmentation in Cluttered Scenes},
  howpublished = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year =         {2022}
}
```

