# CoADNet-CoSOD
CoADNet: Collaborative Aggregation-and-Distribution Networks for Co-Salient Object Detection (NeurIPS 2020)

# Datasets
We employ [COCO-SEG](https://drive.google.com/file/d/1OuZ7BMQzakuNjXLpkdncgFM0ZFrqVJuq/view?usp=sharing) as our training dataset, which covers 78 different object categories containing totally 200k labeled images. There is also an auxiliary dataset [DUTS](https://drive.google.com/file/d/1p9DBTfPoFQwPtsz1kDl-mDjtZ8R-GrjD/view?usp=sharing), which is a popular benchmark dataset (the training split) for (single-image) salient object detection.

We employ four datasets for performance evaluation, as listed below:
1) [Cosal2015](https://drive.google.com/file/d/18f3NTNsnXokPX3R3MmUeLZxV81w_cs0P/view?usp=sharing): 50 categories, 2015 images.
2) [iCoseg](https://drive.google.com/file/d/1WhKqdfCWfDhZGn-PJSmawrTMroAWlwlt/view?usp=sharing): 38 categories, 643 images.
3) [MSRC](https://drive.google.com/file/d/1stOjNblaZTQhSU_pneSUzKgYeAWj2Gvy/view?usp=sharing): 7 categories, 210 images.
4) [CoSOD3k](https://drive.google.com/file/d/1FmeV7gtJ-rpHm4BFy_cS-lbxGm0CkCeb/view?usp=sharing): 160 categories, 3316 images.

Put all the above datasets as well as the corresponding [info files](https://drive.google.com/file/d/1-kiX_B0fzAODIIS4SkJNprAuI9eu-4Hp/view?usp=sharing) under `../data` folder.

# Training
1) Download [backbone networks](https://drive.google.com/file/d/1wxZI41ADcmwBt4H6yDT7Q4V-G8WbreTJ/view?usp=sharing) and put them under `./ckpt/pretrained`
2) Run `Pretrain.py` to pretrain the whole network, which helps to learn saliency cues and speeds up convergence.
3) Run `Train-COCO-SEG-S1.py` to train the whole network on the COCO-SEG dataset. Note that, since COCO-SEG is modified from a generic semantic segmentation dataset (MS-COCO) and thus may ignore the crucial saliency patterns, we need a post-refinement procedure as conducted in `Train-COCO-SEG-S2.py`. When using other more appropriate training datasets such as CoSOD3k, we skip this procedure.

# Testing
We organize the testing codes in a Jupyter notebook `test.ipynb`, which performs testing on all the four evaluation datasets.
Note that there is an `is_shuffle` option during testing, which enables us to perform multiple trials to output more robust predictions.
