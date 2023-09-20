## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Setting up dataset

The data/ folder should look like this:
```
data
├── refcoco
|   ├── anns
|   |   ├── refcoco
|   |   ├── refcoco+
|   |   ├── refcocog
│   ├── images
|   │   ├── train2014  # images from train 2014
│   ├── masks
|   |   ├── refcoco
|   |   ├── ...

```

1. Download [MSCOCO](http://mscoco.org/dataset/#overview) and link the ```train2014``` folder to ```data/refcoco/images/train2014```.

2. Download [refcoco/masks](https://drive.google.com/file/d/1oGUewiDtxjouT8Qp4dRzrPfGkc0LZaIT/view?usp=sharing) and link it to ```data/refcoco/masks```

3. Download [refcoco/anns](https://drive.google.com/file/d/1Prhrgm3t2JeY68Ni_1Ig_a4dfZvGC9vZ/view?usp=sharing) and put it under ```data/refcoco/```


## Build Spatial Reasoning Dataset
1. Run the following script to process RefCOCO dataset. It generates new annotation files under ```data/refcoco/anns_spatial```
```
python datasets/build_spatial_anno.py
```
2. New dataloader (under development): ```datasets/SpatialReasoningDataset.py```


## Old dataloader

Run ```datasets/refer_segmentation.py``` at root directory to dive into the dataloader and visualize annotations.