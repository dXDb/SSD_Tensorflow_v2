# SSD (Single Shot MultiBox Detector) - Tensorflow 2.3.0 with Inception v3

## Preparation
- Download PASCAL VOC dataset (2007 or 2012) and extract at `./data`
- Install necessary dependencies:
```
pip install -r requirements.txt
```

## Make Data list

- make txt files like in `data_list/train` or  `data_list/val`
- change  `voc_data.py`  file at 33 line self.idx_to_name with your custom class 

```
python data_list/create_data_list.py
```

## Data Augmentation

change  `train.py`  file at 82 line augmentation=['what', 'you', 'want', 'patch'] with yours. It will work with a 50% chance.

- patch : Sample a patch so that the minimum Jaccard overap with the objects is 0.1, 0.3, 0.5, 0.7 or 0.9.
- flip : horizontal flip
- brightness : adjust the brightness by a random factor (-0.32 ~ 0.32)
- contrast : adjust the contrast by a random factor (0.5 ~ 1.5)
- hue : adjust the hue by a random factor (-0.18 ~ 0.18)
- saturation : adjust the saturation by a random factor (0.5 ~ 1.5)

## Training

Arguments for the training script:

```
>> python train.py --help
usage: train.py [-h] [--data-dir DATA_DIR] [--data-year DATA_YEAR]
                [--arch ARCH] [--batch-size BATCH_SIZE]
                [--num-batches NUM_BATCHES] [--neg-ratio NEG_RATIO]
                [--initial-lr INITIAL_LR] [--momentum MOMENTUM]
                [--weight-decay WEIGHT_DECAY] [--num-epochs NUM_EPOCHS]
                [--checkpoint-dir CHECKPOINT_DIR]
                [--pretrained-type PRETRAINED_TYPE] [--gpu-id GPU_ID]
```
Arguments explanation:
-  `--data-dir` dataset directory (must specify to VOCdevkit folder)
-  `--data-year` the year of the dataset (2012)
-  `--arch` SSD network architecture (ssd300 or inception)
-  `--batch-size` training batch size
-  `--num-batches` number of batches to train (`-1`: train all)
-  `--neg-ratio` ratio used in hard negative mining when computing loss
-  `--initial-lr` initial learning rate
-  `--momentum` momentum value for SGD
-  `--weight-decay` weight decay value for SGD
-  `--num-epochs` number of epochs to train
-  `--checkpoint-dir` checkpoint directory
-  `--pretrained-type` pretrained weight type (`base`: using pretrained VGG backbone, other options: see testing section)
-  `--gpu-id` GPU ID

- how to train SSD300(VGG16) using PASCAL VOC2012 Custom for 100 epochs:

```
python train.py --data-dir ./data/VOCdevkit --data-year 2012 --num-epochs 100
```

- how to train Inception v3 using PASCAL VOC2012 for 100 epochs on GPU 1 with batch size 8 and save weights to `./checkpoints`:

```
python train.py --data-dir ./data/VOCdevkit --data-year 2012 --arch inception --num-epochs 100 --batch-size 8 --checkpoint_dir ./checkpoints --gpu-id 1
```

## Testing
Arguments for the testing script:
```
>> python test.py --help
usage: test.py [-h] [--data-dir DATA_DIR] [--data-year DATA_YEAR]
               [--arch ARCH] [--num-examples NUM_EXAMPLES]
               [--pretrained-type PRETRAINED_TYPE]
               [--checkpoint-dir CHECKPOINT_DIR]
               [--checkpoint-path CHECKPOINT_PATH] [--gpu-id GPU_ID]
```
Arguments explanation:
-  `--data-dir` dataset directory (must specify to VOCdevkit folder)
-  `--data-year` the year of the dataset (2012)
-  `--arch` SSD network architecture (ssd300 or inception)
-  `--num-examples` number of examples to test (`-1`: test all)
-  `--checkpoint-dir` checkpoint directory
-  `--checkpoint-path` path to a specific checkpoint
-  `--pretrained-type` pretrained weight type (`latest`: automatically look for newest checkpoint in `checkpoint_dir`, `specified`: use the checkpoint specified in `checkpoint_path`)
-  `--gpu-id` GPU ID

- how to test the first training pattern above using the 100th epoch's checkpoint with Inception V3:

```
python test.py --data-dir ./data/VOCdevkit --data-year 2012 checkpoint_path ./checkpoints/ssd_epoch_100.h5 --arch inception
```

- how to test the second training pattern above using the 100th epoch's checkpoint, using only 40 examples with VGG16:

```
python test.py --data-dir ./data/VOCdevkit --data-year 2012 --arch ssd300 --checkpoint_path ./checkpoints/ssd_epoch_100.h5 --num-examples 40
```

## score APs, mAP (IoU >= 0.5)

```
python voc_eval.py --data-dir ./data/VOCdevkit --data-year 2012 
```

- You have to change `voc_eval.py`  at 137 line with your custom data.

## Reference

- Single Shot Multibox Detector paper: [paper](https://arxiv.org/abs/1512.02325)
- Caffe original implementation: [code](https://github.com/weiliu89/caffe/tree/ssd)
- Base : [code] (https://github.com/ChunML/ssd-tf2)
