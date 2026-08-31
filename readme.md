# dogBreedprediction

A Jupyter notebook that fine-tunes a VGG19 model to classify dog breeds, from July 2021.

## What it is

Uses the Kaggle [dog-breed-identification](https://www.kaggle.com/c/dog-breed-identification) dataset (~10k labeled images across ~120 breeds). Loads images and labels, one-hot encodes breeds, and fine-tunes a pretrained VGG19 (transfer learning: freezes the base, trains new top layers, then unfreezes a few more layers for a second training pass). Also experiments with Keras `ImageDataGenerator` for image augmentation. Saves a few `.h5` model checkpoints along the way and runs a prediction on a sample `dog.jpg`.

## Stack

- Python, Jupyter notebook
- Keras/TensorFlow (VGG19 pretrained on ImageNet, `EarlyStopping`, `ImageDataGenerator`)
- pandas, numpy

## Running it

Needs the Kaggle dataset (`train/` images + `labels.csv`, not included in the repo) plus a `dog.jpg` test image. Not executed as part of this review — the dataset wasn't fetched, and there's a typo in the augmentation cell (`featurewise_std_normaliation`, `datagen.fix` instead of `.fit`) that would need fixing to run end-to-end.

## Status

A learning exercise in transfer learning and image classification from July 2021; not maintained. The notebook is left mid-experiment (last cell comment: "further might need to train it more to get the correct name").
