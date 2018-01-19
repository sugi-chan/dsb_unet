# Image Segmentation 2018 DSB

## Background

In my spare time I am going to be working on building out a response for the 2018 data science bowl hosted on Kaggle. This year's competition is based around generated image masks for cells which appear in images. The images are mostly 256x256 but some extend to be 256x320, for the most part the data looks clean and make for a standard segmentation problem. The initial pipeline will be built around a Unet architecture. For now it is a relatively small network, but testing will be done to expand the size of the network. 

## Initial Results

The mini Unet is enough to get a leaderboard score of .269 with relatively little data and no augmentation. Next iterations are going to use heavily augmented data, this may lead to overfitting, but it is worth testing.

## Further Testing/TODO List

-Test code to use generators for data augmentation (built, need to debug)

-Deploy deeper Unet I think I can get at least up to 512 conv layers using my 1080X Nvidia card (TODO)

-add images to readme
