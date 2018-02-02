# Image Segmentation 2018 DSB

## Background

In my spare time I am going to be working on building out a response for the 2018 data science bowl hosted on Kaggle. This year's competition is based around generated image masks for cells which appear in images. The images are mostly 256x256 but some extend to be 256x320, for the most part the data looks clean and make for a standard segmentation problem. The initial pipeline will be built around a Unet architecture. For now it is a relatively small network, but testing will be done to expand the size of the network. 

## Initial Results

After implementing an initial Unet model and building out the pipeline I have been experimenting with changes to the architecture, mainly adding additional layers to see how much I can cram through an Nvidia 1080XTi card. After climbing into the top 50 it has been very minimal improvement. 

Areas to improve will be the data augmentation methods, maybe normalization methods to make the inputs more similar (https://www.kaggle.com/kmader/normalizing-brightfield-stained-and-fluorescence) could be a good route for this, and architecture changes. Leaders are likely making use of higher computing then what I have available, so my efforts will likely have to be around data augmentation.  

## Notes
The mini Unet is enough to get a leaderboard score of .269 with relatively little data and no augmentation. Next iterations are going to use heavily augmented data, this may lead to overfitting, but it is worth testing.

Deeper Unet with 14K data augmentation gets to .334 or something which is good enough for a early top 20 spot which is cool. Next steps are going to build out wauys to augment the dataset...

Issue with continuing to scale up the augmentation is that it is likely to cause the repetition of data which may lead to overfitting in the long run. Currently testing a dataset with around 26K samples, log loss is performing well. Time shall tell. (this did ok, but similar results to the 14K set I made, probably because it begins to repeat.

Run log 11: deeper UNET performing well with heavy dropout added as well as some additional convolutional layers. will likely restart it and run it for another 15-20 epochs to see how it improves. ended with loss of around .041 and mean iou .87~ was a score of .325, I think the model can improve if run for additional epochs so will run for 30 additional epochs, saving final and best models.

There are oddly shaped files in the test set? so although training has been done on the 256x256 set we may need to evaluate images of any set of dimensions? (odd as in 151x500)

11/25: Running generator unet on normalized data, also added functionality to save after every run. Need to test how runs with high mean IOU scores perform.

## Further Testing/TODO List

-Test code to use generators for data augmentation (built, need to debug)
  Can go ahead and test on single images. Or another thing to do is apply noise ahead of time
  - Currently not performing well, but the networks are effectively trained much less than the other Unets I have been training
  - Steps: Augment data to a few thousand, normalize, run through generator scripts

-test gaussian noise layer? might help with data augmentation

-Start building ensemble... look to some other challenges on ensemble methods in segmentation challenges... since one model is not going to be enough to win long term... can also look to other challenges on how to implement ensemble methods for segmentation tasks.

-add images to readme

## Done List

-Deploy deeper Unet I think I can get at least up to 512 conv layers using my 1080X Nvidia card (DONE)
  1080 is able to handle deeper Unet, but it requires that we use single image batches. Was able to build the Unet down to 1024 level.
  
-Built Normalization script. makes images a uniform fluorescent color rather some being black and white and some purple. will apply to runs after finishing the current run 11 tests. (1/25)
