
from PIL import Image
import os


aug_path = 'C:/Users/micha/Desktop/2018_dsb/input/stage1_aug_train2/'

# Get train and test IDs
aug_ids = next(os.walk(aug_path))[1]


for ax_index, image_id in enumerate(aug_ids,1):
	if ax_index % 1000 == 0:
		print(ax_index)
	mask_file = aug_path+"{}/masks/{}.png".format(image_id,image_id)
	col = Image.open(mask_file)
	grey = col.convert('L')
	bw = grey.point(lambda x: 0 if x<128 else 255,'1')
	bw.save(mask_file)


