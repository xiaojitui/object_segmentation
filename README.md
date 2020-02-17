# object_segmentation

This is a model to do image segmentation by U-Net. 
It is developed based on: https://github.com/jakeret/tf_unet



The ground truth is recorded in 'annotate.txt'

To train the U-Net, run:
python unet_main.py

The trained model will be saved in 'pre_trained' folder. 

To do prediction, use the 'predict' function in 'unet_main.py'
