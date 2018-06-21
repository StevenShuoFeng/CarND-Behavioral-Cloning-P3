# ** Project-3 Report: Behavioral Cloning** 

## Shuo Feng Jun-21-2018

---
**Hight level summary**

This project intends to use deep learning to train a regerssion network that can predict best steering angles based on the image of view at any moment. The solution is a supervised regression approach. During training, driver (me) controls the car to drive properly in the simulator and the keyboard input of the steering control as well as the corresponding screen shot of the view are recorded. Then, the images (screenshots) are used as the training data and the recorded steering angles are used as training label to train a single output neural network. The built model is then used to provide steering controls while simulator is in autonomous driving mode.

**Goals and Steps**
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

**File Structure in Repo**
-- model_center_andLeftRight_cropped_TwoTracks_NvidiaNet.h5: the final model
-- model.py and model.ipynb: the souce code for building the model
-- README.md: the final writeup file
-- /writeupImg: recorded videos and images for writeup

[//]: # (Image References)

[image1]: ./writeupImg/angleHist.png "Distribution of angles"
[image2]: ./writeupImg/lossCurve.png "Loss curve"
[image3]: ./writeupImg/nvidiaNet.png "NvidiaNet Structure"
[image4]: ./writeupImg/v1_img.png "Track 1 Video Cover Photo"
[image5]: ./writeupImg/v2_img.png "Track 2 Video Cover Photo"


### Model Architecture and Training Strategy
#### 1. Solution Design and Iterative Model Building Approach

As stated in the high level summary above, the ultimate goal is to build a neural network that for each individual view image as an input, output the steering angle. The development envolves a few iterations of updating network architecture and adding more training data as described below.

I started with 1 lap of training data from track-1 and a two-layer convolution neural network. The number of training samples is about 3k. From the loss curve, the training loss decrease while the validation loss basically stay unchanged. This is a sign of underfitting. 

Then, the AlexNet and the NvidiaNet are tried out. From the learning curve, either the validation loss decay very slow or it's constantly higher than the training loss curve. Both these are signs of underfitting. Two more laps of training data from track-1 was added, one lap driving clockwise and the other counter clockwise.

The model built at this moment can drive the car well on track-1 in autonomous mode. The trial on track-2 failed. 

Finally, 2 more laps of training data is collected from track-2, one for each direction of driving. The total number of training samples is aboout 16k. The final structure of the neural network is shown in session #3. 

#### 2. Data Pre-processing and Augmentation

During the trials of adding more data in the previous session, different data processsing procedures are tried out. They're listed below:

- Zero-center the image by subtracting 128 for all pixels and all channels
- Crop the top (60 rows) and bottom (20 rows) of the view to remove context from sky and engine cover of the car
- Create augmented image by flip the image and the sign of the corresponding angle
- Make use of the images from left and right camera with adjusted steer angle

These steps help to reduced the complexity of the images and also provide more augmented training data.
During the training data collection, the distribution of steering angles in the train and validation set are monitored to make sure there's no obvious imbalance. The distribution of steering angles in the data set used is shown below:

![alt text][image1]

#### 3. Final Model Architecture

The final model used in this project is the neural network developed by [Nvidia](https://devblogs.nvidia.com/deep-learning-self-driving-cars/) and the AlexNet was also tried out. The whole seesion of model structure building can be find in [model.py lines 70-110](https://github.com/StevenShuoFeng/CarND-Behavioral-Cloning-P3/blob/master/model.py#L71). 

The beginning layers are the same for the two, which includes the zero-centered proprocessing and croppoing of upper and lower part of each image (model.py lines 70-75).

Within line 90~103, the Nvidia net structure is defined. The [original acticture](https://devblogs.nvidia.com/deep-learning-self-driving-cars/) is modified by removing the first fully-connected layer of size 1164 due to the computation complexity. It includes 5 convolutional layers followed by 4 fully connected layers. And the output is a single float value. The amount of weights and output shape is summarized in the table below. More details about size of convolution kernel and step sizes can be found in the [source code](https://github.com/StevenShuoFeng/CarND-Behavioral-Cloning-P3/blob/master/model.py#L90).

|Layer (type)                     |Output Shape          |Param #     |Connected to                 |  
|:---|:---|:---|:---|
|lambda_1 (Lambda)                |(None, 160, 320, 3)   |0           |lambda_input_1[0][0]         |    
|cropping2d_1 (Cropping2D)        |(None, 80, 320, 3)    |0           |lambda_1[0][0]               |    
|convolution2d_1 (Convolution2D)  |(None, 38, 158, 24)   |1824        |cropping2d_1[0][0]           |    
|convolution2d_2 (Convolution2D)  |(None, 17, 77, 36)    |21636       |convolution2d_1[0][0]        |    
|convolution2d_3 (Convolution2D)  |(None, 7, 37, 48)     |43248       |convolution2d_2[0][0]        |    
|convolution2d_4 (Convolution2D)  |(None, 5, 35, 64)     |27712       |convolution2d_3[0][0]        |    
|convolution2d_5 (Convolution2D)  |(None, 3, 33, 64)     |36928       |convolution2d_4[0][0]        |    
|flatten_1 (Flatten)              |(None, 6336)          |0           |convolution2d_5[0][0]        |    
|dropout_1 (Dropout)              |(None, 6336)          |0           |flatten_1[0][0]              |    
|dense_1 (Dense)                  |(None, 100)           |633700      |dropout_1[0][0]              |    
|dropout_2 (Dropout)              |(None, 100)           |0           |dense_1[0][0]                |    
|dense_2 (Dense)                  |(None, 50)            |5050        |dropout_2[0][0]              |    
|dense_3 (Dense)                  |(None, 10)            |510         |dense_2[0][0]                |    
|dense_4 (Dense)                  |(None, 1)             |11          |dense_3[0][0]                |   

Total params: 770,619
Trainable params: 770,619
Non-trainable params: 0

The structure of the original network is shown below:

![alt text][image3]


#### 4. Creation of the Training Set & Training Process

As described in session 1, there're about 16k samples in total be collected during the iterations. These data are splited into training and validing data sets by a 80-20 ratio after random shuffle. 

A model is built with the NvidaNet achitecture above. And this version of model, with Adam optimizer, successfully drive on track-1 and track-2 without too much shades in autonomous mode.

#### 5. Final Results

The videos of this model running in simulator are recorded and the mp4 file can be found [here] (https://github.com/StevenShuoFeng/CarND-Behavioral-Cloning-P3/tree/master/writeupImg). 

[![Video for Track-1](http://img.youtube.com/vi/iCY66k_YYVc/0.jpg)](http://www.youtube.com/watch?v=iCY66k_YYVc)

[![Video for Track-2](http://img.youtube.com/vi/SAyf1X6M6WM/0.jpg)](http://www.youtube.com/watch?v=SAyf1X6M6WM)

