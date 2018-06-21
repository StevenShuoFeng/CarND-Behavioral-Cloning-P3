# ** Project-3 Report: Behavioral Cloning** 

## Shuo Feng Jun-21-2018

---
**Hight level summary**

This project intends to use deep learning to train a regerssion network that can predict best steering angles based on the image of view at any moment. The solution is a supervised regression approach. During training, driver (me) controls the car to drive properly in the simulator and the keyboard input of the steering control as well as the corresponding screen shot of the view are recorded. Then, the images (screenshots) are used as the training data and the recorded steering angles are used as training label to train a single output neural network. The built model is then used to provide steering controls while simulator is in autonomous driving mode.

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeupImg/angleHist.png "Distribution of angles"
[image2]: ./writeupImg/lossCurve.png "Loss curve"
[image3]: ./writeupImg/nvidiaNet.png "NvidiaNet Structure"
[image4]: ./writeupImg/placeholder_small.png "Recovery Image"
[image5]: ./writeupImg/placeholder_small.png "Recovery Image"
[image6]: ./writeupImg/placeholder_small.png "Normal Image"
[image7]: ./writeupImg/placeholder_small.png "Flipped Image"


### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

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

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
