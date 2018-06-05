# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model  consists of 9 layers network, including a normalization layer, 5 convolutional layers
and 3 fully connected layers. (model.py lines 61-75) 

We use strided convolutions in the first three convolutional layers with a 2X2 stride and a 5X5 kernel and a non-strided convolution with a 3X3 kernel size in the last two convolutional layers.

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on Udacity provided data sets and test with diffrent epochs numbers by checking mean squared error on the training set and validation set, to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 77).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road , also filped the center lane driving picture as augmented data.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The start point was to adopt  CNN network mentioned in NVIDA "End to End Learning for Self-Driving Cars" paper.I thought this model might be appropriate because it is proved to be very successful in Road image recognition, which is quite close as this behaviour cloning case.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. First, I use 'fit' function to train the network, and the memeroy is not sufficient, so I use the 'mole.fit_generator' to split the data, and it worked well.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, to improve the driving behavior in these cases, I augmented the data by fliping the center image.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

I also tried adding Maxpooling layers and dropout layers after convolution layers, or fully conncetion layers,  but still the NVIDIA network works best. So I kept the NVIDIA structure as the final model architecture.The network consists of 9 layers, including a normalization layer, 5 convolutional layers and 3 fully connected layers(picture is from NVIDIA papaer).

![avatar](/img/NVIDA.png)


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I used the captured sample data from Udacity, contains the image from 1st track, center, left and right captured picute.Below are examples:

![avatar](/img/center_2016_12_01_13_30_48_287.jpg) 
             center image 
![avatar](/img/left_2016_12_01_13_30_48_287.jpg) 
             left image
![avatar](/img/right_2016_12_01_13_30_48_287.jpg)
             right image
To augment the data sat, I also flipped center images and angles, For example, here is an image that has then been flipped:

![avatar](/img/flip.png)
             filp of the center image

After the collection process, I had 32144 number of data points, among which 24108 is from Udacity sample data, and another 8036 is from center image flipping. And these data been trimed, only the road pixles are kept for less noise. 

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5  after several round test with diffrent epoch number test, I used an adam optimizer so that manually training the learning rate wasn't necessary.
