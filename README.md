## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
---

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! 

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the training data distributed:

<img src="./save figures/training_hist.png" width="480" />

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because it is much easier to get the color change gradent by using the gray scale and it can decrease the saze of training mateix. 

Here is an example of a traffic sign image before and after grayscaling.

<img src="./save figures/gray-scale.png" width="120" />

As a last step, I normalized the image data because the initial gray scale figures may vary a lot that could decrease the accurancy of training. 

<img src="./save figures/normalized.png" width="120" />

And due to it is highly ordered data set when considering the class, if I put this kind of data set into training, it may influence a lot how the training will perform in the test data. So I shuffled the training data before put them into the model.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 gray scale image   					| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6  				|
| RELU                  |                                               |
| Max pooling           | 2x2 stride,  outputs 5x5x16                   |
| Flatten               | output 400                                    |
| Fully connected		| output 120  									|
| RELU                  |                                               |
| Fully connected       | output 84                                     |
| RELU                  |                                               |
| Fully connected 		| output 45    									|
|						|												|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an AdamOptimizer with cross entropy method, learning rate is 0.0009, batch size equals 128 and epochs equals 50. It is very much enough to set the epochs to be 50. There would be not much improvement of the model but longer simulation time with higher epochs. 

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.997
* validation set accuracy of 0.941 
* test set accuracy of 0.913

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

<img src="./GermanSignTest/1.png" width="120" />
<img src="./GermanSignTest/2.png" width="120" />
<img src="./GermanSignTest/5.png" width="120" />
<img src="./GermanSignTest/6.png" width="120" />
<img src="./GermanSignTest/8.png" width="120" />
<img src="./GermanSignTest/9.png" width="120" />


The first image might be difficult to classify because the traffic sign itself is very similar to other signs so it would be hard to distringuish this sign from the others. The second image is different because the model should correctly distinguish the angle of the arrow in the sign so that it can give the right prediction. And this sign itself is similar to other turning signs. The third image is diffcult is that there are multiple speed limit signs as potential prediction. SO the model should calculate the edge of the number correctly so that a right prediction can be given. The forth sign is diffcult because itself is similar to the "keep right" sign. So the model may have diffculty getting the right angle of the arrow. The fifth sign is diffculty is that this sign is reletively too simple. So that the detection may be influenced by many other conditions, and the model should subtrack those conditions and get the right prediction eventually. The last figure is hard for it is on the opposite of the fifth sign. This one is too complacated so that the model should detect the edges very carefully so that it can get the right prediction.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			                          |     Prediction	        					| 
|:---------------------------------------:|:-------------------------------------------:| 
| Right-of-way at the next intersection   | Right-of-way at the next intersection		| 
| Speed limit (30km/h)                    | Speed limit (50km/h)	                    |
| Keep right                              | Keep right                                  |
| Turn left ahead                         | Turn left ahead                             |
| General caution                         | General caution                             |
| Road work                               | Turn right ahead                            |


The model gives an accuracy of 66.7%. THis gives correct prediction of 4 on the given 6 German triffic signs. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

For the first image, the top five soft max probabilities were

| correct?            	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| yes         			| Right-of-way at the next intersection   		| 
| no     				| Pedestrians       							|
| no					| Beware of ice/snow                 			|
| no	      			| Speed limit (20km/h)                			|
| no				    | Speed limit (30km/h)  						|


For the second image: (Even though the fist guess is wrong, but the second guess hit the correct one)


| correct?          	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| no					| Speed limit (50km/h)              			|
| yes         			| Speed limit (30km/h)                  		| 
| no	      			| Speed limit (60km/h)              			|
| no                    | Wild animals crossing                         |
| no				    | Speed limit (80km/h)							|

For the third image:


| correct?          	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| yes         			| Keep right                            		| 
| no					| Speed limit (20km/h)							|
| no	      			| Speed limit (30km/h)          				|
| no				    | Speed limit (50km/h)    						|
| no	      			| Speed limit (60km/h)              			|

For the forth image:


| correct?          	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| yes         			| Turn left ahead                         		| 
| no     				| Slippery road      							|
| no				    | No passing             						|
| no					| Speed limit (20km/h)							|
| no	      			| Speed limit (30km/h)          				|

for the fifth image:


| correct?          	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| yes         			| General caution                        		| 
| no	      			| Traffic signals                   			|
| no					| Speed limit (20km/h)							|
| no	      			| Speed limit (30km/h)          				|
| no				    | Speed limit (50km/h)    						|

for the sixth image:


| correct?          	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| no         			| Turn right ahead                       		|  
| no     				| Priority road             					|
| no					| Go straight or right                  		|
| no	      			| Dangerous curve to the right 	 				|
| no				    | Traffic signals     							|

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


