# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./pics/visualization1.png "Visualization training data"
[image2]: ./pics/grayscale.png "Grayscaling"
[image3]: ./pics/histogram.png "Histogram"
[image4]: ./pics/visualization2.png "Visualization signs from web"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/patdring/CarND-Term1-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the python methods and the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the number of training data is distributed across the classes (classes = traffic signs).
It can be seen that some classes occur more frequently than others.  

![alt text][image3]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. 

Here is an example of a traffic sign image before and after preprocessing.

Before ...

![alt text][image1]

After preprocessing ...

![alt text][image2]

As a first step, I decided to convert the images to grayscale to reduce the input data ((32, 32, 3) -> (32, 32, 1)) and thus accelerate learning. The color of traffic signs should not be importent for classification.

As a last step, I normalized the image data to make the training also faster and reduce the chance of getting stuck in local optima.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description
|:---------------------:|:---------------------------------------------: 
| Input         		| 32x32x1 grayscale image
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 
| RELU and dropout		|                                               
| Max pooling	      	| 2x2 stride,  outputs 14x14x6   
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x16
| RELU and dropout		|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16
| Flatten               | outputs = 400   
| RELU and dropout		|                            
| Fully connected		| outputs = 120
| RELU and dropout		|                                        
| Fully connected		| outputs = 84        
| RELU and dropout		|                                               
| Fully connected		| outputs = 43                                  
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an AdamOptimizer with a batch size of 128, 50 epochs and a learning rate of 0.001. Dropout was set to 0.6.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 0.956  
* test set accuracy of 0.940

My first implementation was LeNet-5 shown in the udacity TensorFlow Neural Network Lab. The recognition of traffic signs and handwritten letters seemed to me to be a comparable task with very similar requirements. First I used all image color channels (32, 32, 3) and got a validation set accuracy of 0.88. Then I began to grayscale and normalized the input data. The validation set accuracy still not reached 0.93 so I decided to implement the dropout mechanism. This helped to reach a validation set accuracy > 0.93. Training for more than 50 epochs do not increase the validation accuracy. I trained the network 100 epochs but it didn't make much difference. That the accuracies of test and validation are not far apart is for me a sign that the model fulfils its purpose. The pictures from the web later confirmed this again. 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are eight German traffic signs that I found on the web:

![alt text][image4]

The third image (Speed limit (60km/h)) may be difficult for the model to see. The actual traffic sign is small in relation to the surroundings but I wanted to see the models limits. All other traffic signs should be recognized or at least listed among the top 5 probabilities.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set 

Here are the results of the prediction:

| Image                 |Prediction                                     |
|:---------------------:|:---------------------------------------------:|
| Speed limit (30km/h)  | Speed limit (30km/h)   						| 
| Ahead only     	    | Ahead only 									|
| Speed limit (60km/h)	| No passing                                	|
| Turn left ahead  		| Keep right								    |
| Go straight or left	| Go straight or left      						|
| Bumpy road			| Bumpy road             						|
| Stop			        | Stop              							|
| General caution		| General caution      							|

The model was able to correctly guess 6 of the 8 traffic signs, which gives an accuracy of 75.0%. In my opinion this compares favorably to the accuracy on the test set of 94.0% considering the small number of 8 test images, one of which was deliberately "bad".

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. 

The code for making predictions on my final model is located in the 10th and 12th cell of the Ipython notebook.

##### Traffic Sign: Speed limit (30km/h) 
##### Top 5 probabilities:

|  Prediction                                   |  Probabilities        |
|:---------------------------------------------:|:---------------------:|
| Speed limit (30km/h)                          |              0.891756 |
| Roundabout mandatory                          |              0.097194 |
| Speed limit (20km/h)                          |              0.006038 |
| Priority road                                 |              0.002213 |
| Speed limit (50km/h)                          |              0.001248 |

The model is sure to have found the right traffic sign with a probability of 0.89. The gap to the second prediction is large.

##### Traffic Sign: Ahead only 
##### Top 5 probabilities:

|  Prediction                                   |  Probabilities        |
|:---------------------------------------------:|:---------------------:|
| Ahead only                                    |              0.999783 |
| Yield                                         |              0.000075 |
| Turn left ahead                               |              0.000067 |
| Speed limit (60km/h)                          |              0.000041 |
| Go straight or right                          |              0.000019 |

The model is very likely to have found the right traffic sign. The other predictions are almost negligible.

##### Traffic Sign: Speed limit (60km/h) 
##### Top 5 probabilities:

|  Prediction                                   |  Probabilities        |
|:---------------------------------------------:|:---------------------:|
| No passing                                    |              0.607145 |
| No passing for vehicles over 3.5 metric tons  |              0.128009 |
| Ahead only                                    |              0.063933 |
| Speed limit (60km/h)                          |              0.037691 |
| Yield                                         |              0.015939 |

Here the model is wrong with a probability of 0.61 for "No passing". The correct traffic sign appears in the top 5, but with a small probability of 0.037.

##### Traffic Sign: Turn left ahead 
##### Top 5 probabilities:

|  Prediction                                   |  Probabilities        |
|:---------------------------------------------:|:---------------------:|
| Keep right                                    |              0.999227 |
| Turn left ahead                               |              0.000773 |
| Go straight or right                          |              0.000000 |
| Yield                                         |              0.000000 |
| General caution                               |              0.000000 |

Here the model is total wrong with a probability of 0.999 for "Keep right". The correct traffic sign appears in the top 5, but with a small probability of 0.000773.
To misinterpret these two traffic signs (left ahead vs. keep right) is very bad.

##### Traffic Sign: Go straight or left 
##### Top 5 probabilities:

|  Prediction                                   |  Probabilities        |
|:---------------------------------------------:|:---------------------:|
| Go straight or left                           |              0.999578 |
| Traffic signals                               |              0.000140 |
| Roundabout mandatory                          |              0.000133 |
| General caution                               |              0.000079 |
| Turn right ahead                              |              0.000025 |

The model is very likely to have found the right traffic sign. The other predictions are almost negligible.

##### Traffic Sign: Bumpy road 
##### Top 5 probabilities:

|  Prediction                                   |  Probabilities        |
|:---------------------------------------------:|:---------------------:|
| Bumpy road                                    |              0.952507 |
| Bicycles crossing                             |              0.042324 |
| Road work                                     |              0.001893 |
| Traffic signals                               |              0.001217 |
| Road narrows on the right                     |              0.001119 |

The model is very likely to have found the right traffic sign. The other predictions are almost negligible.

##### Traffic Sign: Stop 
##### Top 5 probabilities:

|  Prediction                                   |  Probabilities        |
|:---------------------------------------------:|:---------------------:|
| Stop                                          |              0.999832 |
| Keep right                                    |              0.000092 |
| Turn left ahead                               |              0.000021 |
| Yield                                         |              0.000019 |
| Turn right ahead                              |              0.000017 |

The model is very likely to have found the right traffic sign. The other predictions are almost negligible.

##### Traffic Sign: General caution 
##### Top 5 probabilities:

|  Prediction                                   |  Probabilities        |
|:---------------------------------------------:|:---------------------:|
| General caution                               |              0.998715 |
| Traffic signals                               |              0.001177 |
| Pedestrians                                   |              0.000107 |
| Right-of-way at the next intersection         |              0.000001 |
| Go straight or left                           |              0.000000 |

The model is very likely to have found the right traffic sign. The other predictions are almost negligible.
