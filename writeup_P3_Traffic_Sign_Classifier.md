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

[image1]: ./writeup_images/image_example.png "Sample Image"
[image2]: ./writeup_images/histogram.png "Grayscaling"
[image3]: ./writeup_images/grayscale_example.png "Grayscaling"
[image4]: ./writeup_images/final_accuracy.png "Final Model Accuracy"
[image5]: ./writeup_images/accuracy_1.png "Other Iteration 1"
[image6]: ./writeup_images/accuracy_2.png "Other Iteration 2"
[image7]: ./writeup_images/accuracy_5.png "Other Iteration 3"
[image8]: ./writeup_images/DOE.jpg "New Images"
[image9]: ./writeup_images/new_images.jpg "New Images"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

This writeup is inteded to fulfil the above requirement. The code is implemented in the notebook Traffic_Sign_Classifier.ipynb , and is organized in the same sequence as this document and the rubric. The notebook will be referenced for explanations used in this document.

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

__Code: Notebook Section 1.1__
I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is: __34799__
* The size of the validation set is: __4410__
* The size of test set is __12630__
* The shape of a traffic sign image is: __32 x 32 x 3__
* The number of unique classes/labels in the data set is: __43__

#### 2. Include an exploratory visualization of the dataset.

__Code: Notebook Section 1.2__
It helps us to get a sense of how the  data is organized. We first pick a random image from the training dataset and verify it looks like what we expect it to. Also convert it grayscale to see how well the features are preserved. We print the classification of the image and refer to the signnames.csv to verify it's correct.

This is repeated for a few images, serving as a sanity check on the dataset before starting actual processing.


![alt text][image1]

We also look at the training and validation data histograms. They give us an idea how the distribution count of the signs is spread over the 43 classes.

![alt text][image2]

It appears that the quantity-wise the dataset is skewed towards the first 10 or so classes. A reference to signnames.csv shows that these are speed limit signs, which makes sense owing to their occurrence being more frequent.


|ClassId|SignName					|
|:-----:|:-------------------------:|
|0		|Speed limit (20km/h)		|
|1		|Speed limit (30km/h)		|
|2		|Speed limit (50km/h)		|
|3		|Speed limit (60km/h)		|
|4		|Speed limit (70km/h)		|
|5		|Speed limit (80km/h)		|
|6		|End of speed limit (80km/h)|
|7		|Speed limit (100km/h)		|
|8		|Speed limit (120km/h)		|




### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

__Code: Notebook Section 2.1__

To preprocess the image I started with the suggestion from the instructions. I converted the image to grayscale using the function `cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)`. This is the same as what I did in the visualization step. An example is shown below

![alt text][image3]


In the next step, I normalized the image using `(img - 128)/ 128` operation. This centers the mean around zero, since the data ranges from 0-255. The new data also ranges from -1 to 1 so it's magnitude is between 0 to 1.


The grayscaling converts the shape of the X dataset since the elements have been changed from `[32, 32, 3]` to `[32, 32, 1]`. For example, X_train is converted from an original shape of `[34, 32, 32, 3]` to `[34, 32, 32, 1]`.

Again for a sanity check, I print out vlaues of one element from each of the original `X_train` after grayscaling, and preprocessed `X_train_norm` to compare a few pixels. This is to avoid any miscalculations due to overflows or wrong casting.
Below is a comparison of first three pixels from the first image. Noted that there was one value for each pixel in the final dataset as expected, compared to 3 values in each pixel for original image.

```
# RGB
[28 25 24]
[27 24 23]
[27 24 22]

# Grayscale, Normalized
[-0.796875 ]
[-0.8046875]
[-0.8046875]

```

Verification

(28 + 25 + 24) / 3 = 25.67 -> 26
(26-128) / 128 = -0.796875

The calculation adds up, so we move to the next step.

Now since the shape of X dataset has changed, I would need to make provision for swtiching testing on both the original and normalized dataset. This will be described in the later sections.



#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

__Code: Notebook Section 2.2__

As a starting point, I used the model used in the LeNet lab. I then then modified the layers and added parts like dropout and flattening during the tuning process to arrive at the final architecture. 

| Layer         		|     Description	        							| 
|:---------------------:|:-----------------------------------------------------:| 
| Input         		| 32x32x3 RGB image   									| 
| Convolution 5x5     	| 1x1 stride, VALID padding, outputs 28x28x6 			|
| RELU					|														|
| Max pooling	      	| 2x2 stride and ksize, SAME padding  outputs 14x14x16 	|
| Convolution 5x5	    | 1x1 stride, VALID padding, outputs 10x10x6			|
| RELU					|														|
| Dropout				| Variable for tuning, final result using 0.7			|
| Max pooling	      	| 2x2 stride and ksize, SAME padding  outputs 5x5x16 	|
| Flatten				| Input 5x5x16, Output 400								|
| Fully connected		| Input 400, Output 120									|
| RELU					|														|
| Dropout				| Variable for tuning, final result using 0.7			|
| Fully connected		| Input 120, Output 43									|
| Softmax				| 														|
|						|														|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

__Code: Notebook Section 2.3__

To train the model I used certain parameter definitions as variables. This allowed quick changes for testing.

```
# Set parameters
EPOCHS = 50
BATCH_SIZE = 64
rate = 0.0005
col_channels = 1 # 1: for grayscale and 3: for color images
ker_size = 5
num_depth_1 = 6
num_depth_2 = 16
num_depth_3 = 400
num_depth_4 = 120
dropout_train = 0.72
```

As mentioned earlier, the 'col_channels' allows for switching testing between original and denormalized dataset.
The num_depth channels enable to edit the dimensions of the layers in the network architecture.


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

__Code: Notebook Section 2.4__

My final model results were:
* training set accuracy of __92.6%__
* validation set accuracy of __94.7%__ 
* test set accuracy of __99.8%__


A DOE (Design of Experiments) approach was taken to arrive at the solution. With the hyperparameters defined as variables (described in the section above), it was possible to quickly make changes.
Many iterations of tuning were needed to achieve the final solution. Changes to hyperparameters were done manually. The magnitude and direction of change was decided based on previous results (more like human driven gradient descent, haha). 

![alt text][image4]



__Some Earlier Iterations__
Shown below are some earlier iteration that did not meet the benchmark. This is to illustrate the steps taken.

![alt text][image5]



![alt text][image6]



![alt text][image7]


Some salient points about the decisions made during the process are mentioned below
* The first architecture used as a starting point was LeNet, with some minor modifications like flattening, etc. This provided a baseline result, around 80%, and gave me direction to proceed. LeNet model was chosen as a starting point because it has a demonstrated capability of working on image identification.
* Tuning hyperparameters with the base architecture did not yield much improvement over exsiting result
* The architecture was adjusted to include dropouts after the RELU activation
* Accuracy for both the training set and validation sets were monitored and based on whether it indicated overfitting or underfitting, paramters were modified

__Parameters Tuned__
* EPOCH - convergence
* Batch Size - Smaller batch slower, but more number of updates within training
* Learning Rate: Higher improves faster but erratic at higher epochs when close to convergence. Smaller rate leads to slower convergence, but stable and better results at higher EPOCHS

To better describe how I did the tuning using the DOE approach, here is a list showing some trials I did.


![alt text][image8]


__Reasoning behind Layers Applied__
Pooling allows to get more important features and ignore the unneccessary noise, irrelevant feature
Covolution works better at identifying features. More convolution layers, help identify larger features
Dropout prevents overfitting, and allows the model to learn on a limited subset, preparing it to identify features from uknown images, and difficult ones having fewer features visible.

Repeated tests on the training and validation dataset showed good results, indicating the model is robust and performs consistently. Since the model had only trained on a shuffled training dataset, it's perfomance on validation set was indication of good fit. Results on the test set which it had not seen at any point before also provided evidence that it works well on unseen images.




### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

__Code: Notebook Section 3.1__

To test the validity of the model, we now try it on some images obtained from the internet.
Below is a picture of the set of 10 images used for section. They were randomly picked up from differnt sourcs on an image search.


![alt text][image9]


My expectation was that the model may not be able to perform at the same level on these images. Some of the reasons are 
    * The original images were different sizes, and resized to 32x32 whiich may have altered original proportions
    * Some images were taken at an angle. Although the training data may have some like that, these are the difficult ones to identify
    * Images like the 1st and 6th one have a lot of background features in them which may increase complexity


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).


__Code: Notebook Section 3.2__

The prediction results of the new set are below

```
Prediction:  [ 9 13 17  9 38 17  3 11  2  4]
Actual    :  [18 13 14  9 26 17  2 11  3 27]
```

Translating the above results into the dictionary provided in `signnames.csv`

| Image			        |     Prediction	        			| 
|:---------------------:|:-------------------------------------:| 
| General caution      	| No passing   							| 
| Yield     			| Yield 								|
| Stop					| No entry								|
| No passing	     	| No passing					 		|
| Traffic signals		| Keep right      						|
| No entry      		| No entry   							| 
| Speed limit (50km/h)	| Speed limit (60km/h) 					|
| Right-of-way			| Right-of-way							|
| Speed limit (60km/h)	| Speed limit (50km/h)					|
| Pedestrians			| Speed limit (70km/h)    				|


Remarks: The model predicted 4 of the above correctly. It did get two others close, but mixed up picture 7 and 9. Which on examples like picture 10, it was way off the mark since the predicted sign does not resemble the actual one.



__Accuracy__

__Code: Notebook Section 3.3__

The test on new images reported 40% accuracy, which is below expectations. This may be attributed to the new images not being correct proportions and stray background features. 

```
Test Accuracy = 0.400
```



#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The softmax code is located inside the __notebook section 3.2__. The output is shown below



__Softmax__
The softmax values obtained for the new images are below. 
It appears that the model has high confidence even for incorrect predictions. More details are described below

```
TopKV2(values=array([[  7.83934951e-01,   2.05991596e-01,   1.00617483e-02],
       [  1.00000000e+00,   1.19369869e-20,   8.05818827e-23],
       [  9.98682320e-01,   1.11214083e-03,   9.03480031e-05],
       [  1.00000000e+00,   9.60996515e-18,   1.34330496e-20],
       [  9.97551501e-01,   1.34418067e-03,   1.10414182e-03],
       [  1.00000000e+00,   1.06440373e-25,   2.53026196e-28],
       [  9.93717670e-01,   5.19398460e-03,   7.93453481e-04],
       [  1.00000000e+00,   1.95670265e-15,   1.26712427e-18],
       [  8.57524574e-01,   1.42436892e-01,   3.85461535e-05],
       [  8.36983085e-01,   1.62974298e-01,   1.90957708e-05]], dtype=float32), indices=array([[ 9, 12, 35],
       [13, 34, 35],
       [17,  9, 41],
       [ 9, 41, 17],
       [38, 29, 36],
       [17, 32, 14],
       [ 3,  1,  0],
       [11, 30, 28],
       [ 2,  3,  1],
       [ 4,  1, 26]], dtype=int32))

```


Since the model shows very high confidence in the predictions, only the first 2 or 3 are significant at most. The later values are very close to 0, so I am not listing them here.
Listing the top 3 probabilities for the images 1-5 below

__Image 1__

| Image			        |     Prediction	        			| 
|:---------------------:|:-------------------------------------:| 
| General caution      	| No passing   							| 


Prediction: `[ 9, 12, 35]`

| Probability         	|     Prediction	        			| 
|:---------------------:|:-------------------------------------:| 
| 0.78         			| No passing   							| 
| 0.21 					| Priority road 						|
| 0.0.1					| Ahead only							|





__Image 2__

| Image			        |     Prediction	        			| 
|:---------------------:|:-------------------------------------:|
| Yield     			| Yield 								|


Prediction: `[13, 34, 35]`

| Probability         	|     Prediction	        			| 
|:---------------------:|:-------------------------------------:| 
| 1.0         			| Yield   								| 
| 1.2 e-20 				| Turn left ahead 						|
| 8.0 e-23				| Ahead only							|




__Image 3__

| Image			        |     Prediction	        			| 
|:---------------------:|:-------------------------------------:|
| Stop					| No entry								|


Prediction: `[17,  9, 41]`

| Probability         	|     Prediction	        			| 
|:---------------------:|:-------------------------------------:| 
| 0.998         		| No entry   							| 
| 1.11 e-03 			| No passing 							|
| 9.0 e-05				| End of no passing						|




__Image 4__

| Probability         	|     Prediction	        			| 
|:---------------------:|:-------------------------------------:| 
| No passing	     	| No passing					 		|


Prediction: `[ 9, 41, 17]`

| Probability         	|     Prediction	        			| 
|:---------------------:|:-------------------------------------:| 
| 1.0         			| No passing   							| 
| 9.61e-18 				| End of no passing 					|
| 1.34e-20				| No entry								|




__Image 5__

| Probability         	|     Prediction	        			| 
|:---------------------:|:-------------------------------------:| 
| Traffic signals		| Keep right      						|


Prediction: `[38, 29, 36]`

| Probability         	|     Prediction	        			| 
|:---------------------:|:-------------------------------------:| 
| 0.99         			| Keep right   							| 
| 1.34e-03 				| Bicycles crossing 					|
| 1.10e-03				| Go straight or right					|


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


