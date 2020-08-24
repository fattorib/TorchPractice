# Intel Image Classification

The Intel Image Classification dataset consists of images of the natural world. The images are broken into 6 categories: Buildings, Forests, Glaciers, Mountains, Seas, and Streets. Here, I design and implement a basic CNN that can acheive over 80% accuracy on the test data.

## Data
We are given three datasets: Train, Test, and Predict. Train/Test are labelled and (unsurprisingly) Predict is not. All images given as 150 x 150 RGB. Here is a selection of the images from Train: ![text](https://github.com/fattorib/TorchPractice/blob/master/Conv/images/Classes.png)

As you can see there are a very wide range of images to train on!

## Model 
The model we used was implemented in Pytorch. It consists of 4 Convolutional Layers, each with max pooling in between. We then have 3 fully connected layers with dropout and a Log-Softmax output used for the actual predictions.

## Training 
The model was trained for about 30 epochs with SGD. The first 15 epochs were at a learning rate of 0.01. After that, the learning rate was decreased to 0.005 for another 10 epochs and then 0.001 for the final 5 epochs. Training was conducted on an Nvidia GTX 1050 Ti. It took about an hour in total to train on 14000 images. The test accuracy of the saved model was 81%. 

Going forward I will be making use of cloud services to allow much faster model training. 

## Predictions
With the trained model, we are able to apply it the the Predict set and view its predictions. This is implemented in ModelPrediction.py which outputs the image as well as its class probabilities: ![text](https://github.com/fattorib/TorchPractice/blob/master/Conv/images/Predictions.png)
