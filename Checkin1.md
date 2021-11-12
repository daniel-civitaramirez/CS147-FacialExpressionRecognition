Outline Details
 
Title:  Facial Expression Recognition using Neural Networks to Associate Colors to Emotions.
 
Who: Daniel Civita Ramirez (dcivita1), Jimena Barrio Arbona(jbarrioa), Elena Abati Lopez (eabati)
 
Introduction: We chose to create a Facial Expression / Emotion Recognition classification model. Furthermore, once the emotion is recognized we will apply a filter to the image that colors the face identified in the picture with a color representative of the identified emotion. For example, if our model recognizes an angry face in an image, the face would turn red, if it recognizes happiness, the face will turn blue.
 
We arrived at this topic by thinking about how we could use deep learning to do something that could be beneficial to society. While reading about different ways deep learning models had been used in the past we came upon facial recognition and the many different outcomes or goals these projects had. To turn it into something that could help other people, and do something nobody had ever done before, we thought of people with diverse mental disorders that struggle to recognize human emotions. We were motivated by the idea of creating a tool that could help them. Thus, we came up with the idea of associating a different color to each one of the emotions our model could recognize. Through the use of Computer Vision, our pipeline can then recognize the outline of the face in the image and turn the face to each one of the colors associated after the emotion classification.
 
Related Work: 'Living List' of similar projects / implementations:
Convolutional Neural Network For Facial Expression Recognition: Stanford paper that used CNN model designed for a facial expression recognition task with PyTorch, which classifies seven facial emotion categories. Their model had the following architecture scheme, in which M and N are number of convolutional and fully connected layers respectively: [Conv-(SBN)-ReLU-(Dropout)-(Max-pool)]M → [Affine-(BN)-ReLU-(Dropout)]N →  Affine →  Softmax. With such architecture, they tested the effect of different M and N in the model's classification with 3x3 filters, stride of 1, 32 epochs, and 128 batch size. As an attempt to improve their model, they concatenated a Histogram of Oriented Gradients (HOG) features to the last convolution layer because they are sensitive to edges (but it did not yield that much of an improvement). 
Facial Expression Recognition using CNN: Kaggle notebook that designed a CNN for the classification of the FER-2013 database with Keras. Their model preprocessed the with a Convert, Reshape, Normalize, One-hot encoding approach, and then had the following architecture: [Conv-(BN)-ReLU-Conv-(BN)-ReLU-(Max-pool)]3 → Flatten → [Dense-(BN)-Relu]3 → Softmax.
CNN-RNN: A Unified Framework for Multi-label Image Classification: Created an CNN-RNN framework for multi label classification problems. Essentially, "the CNN part extracts semantic representations from images while the RNN part models image/label relationship and label dependency".

Data: The main data set we intended to use is the FER-2013. It consists of 28000, 3500, and 3500 labelled images in the training set, development set, test set respectively. This dataset was created with Google Images for emotional detection based on an individual's faces, and has the following seven classification labels: happy, sad, angry, afraid, surprise, disgust, and neutral. In its entirety, the data is 302MB. 
 
In addition, we are considering using the AffectNet dataset, which contains more than 440k facial images collected on the Internet. These images were manually classified by the University of Denver with the following classification labels: happy, sad, anger, fear, surprise, disgust, neutral, contempt, none, and uncertain. As evident, these two datasets share 7 classification labels, so, using Data Science, we could potentially merge them after thorough preprocessing. The only issue with such a dataset is that access is limited and would require a request from our professor. In addition, it is 120GB, so we would need to apply for and use GCP credits in Google Cloud.
 
Methodology: For this project we are going to implement a Convolutional Neural Network (CNNs) to classify the different facial expressions to their corresponding emotions. We will use the Tensorflow library to train our model. Given the size of our data set, we might need more GPUs and we will apply for GCP credits to use Google’s Cloud. 
 
Convolutional Neural Network Potential Architecture
[Convolution-BatchNormalization-ReLU-MaxPool]3
Flatten
[Dense-BatchNormalization-ReLu]2
Softmax
In addition, we plan on using Tensorflow's Keras Tuner to achieve the highest accuracy possible without overfitting our model.
 
We think this design makes sense because since we are working with images a convolution is a good way to extract features from images. Once we have those descriptors we want to flatten them to a single array that can go through the fully connected layers where the neurons will learn what the features mean and assign them to the output label. 
 
That said, if our CCN model does not train appropriately or does not reach our target goals, we are considering designing a CNN-RNN Image Classification Model (as described in Related Work). Furthermore, we are considering comparing the results of our CNN to other machine learning algorithms such as Support Vector Machines (SVMs). 
 
For the second part of our project, we are thinking of using OpenCV’s Haar Classifier to identify the face within the image and then change its color taking advantage of the RGB matrix of the image.
 
Metrics: Since this is a classification problem the notion of “accuracy” does apply. We can measure the number of images that were correctly classified/total number of images passed into the network. We will also look at the confusion matrix between the different emotions and see the accuracy for each individual label. For the second part of the project where we will locate the face within the image and apply the corresponding color filter, we have not thought of any metrics for the accuracy because we will be using an OpenCV library. We, nonetheless, believe that the most important accuracy measure is within the CNN itself rather than the application of the filter. 
 
Our base goal is a 0.50 accuracy
Our target goal is a 0.65 accuracy
Our stretch goal is a 0.75 accuracy
 
Ethics: The broader social issue our project would tackle is the variety of mental disorders characterized by deficits in facial emotion recognition, including schizophrenia, autism, and alexithymia. People suffering from these disorders have deficits and biases in facial emotion recognition demonstrated to be linked to impairments in social and emotional function. These aggravate the development of other mental disorders and have a negative impact on treatment. This is due to the fact that humans are vision-driven animals, and often infer the psychological state of others as a function of their facial expressions. Furthermore, we will take into account the Color Blindness disorder and choose a color palette that adjusts to their capabilities.
 
The major stakeholders for our project are therefore individuals who have deficits and biases in facial emotion recognition. Mistakes in our algorithm classification would mean the wrong classification of emotions, which not only could be misleading for the user, but also lead to more social impairments for our major stakeholders. That is, if the model is an individual's main source of emotion detection and it misclassifies, the individual would interpret the other person's emotion incorrectly, which could exacerbate social impairments. Therefore, it is important that we make sure to create the best model possible so that we can have the best positive impact for our stakeholders with a tool that can actually help.
 
While we have yet to thoroughly analyze our data, we need to consider the historical and social biases it might contain, specifically the potential lack of ethnic and gender diversity in the facial images. We believe deep learning offers not only the best solution with highest accuracy potential for classifying and color labelling emotions (it outshines every other technique when it comes to solving complex problems such as image classification), but also gives us the ability to apply debiasing techniques if needed.
 
Division of labor:
Elena: Get the data ready for the model (download, clean, and structure). Work on measurement metrics like accuracy to assess the implemented model. 
Daniel: Implement the neural networks architecture.
Jimena: Implement face recognition using a CV library and assign colors to the emotions recognized.

