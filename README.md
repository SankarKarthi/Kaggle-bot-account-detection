**Kaggle Bot Detection**


**Overview**

The Kaggle Bot Detection project aims to classify whether a user is a bot or not based on various features extracted from user data. This project uses a neural network model built with Keras and TensorFlow, and a web interface developed using Streamlit to make predictions based on user inputs.

**Features**

Data Preprocessing: Clean and preprocess user data.
Neural Network Model: Train a neural network to classify users as bots or non-bots.
Web Interface: User-friendly interface to input user data and get predictions.
Installation
Prerequisites
Python 3.7 or higher
pip (Python package installer)
Virtual environment (optional but recommended)

**Usage**

Run the Streamlit App:
bash

streamlit run app.py

**Input Features:**

Open the Streamlit app in your browser.
Use the sidebar to input the features:
Is Google Login?
FOLLOWER_COUNT
FOLLOWING_COUNT
AVG_NB_READ_TIME_MIN

**Get Prediction:**

Click on the "Submit" button to get the prediction.
The app will display whether the user is predicted to be a bot or not.
Data Preprocessing

**The preprocessing steps include:**

Dropping unnecessary columns.
Handling missing values.
Converting Boolean values to binary.
Converting categorical variables to dummy variables.
Model Training

**The neural network model is defined and trained with the following architecture:**

Input layer with 25 neurons.
Hidden layers with 12 and 6 neurons, respectively.
Output layer with 2 neurons (for binary classification).
The model is trained using the Adam optimizer and binary cross-entropy loss function.
Technologies Used
Data Processing: pandas, numpy
Machine Learning: scikit-learn, Keras, TensorFlow
Web Framework: Streamlit

