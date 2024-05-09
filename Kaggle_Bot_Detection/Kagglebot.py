import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

# Load the preprocessed DataFrame
df = pd.read_csv("C:/Users/sanka/Downloads/archive (18)/kaggle_bot_accounts.csv")
df = df.drop(columns=['Unnamed: 0', 'NAME', 'GENDER', 'EMAIL_ID', 'REGISTRATION_IPV4', 'TOTAL_VOTES_GAVE_NB', 'TOTAL_VOTES_GAVE_DS', 'TOTAL_VOTES_GAVE_DC', 'REGISTRATION_LOCATION', 'DATASET_COUNT', 'CODE_COUNT', 'DISCUSSION_COUNT'])
df = df.dropna()

# Converting Boolean values to binary
df['IS_GLOGIN'] = df['IS_GLOGIN'].astype(int)
df['ISBOT'] = df['ISBOT'].astype(int)

# Converting gender to dummy
df = pd.get_dummies(df)
df.reset_index(drop=True)

# Creating X and y
y = df['ISBOT']
X = df.drop(columns='ISBOT')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# One-hot encode the target variables
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Define the neural network model
model = Sequential()
model.add(Dense(25, activation='relu', input_dim=X_train.shape[1]))  # Use X_train.shape[1] as input_dim
model.add(Dense(12, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(2, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=5, batch_size=500)

# Streamlit App
st.title("Bot Detection App")

# Sidebar with input features
st.sidebar.header("Input Features")
is_glogin = st.sidebar.checkbox("Is Google Login?")
follower_count = st.sidebar.number_input("FOLLOWER_COUNT", min_value=0)
following_count = st.sidebar.number_input("FOLLOWING_COUNT", min_value=0)
avg_nb_read_time_min = st.sidebar.number_input("AVG_NB_READ_TIME_MIN", min_value=0)

# Create user data array with all features
user_data = np.array([[is_glogin, follower_count, following_count, avg_nb_read_time_min]])

# Add a submit buttonf
if st.sidebar.button("Submit"):
    # Make prediction
    prediction = model.predict(user_data)
    # Display the result
    st.subheader("Prediction Result")
    if prediction[0][1] > 0.5:
        st.write("The user is predicted to be a bot.")
    else:
        st.write("The user is predicted not to be a bot.")
