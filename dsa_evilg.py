#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[2]:


print("TensorFlow v" + tf.__version__)


# In[3]:


data = pd.read_csv('starcraft_player_data.csv')
print("Full train dataset shape is {}".format(data.shape))


# #### The data is composed of 20 columns and 3395 entries. We can see all 20 dimensions of the dataset by printing the first 5 entries using the following code:

# In[4]:


# display first 5 examples
data.head(5)


# In[5]:


data.info()
data.describe(include='all').T


# In[6]:


# separate numerical and categorical variables for easy analysis
cat_cols = data.select_dtypes(include=['object']).columns
num_cols = data.select_dtypes(include=np.number).columns.tolist()

print("Categorical values: ")
print(cat_cols)
print("Numerical values: ")
print(num_cols)


# In[7]:


# default performance = 0
performance_values = 0

# add the 'performance' column to data
data['PlayerPerformance'] = performance_values
data.to_csv('your_dataset.csv', index=False)

data.head()


# In[8]:


# separate features and target variable
categorical_cols = ['Age', 'HoursPerWeek', 'TotalHours']
numerical_cols = ['GameID', 'LeagueIndex', 'APM', 'SelectByHotkeys', 'AssignToHotkeys', 'UniqueHotkeys', 'MinimapAttacks', 'MinimapRightClicks', 'NumberOfPACs', 'GapBetweenPACs', 'ActionLatency', 'ActionsInPAC', 'TotalMapExplored', 'WorkersMade', 'UniqueUnitsMade', 'ComplexUnitsMade', 'ComplexAbilitiesUsed']
target_col = 'PlayerPerformance'

# convert categorical variables to encoding
data = pd.get_dummies(data, columns=categorical_cols)

# scale numerical variables
scaler = StandardScaler()
data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

# split the data into training and testing sets
x = data.drop(target_col, axis=1).values
y = data[target_col].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# In[9]:


model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, input_shape=(x_train.shape[1], 1)),
    tf.keras.layers.Dense(1)
])

model.compile(loss='mean_squared_error', optimizer='adam')


# In[10]:


x_train_rnn = np.expand_dims(x_train, axis=2)  # reshape input

model.fit(x_train_rnn, y_train, epochs=10, batch_size=32)


# In[11]:


x_test_rnn = np.expand_dims(x_test, axis=2)  # reshape input for RNN

predictions = model.predict(x_test_rnn)
mse = np.mean((predictions - y_test) ** 2)
print("Mean Squared Error:", mse)


# In[ ]:


# define RNN model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, input_shape=(x_train.shape[1], 1)),
    tf.keras.layers.Dense(1)
])

# plot the model
plot_model(model, to_file='model_rnn.png', show_shapes=True)


# #### Hypothetical: the model would be better if we had the rank of the players from previous years / competitions. The first test would train the neural map using the previous performance of the player and then predict their 2023 ranking, looking only one game into the future at a time, meaning the map would benefit from all real data up to the game in question. The graph would then show the predictions compared to the real data. The two graph lines shouldn't match up exactly because that would mean the RNN memorized the training data rather than create a prediction. 

# In[ ]:




