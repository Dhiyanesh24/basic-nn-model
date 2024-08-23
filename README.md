# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY
First we can take the dataset based on one input value and some mathematical calculus output value.Next define the neural network model in three layers.First layer has six neurons and second layer has four neurons,third layer has one neuron.The neural network model takes the input and produces the actual output using regression.

## Neural Network Model

![1](https://github.com/user-attachments/assets/8ded9576-c9ed-40cb-bd76-7cf2508c7411)
## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: Dhiyaneshwar P
### Register Number: 212222110009
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential  # Space was missing between "models" and "import"
from tensorflow.keras.layers import Dense
from google.colab import auth
import gspread
from google.auth import default

auth.authenticate_user()
creds,_ = default()
gc = gspread.authorize (creds)
worksheet = gc.open('Exp1-DL').sheet1
data = worksheet.get_all_values()

dataset1 = pd.DataFrame(data[1:], columns=data[0])
dataset1 = dataset1.astype({'X': 'float'})
dataset1 = dataset1.astype({'Y': 'float'})
dataset1.head()

X = dataset1[['X']].values
y = dataset1[['Y']].values
#X
X_train, X_test,y_train,y_test = train_test_split(X,y, test_size = 0.33, random_state = 33)
Scaler = MinMaxScaler()
Scaler.fit(X_train)

X_train1 = Scaler.transform(X_train)
ai_brain = Sequential ([
Dense (8, activation = 'relu'),
Dense (10, activation = 'relu'),
Dense (1)
])
ai_brain.compile(optimizer = 'rmsprop', loss = 'mse')
ai_brain.fit(X_train1, y_train, epochs =2000)

loss_df= pd.DataFrame(ai_brain.history.history)
loss_df.plot()

X_test1=Scaler.transform(X_test)
ai_brain.evaluate(X_test1, y_test)

X_n1=[[7]]
X_n1_1=Scaler.transform(X_n1)
ai_brain.predict(X_n1_1)
```
## Dataset Information
![2](https://github.com/user-attachments/assets/6e502ec7-c8c9-4872-96a8-fbad36b42071)

## OUTPUT

### Training Loss Vs Iteration Plot
![WhatsApp Image 2024-08-23 at 11 01 25_441d6fc3](https://github.com/user-attachments/assets/3c0bc337-5121-4e74-a1d7-d782519e1941)

### Test Data Root Mean Squared Error
![4](https://github.com/user-attachments/assets/7056208a-c44d-4b68-8953-3628d6d8c6d1)

### New Sample Data Prediction
![5](https://github.com/user-attachments/assets/bbd172cc-e1fd-4fd2-9dc6-42d51b3f2533)

## RESULT
Thus a Neural network for Regression model is Implemented.
