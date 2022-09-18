# -----------------------------------------------------------
# File made following sololearn ML courses : 
#       https://www.sololearn.com/learning/1094
#
# Idea of features to add :
#      -  Create graph with matplotlib
#      -  Create REST API with Flask
#      -  Link the API with discord.js to create prediction bot
#      -  Try to do this but with weather data or cars
#
# -----------------------------------------------------------

import pandas as pd #Import pandas library
from sklearn.linear_model import LogisticRegression #Import logistic regression from sklearn library

df = pd.read_csv('https://sololearn.com/uploads/files/titanic.csv') #Get dataset



# -------------------------------------------------------------
# In this part, we gonna prepare the date to be used by sklearn
# -------------------------------------------------------------

df['male'] = df['Sex'] == 'male' #Convert sexe feature to boolean type

X = df[['Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values #Select our features

y = df['Survived'].values #Select target features

# print(X)
# print(y)



# -------------------------------------------------------------
# In this part, we gonna build our ML model
# -------------------------------------------------------------

model = LogisticRegression() #Initialize the class

model.fit(X,y) #We train our model with previous data -> .fit(Input_features, Ouput_expected)

print(model.coef_, model.intercept_) #Print model's mathematical equation



# -------------------------------------------------------------
# In this part, we gonna try our ML model
# -------------------------------------------------------------

model.predict(X) #Make a prediction

print("Prediction : ", model.predict(X[:5])) #We try to predict five first row of data from our datase

print("Expected : ", y[:5]) #We print the expected output



# -------------------------------------------------------------
# In this part, we gonna evaluate our model
# -------------------------------------------------------------

y_pred = model.predict(X) #We try to predict all the dataset

print((y == y_pred).sum()) #Print the number of correct prediction
print((y == y_pred).sum() / y.shape[0]) #Print the accuracy of the model -> number of correct prediction / number of data row

print(model.score()) #Print the accuracy directly with sklearn prebuilt function