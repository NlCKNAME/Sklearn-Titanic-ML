# Sklearn-Titanic-ML
Simple machine learning model to predict the chance of survive of Titanic's passengers 

This repositorie contain :
 - Initial file from Sololearn course : main.py
 - File to use it as API : app.py

### Update
Created API based on initial code
When the server is started, it run as http://localhost:5000

The possible endpoint are :
- `/ML/Titanic/Score/` : Get the model score
- `/ML/Titanic/Train/<number>` : Train model with first n-number data from initial dataset
- `/ML/Titanic/Predict/<Pclass>/<male>/<age>/<SisSpouse>/<ParentChild>/<Fare>` : Predict the output for specific passenger

### To Do
- Add error handler on API
- System to save/load ML model
- Generate chart with Matplotlib
