# -----------------------------------------------------------
# File made following sololearn ML courses : 
#       https://www.sololearn.com/learning/1094
#
# File can be used as API to predict the Titanic passenger 
# chance of survive
#
# -----------------------------------------------------------

#Import library
import pandas as pd
from sklearn.linear_model import LogisticRegression
from flask import Flask, render_template, jsonify
import io

#Create server
app = Flask(__name__)

# -------------------------------------------------------------
# In this part, we create the model
# -------------------------------------------------------------

print("Creating ML model : ")

df = pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')

df['male'] = df['Sex'] == 'male'

X = df[['Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values
y = df['Survived'].values

model = LogisticRegression()

model.fit(X, y)

print("DONE !")
print("Model Score : ", model.score())

# -------------------------------------------------------------
# In this part, we set and configure API endpoint
# -------------------------------------------------------------

@app.route('/ML/Titanic/Score')
def Score():
    data = {
        'score' : model.score(X, y)
    }

    return jsonify(data)

@app.route('/ML/Titanic/Train/<number>')
def Train(number):
    print(number)

    data = {
        'prediction' : pd.Series(model.predict(X[slice(0,int(number))])).to_json(orient='values'),
        'reality' : pd.Series(y[slice(0,int(number))]).to_json(orient='values')
    }

    print(data)
    return(jsonify(data))

@app.route('/ML/Titanic/Predict/<Pclass>/<male>/<age>/<SisSpouse>/<ParentChild>/<Fare>')
def Predict(Pclass, male, age, SisSpouse, ParentChild, Fare):

    prediction = {
        'Pclass' : [Pclass], 
        'male' : [male], 
        'Age' : [age], 
        'Siblings/Spouses' : [SisSpouse], 
        'Parents/Children' : [ParentChild], 
        'Fare' : [Fare]
    }

    pred = pd.DataFrame(data=prediction) #Create pandas Dataframe from dictionnary

    data = {
        'prediction' : pd.Series(model.predict(pred)).to_json(orient='values')
    }   

    print(data)
    return(jsonify(data))

if __name__ == '__main__':
    app.run() 
