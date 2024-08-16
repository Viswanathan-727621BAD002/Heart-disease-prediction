from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Load and clean the dataset
db = pd.read_csv(r"C:\Users\Viswa\OneDrive - Dr.MCET\Desktop\heart disease prediction project\New folder\db.csv")

db.dropna(axis=0, inplace=True)

# Split the dataset
x = db.drop('TenYearCHD', axis=1)
y = db['TenYearCHD']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.06, random_state=42)

# Train the Logistic Regression model
model = LogisticRegression(max_iter=21200)
model.fit(x_train, y_train)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        new_data = [
            float(request.form['sex']),
            float(request.form['age']),
            float(request.form['education']),
            float(request.form['currentSmoker']),
            float(request.form['cigsPerDay']),
            float(request.form['bpMeds']),
            float(request.form['prevalentStroke']),
            float(request.form['prevalentHyp']),
            float(request.form['diabetes']),
            float(request.form['totChol']),
            float(request.form['sysBP']),
            float(request.form['diaBP']),
            float(request.form['imc']),
            float(request.form['heartRate']),
            float(request.form['glucose'])
        ]
        
        # Convert the input data to the correct shape
        info = np.array(new_data).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(info)
        
        if prediction == 0:
            result = "There is no risk of heart disease in the next 10 years."
        else:
            result = "There is a risk of heart disease in the next 10 years."
        
        return render_template('index.html', prediction=result)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
