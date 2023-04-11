from flask import Flask,request,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
scalr = StandardScaler()
import pickle

# create flask app
app = Flask(__name__)
# laoding model
model = pickle.load(open('model.pkl','rb'))


@app.route("/")
def index():
    return render_template('index.html')
@app.route("/predict", methods=["POST"])
def predict():
    gender = int(request.form['gender'])
    age = int(request.form['age'])
    salary = int(request.form['salary'])

    input_data = pd.DataFrame({'Gender':[gender],'Age':[age],'Salary':[salary]})

    # scalling df
    scaled_data = scalr.fit_transform(input_data)
    # prediction
    prediction = model.predict(scaled_data)[0]

    if prediction == 1:
        output = "Purchased"
    else:
        output = "Not Purchased"

    return render_template('index.html',message = output)





# calling in python main
if __name__ == "__main__":
    app.run(debug=True)