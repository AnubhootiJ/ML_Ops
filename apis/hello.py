"""
from flask import Flask

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!, It's working</p>"

"""
from flask import Flask
from flask import request
#from flask_restx import Resource, Api

import numpy as np
#from example.utils import load_model
from joblib import load

best_model_path = "D:\IITJ\Semester-3\MLOps_HandsON\ML_Ops\models\model_0.01.joblib"

app = Flask(__name__)
#api = Api(app)

def load_model(path):
    print("laoding Model")
    clf = load(path)
    print("Model Loaded")
    return clf

@app.route('/hello')
class HelloWorld():
    def get(self):
        return {'hello': 'world'}

@app.route('/predict', methods=['POST'])
def predict():
    clf = load_model(best_model_path)
    input_json = request.json
    image = input_json['image']
    print(image)
    image = np.array(image).reshape(1,-1)
    predicted = clf.predict(image)
    return "Prediction = " + str(predicted[0])


if __name__ == '__main__':
    app.run(debug=True)