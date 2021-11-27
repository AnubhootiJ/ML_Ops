from flask import Flask
from flask import request

import numpy as np
from joblib import load

#svm_model_path = "D:\IITJ\Semester-3\MLOps_HandsON\ML_Ops\models\model_0.01.joblib"
#dec_model_path = "D:\IITJ\Semester-3\MLOps_HandsON\ML_Ops\models\decisionTree\dec_32_(0.2, 0.1).joblib"

svm_model_path = "model_0.01.joblib"
dec_model_path = "dec_32_(0.2,0.1).joblib"

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

@app.route('/svm_predict', methods=['POST'])
def svm_predict():
    clf = load_model(svm_model_path)
    input_json = request.json
    image = input_json['image']
    #print(image)
    image = np.array(image).reshape(1,-1)
    predicted = clf.predict(image)
    return "Prediction = " + str(predicted[0])

@app.route('/dt_predict', methods=['POST'])
def dt_predict():
    clf = load_model(dec_model_path)
    input_json = request.json
    image = input_json['image']
    #print(image)
    image = np.array(image).reshape(1,-1)
    predicted = clf.predict(image)
    return "Prediction = " + str(predicted[0])


#if __name__ == '__main__':
#    app.run(debug=True)