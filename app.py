from flask import Flask,request,jsonify ,json
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder




model = pickle.load(open('model.pkl','rb'))

le = LabelEncoder()
app = Flask(__name__)
@app.route('/')
def home():
    return "Hello world"

@app.route('/predict',methods = ['post'])
def predict():
    symp1 = request.form.get("Symptom_1")
    symp2 = request.form.get("Symptom_2")
    symp3 = request.form.get("Symptom_3")

    input_query = np.array([[symp1,symp2,symp3]])
    result= model.predict(input_query)[0]
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)