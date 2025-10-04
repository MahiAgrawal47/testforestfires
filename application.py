from flask import Flask,request,render_template,jsonify
import numpy as np
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app=application

ridge_model=pickle.load(open('models/ridge.pkl','rb'))
standard_scaler=pickle.load(open('models/scaler.pkl','rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST','GET'])
def predict_datapoint():
    if request.method=='POST':
        temperature = float(request.form['Temperature'])
        rh = float(request.form['RH'])
        ws = float(request.form['Ws'])
        rain = float(request.form['Rain'])
        ffmc = float(request.form['FFMC'])
        dmc = float(request.form['DMC'])
        dc = float(request.form['DC'])
        isi = float(request.form['ISI'])
        classes = int(request.form['classes'])
        region = int(request.form['region'])


        new_data_scaled = standard_scaler.transform([[temperature,rh,ws,rain,ffmc,dmc,dc,isi,classes,region]])
        result = ridge_model.predict(new_data_scaled)
        return render_template('home.html', results=result[0])

    else:
        return render_template('home.html')


if __name__=="__main__":
    app.run(debug =True ,host="0.0.0.0")    