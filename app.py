from flask import Flask, request, app,render_template
from flask import Response
import pickle
import numpy as np
import pandas as pd


app= Flask(__name__)


scaler=pickle.load(open("/config/workspace/model/StandardScaler.pkl", "rb"))
model = pickle.load(open("/config/workspace/model/modelprediction.pkl", "rb"))

## Route for homepage

@app.route('/')
def index():
    return render_template('index.html')

## Route for Single data point prediction
@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    

    if request.method=='POST':

        Item_Weight=float(request.form.get("Item_Weight"))
        Item_Fat_Content = int(request.form.get('Item_Fat_Content'))
        Item_Visibility = float(request.form.get('Item_Visibility'))
        Item_Type = int(request.form.get('Item_Type'))
        Item_MRP = float(request.form.get('Item_MRP'))
        Outlet_Establishment_Year = int(request.form.get('Outlet_Establishment_Year'))
        Outlet_Size = int(request.form.get('Outlet_Size'))
        Outlet_Location_Type = int(request.form.get('Outlet_Location_Type'))
        Outlet_Type = int(request.form.get('Outlet_Type'))

        new_data=scaler.transform([[Item_Weight,Item_Fat_Content,Item_Visibility,Item_Type,Item_MRP,Outlet_Establishment_Year,Outlet_Size,Outlet_Location_Type,Outlet_Type]])
        predict=model.predict(new_data)
       
        
            
        return render_template('single_prediction.html',result=predict[0])

    else:
        return render_template('home.html')


if __name__=="__main__":
    app.run(host="0.0.0.0")