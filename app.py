from flask import Flask,request,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from Pipeline.predict_pipeline import CustomData,PredictPipeline

app=Flask(__name__)

@app.route('/')
def index():
    return render_template('predict.html')
#'gender', 'smoker', 'region', 'medical_history', 'family_medical_history', 'exercise_frequency', 'occupation', 'coverage_level'
@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('predict.html')
    else:
        
        data=CustomData(
            gender=request.form.get('gender'),
            smoker=request.form.get('smoker'),
            region=request.form.get('region'),
            medical_history=request.form.get('medical_history'),
            family_medical_history=request.form.get('family_medical_history'),
            exercise_frequency=request.form.get('exercise_frequency'),
            occupation=request.form.get('occupation'),
            coverage_level=request.form.get('coverage_level'),
            age=request.form.get('age'),
            bmi=request.form.get('bmi'),
            children=request.form.get('children')
        )
        pred_df=data.get_data_as_dataframe()
        print(pred_df)
        predict_pipeline=PredictPipeline()
        results=predict_pipeline.predict(pred_df)
        return render_template('predict.html',results=format(np.round(results[0][0],2),","))
    
if __name__=='__main__':
    app.run(host='0.0.0.0',port=8080)
