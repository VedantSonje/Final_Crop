import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index_final.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    df=pd.read_csv('cropdata.csv')
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    #prediction = model.predict(final_features)

    s1=set()
    for i in range(len(df['label'])):
        s1.add(df['label'][i])
    list1=list(s1)
    list1.sort()
    a1=model.predict_proba(final_features)
    l1=[]
    for j in range(len(a1)):
        l2=[]
        for i in range(len(a1[j])):
            if(a1[j][i]>0.2):
                l2.append(list1[i])
        l1.append(l2)

    output = l1
    print(output[0])
    return render_template('index_final.html', prediction_text='Recommended Crop According to season : {}'.format(str(output[0])))


if __name__ == "__main__":
    app.run(debug=True)