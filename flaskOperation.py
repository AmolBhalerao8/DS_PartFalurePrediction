import joblib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from flask import Flask, jsonify, request
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


app = Flask(__name__)

#Predict the parts that may fail according to the location, Part_ID and age:
@app.route('/train_model')
def train():
    data = joblib.load('LRtrain.pkl')
    x = data[['Part_ID', 'City', 'days']]
    y = data['repair']
    logm = LogisticRegression()
    logm.fit(x, y)
    joblib.dump(logm, 'train_data.pkl')
    return "model trained successfully"


@app.route('/test_model', methods = ['POST'])
def test():
    pkl_file = joblib.load('train_data.pkl')
    test_data = request.get_json()
    f1 = test_data['Part_ID']
    if int(test_data['Part_ID']) in range(901, 912):
        f1 = test_data['Part_ID']
    else:
        return 'Pls enter correct Part_ID'


    f2 = test_data['City']
    f3 = test_data['days']

    if test_data['City'] == 'ahemadabad':
        f2 = 0
    elif test_data['City'] == 'banglore':
        f2 = 1
    elif test_data['City'] == 'cuttack':
        f2 = 2
    elif test_data['City'] == 'mumbai':
        f2 = 3

    elif test_data['City'] == 'noida':
        f2 = 4
    elif test_data['City'] == 'panaji':
        f2 = 5
    else :
        return "Your city is not listed"


    my_test_data = [f1, f2, f3]
    my_data_array = np.array(my_test_data)
    test_array = my_data_array.reshape(1, 3)
    df_test = pd.DataFrame(test_array, columns= ['Part_ID', 'City', 'days'])

    y_pred = pkl_file.predict(df_test)
    if y_pred == 1:
        return 'Yes part may fail'
    else:
        return 'part may not fail'

#To determine whether the defective part is to be replaced or repaired
@app.route('/DTtrain_model')
def DTtrain():
    data = joblib.load('LRtrain.pkl')
    data1 = joblib.load('df_feat.pkl')
    x = data[['City', 'days']]
    y = data['repair']
    dt = DecisionTreeClassifier()
    dt.fit(x, y)
    joblib.dump(dt, 'DTtrain.pkl')
    return 'model trained successfully'


@app.route('/DTtest_model', methods = ['POST'])
def DTtest():
    pkl_file1 = joblib.load('DTtrain.pkl')
    test_data = request.get_json()
    #f1 = test_data['Part_ID']
    #if int(test_data['Part_ID']) in range(901, 912):
        #f1 = test_data['Part_ID']
    #else:
     #   return 'Pls enter correct Part_ID'

    f2 = test_data['City']
    f3 = test_data['days']

    if test_data['City'] == 'ahemadabad':
        f2 = 0
    elif test_data['City'] == 'banglore':
        f2 = 1
    elif test_data['City'] == 'cuttack':
        f2 = 2
    elif test_data['City'] == 'mumbai':
        f2 = 3

    elif test_data['City'] == 'noida':
        f2 = 4
    elif test_data['City'] == 'panaji':
        f2 = 5
    else:
        return "Your city is not listed"


    my_test_data = [f2, f3]
    my_data_array = np.array(my_test_data)
    test_array = my_data_array.reshape(1, 2)
    df_test = pd.DataFrame(test_array, columns=['City', 'days'])

    y_pred1 = pkl_file1.predict(df_test)

    if y_pred1 == 0:
        return 'Part should be Repaired'
    else:
        return 'Part should be Replaced'





app.run(port = 6000)