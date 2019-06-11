
#importing libraries
import os
import numpy as np
import flask
import pickle
from flask import Flask, render_template, request

#creating instance of the class
app=Flask(__name__)

#to tell flask what url shoud trigger the function index()


@app.before_first_request
def load_global_data():
    global dp, nb_model, lr_model
    dp = pickle.load(open("dataprocessor.pkl", "rb"))
    nb_model = pickle.load(open("NBModel.pkl", "rb"))
    lr_model = pickle.load(open("LRModel.pkl", "rb"))

@app.route('/')

@app.route('/index')
def index():
    return flask.render_template('index.html')

#prediction function
def predict_input_data(predict_list):


    if '(50 words or less)' in predict_list or predict_list[-1].strip() == "":

        return "Please enter a valid news headline!"

    vect = dp.input_vectorize([str(predict_list[-1])])
    
    if 'NB' in predict_list and 'LR' in predict_list:
        nb_predict = dp.decode_label(nb_model.predict(vect) )
        lr_predict = dp.decode_label(lr_model.predict(vect) )

        return "NB: " + str(nb_predict) + "\nLR: " + str(lr_predict) 
        
    elif 'NB' in predict_list:
        return "NB: " + str(dp.decode_label(nb_model.predict(vect)))

    elif 'LR' in predict_list:
        return "LR: " + str(dp.decode_label(lr_model.predict(vect) ) )
    
    else:
        return "Please select a model!"

# model info function
def get_model_info(model):

    ret = None
    if model == 'NB':
        ret = ["ML Model: Multinomial Naive Bayes", "Accuracy: " + str(round(nb_model.get_model_accuracy(), 2)), "Trained using dataset with 187068 records"]
    else:
        ret = ["ML Model: Logistic Regression", "Accuracy: " + str(round(lr_model.get_model_accuracy(), 2)),  "Trained using dataset with 187068 records"]

    return ret
        
@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        req_list = request.form.to_dict()
        req_list=list(req_list.values())
        req_list = list(map(str, req_list))
        
        print(req_list)

        if "NB info" in req_list:
            result = get_model_info('NB')
            return render_template("modelinfo.html", first=result[0], second=result[1], third=result[2])
        
        elif "LR info" in req_list:
            result = get_model_info('LR')
            return render_template("modelinfo.html", first=result[0], second=result[1], third=result[2])
        
        result = predict_input_data(req_list)
            
        return render_template("result.html", prediction=result)






