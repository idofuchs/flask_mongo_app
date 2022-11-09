from flask import Flask, jsonify, request, render_template 
import json

symptoms = {
    'MDVP:Fo(Hz)':None,
    'MDVP:Flo(Hz)':None,
    'MDVP:Jitter(%)':None,
    'MDVP:Jitter(Abs)':None,
    'MDVP:RAP':None,
    'MDVP:PPQ':None,
    'Jitter:DDP':None,
    'MDVP:Shimmer':None,
    'MDVP:Shimmer(dB)':None,
    'Shimmer:APQ3':None,
    'Shimmer:APQ5':None,
    'MDVP:APQ':None,
    'Shimmer:DDA':None,
    'HNR':None,
    'RPDE':None,
    'DFA':None,
    'spread1':None,
    'spread2':None,
    'D2':None,
    'PPE':None
}

symptoms_list = list(symptoms.keys())

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("home.html")


@app.route('/symptoms')
def get_symptoms():
    return render_template("symptoms.html")


@app.route('/fillingform', methods=['POST'])
def get_values():
 
    if request.method == 'POST':
        updated_data = []
        for i in range(20):
            if request.form['symptom'+str(i)] =="":
                updated_data.append(0)
            else:
             updated_data.append(float(request.form['symptom'+str(i)]))
        with open('user_symptoms.json', 'w') as f:
            json.dump(updated_data, f)
        return result()
    else:
        return render_template("fillingform.html")


def load_saved_model_from_db(model_name, client, db, dbconnection):
    import pickle
    import pymongo
    json_data = {}
    #saving model to mongoDB
    # creating connection
    myclient = pymongo.MongoClient(client)
    
    #creating database in mongodb
    mydb = myclient[db]
    
    #creating collection
    mycon = mydb[dbconnection]
    data = mycon.find({'name': model_name})
    
    for i in data:
        json_data = i
    #fetching model from db
    pickled_model = json_data[model_name]
    pickled_scaler = json_data['scaler']

    model = pickle.loads(pickled_model)
    score = json_data['score']
    scaler = pickle.loads(pickled_scaler)

    model_list = [model, score, scaler]
    return model_list

@app.route('/results')
def result():
    import numpy as np

    with open('user_symptoms.json', 'r') as f:
        data = json.load(f)
    
    data = np.array(data)
    data = data.reshape(1, -1)
    
    model_list = load_saved_model_from_db('first_model', 'mongodb://localhost:27017/',
     'Parkinson_Prediction', 'knn_model')

    model = model_list[0]
    scaler = model_list[2]

    data = scaler.transform(data)
    result = model.predict(data)
    if result == 1:
        result = 'Positive'
    else:
        result = 'Negative'
    return render_template("results.html", result=result)

if __name__ == '__main__':
    app.run(debug=True ,host = '0.0.0.0',port = 5000)