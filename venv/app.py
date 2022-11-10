from flask import Flask, jsonify, request, render_template 
import json

app = Flask(__name__)

@app.route('/')
@app.route('/landing_page')
def index():
    return render_template("landing_page.html")


@app.route('/information_page')
def get_symptoms():
    return render_template("information_page.html")

@app.route('/filling_page')
def filling_page():
    return render_template("filling_page.html")


@app.route('/filling_page', methods=['POST', 'GET'])
def get_values():
    symptom0 = request.form['symptom0']
    symptom1 = request.form['symptom1']
    symptom2 = request.form['symptom2']
    symptom3 = request.form['symptom3']
    symptom4 = request.form['symptom4']
    symptom5 = request.form['symptom5']
    symptom6 = request.form['symptom6']
    symptom7 = request.form['symptom7']
    symptom8 = request.form['symptom8']
    symptom9 = request.form['symptom9']
    symptom10 = request.form['symptom10']
    symptom11 = request.form['symptom11']
    symptom12 = request.form['symptom12']
    symptom13 = request.form['symptom13']
    symptom14 = request.form['symptom14']
    symptom15 = request.form['symptom15']
    symptom16 = request.form['symptom16']
    symptom17 = request.form['symptom17']
    symptom18 = request.form['symptom18']
    symptom19 = request.form['symptom19']

    updated_data = [symptom0, symptom1, symptom2, symptom3, symptom4, 
                    symptom5, symptom6, symptom7, symptom8, symptom9,
                    symptom10, symptom11, symptom12, symptom13, symptom14, 
                    symptom15, symptom16, symptom17, symptom18, symptom19]

    with open('user_data.json', 'w') as f:
            json.dump(updated_data, f)

    import numpy as np

    with open('user_data.json', 'r') as f:
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
    return render_template("results_page.html", result=result)

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

@app.route('/results_page')
def result():
    return render_template("results_page.html")


if __name__ == '__main__':
    app.run(debug=True ,host = '0.0.0.0',port = 5000)