import json
import numpy as np

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

with open('user_symptoms.json', 'r') as f:
    data = json.load(f)

data = np.array(data)
data = data.reshape(1, -1)

x = [0.79615774, 0.28199183, 0.08259212, 0.09090909, 0.02697495,
       0.03376206, 0.02713116, 0.06636845, 0.0566968 , 0.08903051,
       0.03147897, 0.05214794, 0.08896041, 0.68184631, 0.65826063,
       0.71690718, 0.24245376, 0.58235465, 0.2511228 , 0.1543442 ]

x = np.array(x)
x = x.reshape(1, -1)

model_list = load_saved_model_from_db('first_model', 'mongodb://localhost:27017/',
    'Parkinson_Prediction', 'knn_model')

model = model_list[0]
scaler = model_list[2]

data = scaler.transform(x)
result = model.predict(x)
if result == 1:
    result = 'Positive'
else:
    result = 'Negative'
print(result)