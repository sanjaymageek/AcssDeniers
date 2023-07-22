from flask import Flask,jsonify,request
from flask_cors import CORS
import numpy as np
import pickle
from flask_cors import CORS,cross_origin

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['POST'])
@cross_origin()
def fun():

    product = request.json
    text = product['description']
    loaded_vec = pickle.load(open('vec.sav', 'rb'))
    dt = np.array([text])
    p=loaded_vec.transform(dt).toarray()
    loaded_model = pickle.load(open('model.sav', 'rb'))
    return str(loaded_model.predict(p)[0])  

    

if __name__=='__main__':
    app.run(port=5000,debug=True)
