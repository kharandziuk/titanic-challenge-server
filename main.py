from flask import Flask
from flask import jsonify
from flask import request
from utils import predict, lookup_func

import csv

app = Flask(__name__)

@app.route('/survive')
def hello_world():
    name = request.args.get('name')
    matched_name, data = lookup_func(name)
    survival_probability = predict(data)
    return jsonify(
        dict(
            name=matched_name,
            probability=survival_probability)
    )
