import pickle
import pandas as pd

from flask import Flask, request, jsonify

app = Flask('market')

with open("./final_model.bin", "rb") as picklefile:
    model = pickle.load(picklefile)

@app.route('/predict', methods=['POST'])
def predict():
    client = request.get_json()
    df = pd.DataFrame(client, dtype=float)
    pred = model.predict(df)
    result = {
        'responder_6': pred[0]
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
