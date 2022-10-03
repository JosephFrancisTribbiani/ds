import yaml
import pickle
import pandas as pd
from flask import Flask, request, jsonify
from flask_restful import Resource, Api


app = Flask(__name__)
api = Api(app)


class Health(Resource):
    def get(self):
        return {'health_status': 'running'}, 200


class Predict(Resource):
    def post(self):
        # parse json data
        json_data = request.get_json()
        data = pd.read_json(json_data, orient='columns')

        # loading the model
        with open('model/regression_model.pkl', 'rb') as pf:
            model = pickle.load(pf)

        y_hat = model.predict(data)
        return jsonify(y_hat=y_hat.tolist())


api.add_resource(Health, '/health')
api.add_resource(Predict, '/predict')


def read_yaml(file_path: str, sec: str = 'app') -> dict:
    """
    Функция для считывания config файла.
    :param file_path: путь к config файлу.
    :param sec: раздел в config файле, defaults to 'app'.
    :return: параметры из config файла.
    """
    with open(file_path, 'r') as f:
        try:
            return yaml.safe_load(f).get(sec)
        except yaml.YAMLError as exc:
            print(exc)


if __name__ == '__main__':
    conn_params = read_yaml(file_path='config.yaml')
    app.run(host=conn_params.get('host', 'localhost'), port=conn_params.get('port', 9000),
            debug=conn_params.get('debug', True))
