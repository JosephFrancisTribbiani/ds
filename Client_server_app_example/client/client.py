import yaml
import requests
import pandas as pd


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


def main():
    # load the data
    data = pd.read_csv('data/client_test_data.csv', sep=';')
    json_data = data.to_json(orient='columns')

    conn_params = read_yaml(file_path='config.yaml', sec='client')
    conn_string = 'http://{}:{}/predict'.format(conn_params.get('host'), conn_params.get('port'))
    resp = requests.post(url=conn_string, json=json_data)

    if resp.status_code == 200:
        json_resp = resp.json()
        predictions = pd.DataFrame(data=json_resp)
        predictions.to_csv('data/predictions.csv', sep=';', index=False)
    return


if __name__ == '__main__':
    main()
