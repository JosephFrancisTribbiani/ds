import yaml


def read_yaml(file_path: str) -> dict:
    """
    Функция для считывания config файла.
    :param file_path: путь к config файлу.
    :return: параметры из config файла.
    """
    with open(file_path, 'r') as f:
        try:
            return yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)
