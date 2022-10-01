from xcontest import get_flights, Takeoff
from dotenv import load_dotenv


def main():
    # итерируемся по местам взлета
    for takeoff in Takeoff:
        # и выгружаем информацию по полетам
        for _ in get_flights(takeoff=takeoff):
            continue
    return


if __name__ == '__main__':
    # загружаем переменные окружения для подключения к xcontest из файла .env
    load_dotenv('.env')
    main()
