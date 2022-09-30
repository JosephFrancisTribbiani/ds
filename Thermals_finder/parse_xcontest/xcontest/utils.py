import os
import re
import time
import json
import logging
import requests
import urllib.parse as urlparse
from functools import wraps
from dotenv import load_dotenv
from bs4 import BeautifulSoup, SoupStrainer
from requests.adapters import Retry, HTTPAdapter


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
logger = logging.getLogger(__name__)


load_dotenv()
__all__ = ['get_flights']


def _xcontest_login(func):
    """
    Обертка для аутентификации на xcontest.
    """
    @wraps(func)
    def inner(*args, **kwargs):
        with requests.Session() as session:
            username, password = os.getenv('APP_XCONTEST_USERNAME'), os.getenv('APP_XCONTEST_PASSWORD')
            base_url = os.getenv('BASE_URL')
            auth_data = {"login[username]": username, "login[password]": password, "login[persist_login]": "Y"}

            # настройка попыток соединения с сервером
            # total - количество попыток
            # backoff_factor - коээфициент при расчете времени "заморозки" перед следующей попыткой. Если равен 1, то на
            # первой попытке ожидание равно 1 сек., на второй - 2 сек., далее - 4, 8, 16 и т.д. Формула расчета:
            # {backoff factor} * (2 ** ({number of retry} - 1))
            # если method_whitelist=False, то применяем retries для всех типов запросов (GET, POST и т.д.)
            # status_forcelist - статусы, в случае которых необходимо повторить запрос
            retries = Retry(
                total=15, backoff_factor=30, method_whitelist=False,
                status_forcelist=[429, 450, 500, 502, 503, 504])
            retries.DEFAULT_BACKOFF_MAX = 14400
            session.mount('https://', HTTPAdapter(max_retries=retries))
            session.post(url=base_url, data=auth_data)

            res = func(*args, session=session, **kwargs)
        return res
    return inner


def _set_params(latitude: float, longitude: float, radius: int = 5000, mode: str = 'START', date_mode: str = 'dmy',
                date: str = None, value_mode: str = 'dst', min_value_dst: str = None, catg: str = None,
                route_types: str = None, avg: str = None, pilot: str = None, sort_by: str = 'time_start',
                direction: str = 'down', offset: int = 0) -> str:
    point = "{} {}".format(latitude, longitude)

    # check the input
    assert re.fullmatch(re.compile(r'\d{1,2}(?:\.\d{1,5})?\s\d{1,2}(?:\.\d{1,5})?'), point), 'Wrong point format'
    assert isinstance(radius, int) & (radius > 0), 'Wrong radius'
    assert catg in [None, 'FAI3', 'FAI3-A', 'FAI3-B', 'FAI3-C', 'FAI3-T', 'FAI125', 'FAI1', 'FAI2', 'FAI3'], \
        'Wrong catg'
    assert catg in [None, 'VP', 'PC', 'FT', 'PT-FT'], 'Wrong route_types'
    assert sort_by in [None, 'time_start', 'pilot', 'launch', 'route', 'dist', 'pts', 'pk'], 'Wrong sort_by'
    assert direction in [None, 'up', 'down'], 'Wrong sort direction'

    params = {
        'filter[point]':         point,
        'filter[radius]':        radius,
        'filter[mode]':          mode,
        'filter[date_mode]':     date_mode,
        'filter[date]':          date,
        'filter[value_mode]':    value_mode,
        'filter[min_value_dst]': min_value_dst,
        'filter[catg]':          catg,
        'filter[route_types]':   route_types,
        'filter[avg]':           avg,
        'filter[pilot]':         pilot,
        'list[sort]':            sort_by,
        'list[dir]':             direction,
        'list[start]':           offset
    }

    params = {k: '' if v is None else v for k, v in params.items()}
    base_url = os.getenv('BASE_URL')
    url = base_url + 'flights-search/?' + urlparse.urlencode(params, safe='?=.&')
    return url


def get_flights(takeoff, freeze: int = 2, data_loc: str = 'data'):
    for page_idx, page in enumerate(_download_pages(takeoff=takeoff, freeze=freeze)):
        logging.info('Iterate over flights on page [%d].', page_idx + 1)
        if page_idx != 0 and page_idx < 10:
            continue
        for flight_idx, flight in enumerate(_parse_page(page=page)):
            logging.info('Extracting data for flight [%d]', flight_idx + 1)
            _extract_flight_data(flight_row=flight, data_loc=data_loc)
            time.sleep(freeze)
            yield flight


@_xcontest_login
def _download_pages(session, takeoff, freeze: int = 2):
    """
    Получение всех полетов для данного take off.

    :param takeoff:
    :param session:
    :param freeze:
    :return:
    """
    params = takeoff.value
    offset, has_next = 0, True

    while has_next:
        url = _set_params(**params, offset=offset)
        logging.info('Getting from the page: %s', url)
        r = session.get(url=url)
        if r.status_code == 200:
            page = r.text
            yield page
            if has_next := _has_next_page(page=page, curr_offset=offset):
                logging.info('Next page exists. Go to next page.')
                offset += 50
                time.sleep(freeze)


def _has_next_page(page: str, curr_offset: int) -> bool:
    """
    Выясняем, есть ли сраницы с полетами, помимо текущей.

    :return:
    """
    logging.info('Check if next page exists.')
    soup = BeautifulSoup(page, "lxml", parse_only=SoupStrainer("div", {"class": "paging"}))

    paging_items = soup.select_one(".paging")
    if not paging_items:
        return False

    last_page_url = paging_items.select_one('a[title="last page"]')['href']
    next_offset = int(re.search(pattern=r'(?<=list\[start\]\=)\d+', string=last_page_url).group(0))
    return curr_offset < next_offset


def _parse_page(page: str):
    soup = BeautifulSoup(page, "lxml", parse_only=SoupStrainer("table", {"class": "flights"}))
    for row in soup.select(".flights tbody tr"):
        yield row


@_xcontest_login
def _extract_flight_data(session, flight_row: BeautifulSoup, data_loc: str = 'data') -> None:
    detail_page_url = urlparse.urljoin('https://www.xcontest.org', flight_row.select_one('a[class="detail"]')['href'])

    logging.info("Go to flight's details page: %s", detail_page_url)
    r = session.get(url=detail_page_url)

    if r.status_code == 200:
        logging.info("Generate url to get flight metadata.")
        detail_page_soup = BeautifulSoup(r.text, "lxml")
        script_tags = detail_page_soup.select('div[class="under-bar"] > script')
        try:
            attributes = [attr.strip() for attr in script_tags[2].string.split(';')]
            year = re.search(
                pattern=re.compile(r'(?<=volume)(?:\D*)(\d+)'), string=attributes[0]).group(1)
            flight_id = re.search(
                pattern=re.compile(r'(?<=addFlight\()(\d+)'), string=attributes[1]).group(1)
            year, flight_id = int(year), int(flight_id)
            key = re.search(pattern=re.compile(r'(?<=key\=)([\d\w-]+)'), string=script_tags[3]['src']).group(0)
            flight_metadata_url = f"https://www.xcontest.org/api/data/?flights3/world/" \
                                  f"{year}:{flight_id}&lng=en&key={key}"
        except:
            logging.warning("FAILED.")
            return

        logging.info('Getting flight metadata.')
        r = session.get(flight_metadata_url)
        if r.status_code == 200:
            logging.info('Parse JSON with flight metadata.')
            try:
                json_data = json.loads(r.text)
                json_path = f'{data_loc}/json'

                if not os.path.isdir(json_path):
                    os.makedirs(json_path)

                json_file_path = json_path + f'/{year}-{flight_id}.json'
                if not os.path.isfile(json_file_path):
                    with open(json_file_path, 'w') as json_file:
                        json.dump(json_data, json_file)
                else:
                    logging.warning("%s already exists.", json_file_path)

                logging.info('Downloading IGC file')
                igc_link = json_data.get('igc').get('link')
                if igc_link:
                    r = session.get(igc_link, headers={'Referer': detail_page_url}, stream=True)

                    igc_path = f'{data_loc}/igc'

                    if not os.path.isdir(igc_path):
                        os.makedirs(igc_path)

                    igc_file_path = igc_path + f'/{year}-{flight_id}.igc'

                    if not os.path.isfile(igc_file_path):
                        with open(igc_file_path, 'w') as igc_file:
                            igc_file.write(r.text)
                    else:
                        logging.warning("%s already exists.", igc_file_path)

                else:
                    logging.warning("FAILED. IGC link is not founded.")

            except:
                logging.warning("FAILED. JSON parsing failed with an error.")

        else:
            logging.warning("FAILED. Status code: %d", r.status_code)

    else:
        logging.warning("FAILED. Status code: %d", r.status_code)
