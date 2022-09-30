import xcontest
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
logger = logging.getLogger(__name__)


def main(data_loc: str = 'data') -> None:
    logger.info('Iterate over takeoffs.')
    for takeoff in xcontest.Takeoff:
        logger.info('Takeoff: %s', takeoff.name)
        for flight in xcontest.get_flights(takeoff=takeoff, data_loc=data_loc):
            pass


if __name__ == '__main__':
    main()
