import logging


def get_logger(name=__name__):
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(name)
