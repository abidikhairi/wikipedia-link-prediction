import logging


def get_logger(name):
    datefmt = '%Y-%m-%d %H:%M:%S.%f' # e.g. 2023-02-15 19:45:30.9123457

    logging.basicConfig(level=logging.INFO, datefmt=datefmt)
    logger = logging.getLogger(name=name)

    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - [%(levelname)s] - [%(module)s]- %(message)s")
    
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger