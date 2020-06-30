import os
import time
import logging

def Logger(level, log_name=None):
    logger = logging.getLogger(__name__)
    logger.setLevel(level)
    formatter = logging.Formatter("%(levelname)s:%(module)s:%(lineno)d:%(message)s")

    assert log_name is not None
    time_tag = time.strftime("%Y-%a-%b-%d-%H-%M-%S", time.localtime())
    if not os.path.exists('./logs'):
        os.mkdir('./logs')
    file_path = './logs/{}-{}.log'.format(log_name, time_tag)
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger