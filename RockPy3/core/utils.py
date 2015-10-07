__author__ = 'mike'
import logging

def create_logger(name):
    log = logging.getLogger(name=name)
    log.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(levelname)-10s %(name)-20s %(message)s', "%H:%M:%S")
    # formatter = logging.Formatter('%(asctime)s: %(levelname)-10s %(name)-20s %(message)s')
    # fh = logging.FileHandler('RPV3.log')
    # fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    # ch.setLevel(logging.WARNING)
    ch.setLevel(logging.NOTSET)
    ch.setFormatter(formatter)
    # log.addHandler(fh)
    log.addHandler(ch)

    return log  # ch#, fh