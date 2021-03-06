#!/usr/bin/env python3
"""
Minimalist and sane interface with the PEP8 breaking (and non idempotent) logging STL module
"""
import logging

logging.basicConfig(level=logging.DEBUG, filename="../../Saved_model/log.tsv", format='%(asctime)s	%(levelname)s	%(message)s', datefmt='%Y-%m-%d %H:%M:%S')

def info(information,debug=False):
    if debug:
        print(information)
    logging.info(information)

def debug(*args,**kwargs):
    logging.debug(args,kwargs)

info("New session initiated")
