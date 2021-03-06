#!/usr/bin/env python3
"""
Minimalist and sane interface with the PEP8 breaking (and non idempotent) logging STL module
"""
import logging

logging.basicConfig(level=logging.DEBUG, filename="../../log.tsv", format='%(asctime)s	%(levelname)s	%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logging.addLevelName(level=logging.info,levelName="GPUusage")

def info(information,debug=False):
    if debug:
        print(information)
    logging.info(information)

def debug(*args,**kwargs):
    logging.debug(args,kwargs)

def gpuStatus(usage):
    msg = "Allocated: " + str(round(usage/(1024**2),1)) + " MB"
    logging.GPUusage(msg)

info("New session initiated")
