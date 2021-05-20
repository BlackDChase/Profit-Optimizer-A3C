#!/usr/bin/env python3
"""
Minimalist and sane interface with the PEP8 breaking (and non idempotent) logging STL module
"""
import logging
import datetime

__author__ = 'Biribiri,BlackDChase'
__version__ = '1.1.0'

"""
weeks = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
today = datetime.datetime.today()
name = weeks[today.weekday()]+"-"+str(today.day).zfill(2)+"-"+str(today.month).zfill(2)+"-"+str(today.year)[2:]
logging.basicConfig(level=logging.DEBUG, filename="../logs/"+name+".tsv", format='%(asctime)s	%(levelname)s	%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
#"""
directory = "../logs/"
filename = str(datetime.datetime.now()).replace(":","-").replace(" ","-").split(".")[0][:-3]+"-Olog.tsv"
logging.basicConfig(level=logging.DEBUG, filename=directory+filename, format='%(asctime)s\t%(levelname)s\t%(message)s', datefmt='%Y-%m-%d\t%H:%M:%S')
#"""
#logging.addLevelName(level=logging.info,levelName="GPUusage")

def info(information,debug=False):
    if debug:
        print(information)
    logging.info(information)

debug = logging.debug

'''
def gpuStatus(usage):
    msg = "Allocated: " + str(round(usage/(1024**2),1)) + " MB"
    logging.GPUusage(msg)
#'''
info("New session initiated")
