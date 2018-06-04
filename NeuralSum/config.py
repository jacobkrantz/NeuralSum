#!/usr/bin/env python
import json
import logging as log

with open('config.json') as json_data_file:
    config = json.load(json_data_file)


# set global logging utility

if(config["log_level"] == "info"):
    log_level = log.INFO
elif(config["log_level"] == "warning"):
    log_level = log.WARNING
else:
    log_level = log.DEBUG

log.basicConfig(
    format='%(asctime)s %(levelname)s:%(message)s',
    datefmt='%X',
    level=log_level
)
