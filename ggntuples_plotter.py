#!/usr/bin/env python3

import os, sys
from pprint import pprint as pprint
import datetime

import ggntuples_handler
import config
config.datasets_data = config.get_datasets_data_no_chunks_analysis()
config.datasets_mc = config.get_datasets_mc_no_chunks_analysis()

import coffea.processor as processor
from coffea import hist
from coffea.util import load, save

import numpy as np 

import matplotlib.pyplot as plt

if __name__ == '__main__':
  import time
  start_time = time.time()

  print("--> Starting ---")

  ####################################
  # MC 
  _processor = ggntuples_handler.PlotterProcessor()
  _dummy_accumulator = _processor.accumulator.identity()
  processor.iterative_executor(config.datasets_mc, _processor.process, _dummy_accumulator)

  ####################################
  # Data 
  _processor = ggntuples_handler.PlotterProcessor()
  _dummy_accumulator = _processor.accumulator.identity()
  processor.iterative_executor(config.datasets_data, _processor.process, _dummy_accumulator)


  print("--> Done in %s seconds ---" % datetime.timedelta(seconds=(time.time() - start_time)))

  