#!/usr/bin/env python3

import os, sys
from pprint import pprint as pprint
import datetime

import ggntuples_handler
import config
config.datasets_data = config.get_datasets_data_no_chunks_trigger()
config.datasets_mc = config.get_datasets_mc_no_chunks_trigger()

import coffea.processor as processor
from coffea import hist
from coffea.util import load, save

import numpy as np

import cppyy

import gc

import yaml
config_doc = yaml.load(open("config/config.yaml", "r"), Loader=yaml.FullLoader)

# Muons BDT Score
cppyy.cppdef(r"""
#include "MuonMVAReader/Reader/interface/MuonGBRForestReader.hpp"
#pragma cling load("libMuonMVAReaderReader.so")

// MVA Reader
MuonGBRForestReader * reader_2016 = new MuonGBRForestReader(2016);
MuonGBRForestReader * reader_2017 = new MuonGBRForestReader(2017);
MuonGBRForestReader * reader_2018 = new MuonGBRForestReader(2018);
""")

# Rochester Correction
# https://twiki.cern.ch/twiki/bin/view/CMS/RochcorMuon
cppyy.cppdef(r"""
#include "utils/roccor_Run2_v3/RoccoR.cc"
RoccoR  rc2016("utils/roccor_Run2_v3/RoccoR2016.txt");
RoccoR  rc2017("utils/roccor_Run2_v3/RoccoR2017.txt");
RoccoR  rc2018("utils/roccor_Run2_v3/RoccoR2018.txt");
""")

if __name__ == '__main__':
  import time
  start_time = time.time()

  print("--> Starting ---")

  ####################################
  # MC 
  _processor = ggntuples_handler.TriggerSelectorProcessor()
  _dummy_accumulator = _processor.accumulator.identity()
  processor.iterative_executor(config.datasets_mc, _processor.process, _dummy_accumulator)
  # processor.futures_executor(config.datasets_mc, _processor.process, _dummy_accumulator, workers = config_doc['n_processors'])

  ####################################
  # Data 
  _processor = ggntuples_handler.TriggerSelectorProcessor()
  _dummy_accumulator = _processor.accumulator.identity()
  processor.iterative_executor(config.datasets_data, _processor.process, _dummy_accumulator)
  # processor.futures_executor(config.datasets_data, _processor.process, _dummy_accumulator, workers = config_doc['n_processors'])
  os.system("rm -rf outputs/trigger_sel_Run2016.root ; hadd -f outputs/trigger_sel_Run2016.root outputs/trigger_sel_Run2016*.root")
  os.system("rm -rf outputs/trigger_sel_Run2017.root ; hadd -f outputs/trigger_sel_Run2017.root outputs/trigger_sel_Run2017*.root")
  os.system("rm -rf outputs/trigger_sel_Run2018.root ; hadd -f outputs/trigger_sel_Run2018.root outputs/trigger_sel_Run2018*.root")

  print("--> Done in %s seconds ---" % datetime.timedelta(seconds=(time.time() - start_time)))

  