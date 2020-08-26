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

if __name__ == '__main__':
  import time
  start_time = time.time()

  print("--> Starting ---")

  ####################################
  # MC - multiprocess
  for ds in config.datasets_mc:
    print("--> H/Z --> Upsilon + Photon analyzer for: " + ds["dataset_name"])
    output = processor.run_uproot_job(config.get_fileset(ds),
                                  treename='ggNtuplizer/EventTree',
                                  processor_instance=ggntuples_handler.GenAnalyzerProcessor(ds["dataset_name"]),
                                  executor=processor.futures_executor,
                                  # executor_args={'workers': 6,},
                                  executor_args={'workers': 6, 'flatten': True},
                                  chunksize=1000000000000,
                                 )
    print("--> Saving output for: "+ ds["dataset_name"])
    # pprint(output)
    save(output, "outputs/ana_" + ds["dataset_name"] + ".coffea")

  print("--> Done in %s seconds ---" % datetime.timedelta(seconds=(time.time() - start_time)))

  