#!/usr/bin/env python3

import os, sys, subprocess
from pprint import pprint as pprint
import datetime

import ggntuples_handler
import config
config.datasets_data = config.get_datasets_data_analysis()
config.datasets_mc = config.get_datasets_mc_analysis()

import utils.accumulator_merger

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor

import coffea.processor as processor
from coffea import hist
from coffea.util import load, save

import numpy as np

import uproot

import gc

import yaml
config_doc = yaml.load(open("config/config.yaml", "r"), Loader=yaml.FullLoader)


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-t", "--test", help="Run a file loader test.", action="store_true")
parser.add_argument("-b", "--background", help="Run in non-interactive mode. If no --dataset is provided, run over all.", action="store_true")
parser.add_argument("-d", "--dataset", help="Dataset to be analyzed. Run in non-interactive mode.")
parser.add_argument("-y", "--year", help="Year for MC dataset. To be used only with --dataset (-d) option")
parser.add_argument("-c", "--chunk", help="Chunk of dataset. To be used only with --dataset (-d) option", default="-1")
parser.add_argument("-m", "--merge", help="Don't run the analyzer. Only merge outputs from buffer zone.", action="store_true")
args = parser.parse_args()

from pick import pick

def file_tester(dataset):
  _errors = []
  _name_year = ""
  if dataset["dataset_name"].startswith("Run"):
    _name_year = dataset["dataset_name"]
  else:
    _name_year = dataset["dataset_name"] + " - " +dataset["year"]
  for i, f in enumerate(dataset["files"]):
    try:
      file = uproot.open(f)
      del file 
      gc.collect() # force the garbage collector to clear memory
    except:
      print("ERROR loading: "+f)
      _errors.append("ERROR loading: "+f)
  return _errors

# def merger(ds):
#   # print("\n\n--> Merging files for: " + ds["dataset_name"] + " - "+ ds["year"])
#   # list saved outputs
#   result = subprocess.run("ls outputs/buffer/ana_" + ds["dataset_name"] + "_"+ ds["year"] + "_*.coffea", shell=True, stdout=subprocess.PIPE)

#   merged_output = processor.dict_accumulator({})
#   from tqdm import tqdm
#   # memory save outputs merging
#   for f in tqdm(result.stdout.decode('utf-8').splitlines(), desc=ds["dataset_name"] + "_"+ ds["year"], unit=" files"):
#   # for f in result.stdout.decode('utf-8').splitlines():
#       merged_output = merged_output + load(f)
#       # os.system("rm -f "+f)
#   # with open("outputs/ana_" + ds["dataset_name"] + "_"+ ds["year"] + ".coffea", 'wb') as fout:
#     # import cloudpickle
#     # cloudpickle.dump(merged_output, fout)  
#   save(merged_output, "outputs/ana_" + ds["dataset_name"] + "_"+ ds["year"] + ".coffea")

def merger(ds):
  from tqdm import tqdm
  # print("\n\n--> Merging files for: " + ds["dataset_name"] + " - "+ ds["year"])
  result = subprocess.run("find outputs/buffer/ -type f -name 'ana_" + ds["dataset_name"] + "_"+ ds["year"] + "_*.coffea'", shell=True, stdout=subprocess.PIPE)
  files_list = []
  for f in tqdm(result.stdout.decode('utf-8').splitlines(), desc="Loading: " + ds["dataset_name"] + " - "+ ds["year"], unit=" files"):
    files_list.append(load(f))
  save(utils.accumulator_merger(files_list), "outputs/ana_" + ds["dataset_name"] + "_"+ ds["year"] + ".coffea")


if __name__ == '__main__':
  ############################
  # dataset selector
  ############################

  # check if a specific dataset has been requested
  if args.dataset != None:
    if args.year == None and not(args.dataset.startswith("Run")):
      print("\n\nERROR: Dataset and Year should be set together!")
      exit()
    args.background = True
    isData = args.dataset.startswith("Run")
    if isData:
      _buff = []
      for d in config.datasets_data:
        if d["dataset_name"] == args.dataset and (d["chunk"] == args.chunk or int(args.chunk) < 0):
          _buff.append(d)
      if _buff == []:
        print("\n\nDataset + chunk not found!")
        exit()
      config.datasets_data = _buff
      config.datasets_mc = []
    else:
      _buff = []
      for d in config.datasets_mc:
        if d["dataset_name"] == args.dataset and d["year"] == args.year and (d["chunk"] == args.chunk or int(args.chunk) < 0):
          _buff.append(d)
      if _buff == []:
        print("\n\nDataset + year + chunk not found!")
        exit()
      config.datasets_data = []
      config.datasets_mc = _buff

  # if no dataset is request ask the user on which ones to run
  if args.background == False:
    # buffer datasets
    datasets_mc_buff = config.datasets_mc
    datasets_data_buff = config.datasets_data

    title = 'Select datasets to analyze (press SPACE to mark, ENTER to continue):'
    options_mc = [d["dataset_name"]+ " - "+d["year"] for d in config.datasets_mc]
    options_data = [d["dataset_name"]+ " - "+d["year"] for d in config.datasets_data]
    options = ["All MC", "All Data"] + options_mc + options_data
    selected = pick(options, title, multi_select=True, min_selection_count=1)

    # clear selected selected
    selected = [s[0] for s in selected]

    # build dataset list for MC
    _buff = []
    for d in config.datasets_mc:
      if (d["dataset_name"]+ " - "+d["year"]) in selected:
        _buff.append(d)
    config.datasets_mc = _buff

    # build dataset list for Data
    _buff = []
    for d in config.datasets_data:
      if (d["dataset_name"]+ " - "+d["year"]) in selected:
        _buff.append(d)
    config.datasets_data = _buff

    # build dataset list if "All" is selected
    if "All MC" in selected:
      config.datasets_mc = datasets_mc_buff
    if "All Data" in selected:
      config.datasets_data = datasets_data_buff

  ############################
  # test files loading
  ############################
  from multiprocessing.dummy import Pool
  import tqdm

  if args.test:
    # MC
    if config.datasets_mc != []:
      mc_errors = []
      print("\n\n--> Starting test for MC datasets...")
      # loop over files
      with ProcessPoolExecutor(max_workers = config_doc['n_processors'] ) as executor:
          results = list(tqdm.tqdm(executor.map(file_tester, config.datasets_mc), unit=" files", total=len(config.datasets_mc)))
      mc_errors.append(results)

      # with Pool(config_doc['n_processors']) as p:
      #   # mc_errors = p.map(file_tester, config.datasets_mc)
      #   for j in tqdm.tqdm(p.imap_unordered(file_tester, config.datasets_mc), total=len(config.datasets_mc)):
      #     mc_errors.append(j)
    
    # Data
    if config.datasets_data != []:
      data_errors = []
      print("\n\n--> Starting test for Data datasets...")
      # loop over files
      with ProcessPoolExecutor(max_workers = config_doc['n_processors'] ) as executor:
          results = list(tqdm.tqdm(executor.map(file_tester, config.datasets_data), unit=" files", total=len(config.datasets_data)))
      data_errors.append(results)

      # with Pool(config_doc['n_processors']) as p:
      #   # data_errors = p.map(file_tester, config.datasets_data)
      #   for j in tqdm.tqdm(p.imap_unordered(file_tester, config.datasets_data), total=len(config.datasets_data)):
      #     data_errors.append(j)

    # print test results
    print("\n\n")
    if config.datasets_mc != []:
      if all(e == [] for e in mc_errors):
        print("--> No loading errors found for MC.")
      else:
        print("--> Loading errors for MC:")
        for e in mc_errors:
          if e != []:
            print(e[0])
    if config.datasets_data != []:
      if all(e == [] for e in data_errors):
        print("--> No loading errors found for MC.")
      else:
        print("--> Loading errors for MC:")
        for e in data_errors:
          if e != []:
            print(e[0])
    exit()

  ############################
  # run over selected datasets
  ############################  
  import time
  start_time = time.time()

  print("\n\n--> Starting ---")
  ############################
  # Merger
  ############################ 
  if args.merge:

    # remove chuncks from list of datasets - MC
    datasets_to_merge_mc = []
    for ds in config.datasets_mc:
      datasets_to_merge_mc.append({
        "dataset_name": ds["dataset_name"],
        "year": ds["year"],
        })
    datasets_to_merge_mc = [dict(t) for t in {tuple(d.items()) for d in datasets_to_merge_mc}]

      # launch threads to merge MC
    if len(datasets_to_merge_mc) > 1:
      # with ThreadPoolExecutor(max_workers=config_doc["n_processors"]) as executor:
      with ProcessPoolExecutor(max_workers=config_doc["n_processors"]) as executor:
        executor.map(merger, datasets_to_merge_mc)
    else:
      for ds in datasets_to_merge_mc:
        merger(ds)

    # remove chuncks from list of datasets - Data
    datasets_to_merge_data = []
    for ds in config.datasets_data:
      datasets_to_merge_data.append({
        "dataset_name": ds["dataset_name"],
        "year": ds["year"],
        })
    datasets_to_merge_data = [dict(t) for t in {tuple(d.items()) for d in datasets_to_merge_data}]

    # launch threads to merge Data
    if len(datasets_to_merge_data) > 1:
      # with ThreadPoolExecutor(max_workers=config_doc["n_processors"]) as executor:
      with ProcessPoolExecutor(max_workers=config_doc["n_processors"]) as executor:
        executor.map(merger, datasets_to_merge_data)
    else:
      for ds in datasets_to_merge_data:
        merger(ds)

  ############################
  # Analyzer
  ############################ 
  else:
    ####################################
    # MC - multiprocess
    for ds in config.datasets_mc:
      print("\n\n--> H/Z --> Upsilon + Photon analyzer for: " + ds["dataset_name"] + " - "+ ds["year"] + " - Chunk: "+ ds["chunk"])
      # clear buffer
      os.system("rm -f outputs/buffer/ana_" + ds["dataset_name"] + "_"+ ds["year"] + "_"+ ds["chunk"] + "_*.coffea")
      # run the analyzer processor
      output = processor.run_uproot_job(config.get_fileset(ds),
                                    treename='ggNtuplizer/EventTree',
                                    processor_instance=ggntuples_handler.AnalyzerProcessor(ds["dataset_name"]+"_"+ds["year"]+"_"+ds["chunk"]),
                                    executor=processor.futures_executor,
                                    # executor=processor.iterative_executor,
                                    executor_args={'workers': config_doc["n_processors"], 'flatten': True},
                                    # executor_args={'flatten': True},
                                    chunksize=config_doc["chunksize"],
                                    # chunksize=1000, maxchunks=2,
                                   )
      print("\n\n--> Done with: "+ ds["dataset_name"] + " - "+ ds["year"] + " - Chunk: "+ ds["chunk"])
      # save(output, "outputs/ana_" + ds["dataset_name"] + "_"+ ds["year"] + ".coffea")

    ####################################
    # Data - multiprocess 
    for ds in config.datasets_data:
      print("\n\n--> H/Z --> Upsilon + Photon analyzer for: " + ds["dataset_name"] + " - "+ ds["year"] + " - Chunk: "+ ds["chunk"])
      # clear buffer
      os.system("rm -f outputs/buffer/ana_" + ds["dataset_name"] + "_"+ ds["year"] + "_"+ ds["chunk"] + "_*.coffea")
      # run the analyzer processor
      output = processor.run_uproot_job(config.get_fileset(ds),
                                    treename='ggNtuplizer/EventTree',
                                    processor_instance=ggntuples_handler.AnalyzerProcessor(ds["dataset_name"]+"_"+ds["year"]+"_"+ds["chunk"]),
                                    executor=processor.futures_executor,
                                    # executor=processor.iterative_executor,
                                    executor_args={'workers': config_doc["n_processors"], 'flatten': True},
                                    # executor_args={'flatten': True},
                                    chunksize=config_doc["chunksize"],
                                    # chunksize=1000, maxchunks=2, 
                                   )
      print("\n\n--> Done with: "+ ds["dataset_name"] + " - "+ ds["year"] + " - Chunk: "+ ds["chunk"])
      # save(output, "outputs/ana_" + ds["dataset_name"] + "_"+ ds["year"] + ".coffea")


  print("\n\n--> Done in %s seconds ---" % datetime.timedelta(seconds=(time.time() - start_time)))

  