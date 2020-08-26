#!/usr/bin/env python3

import os, sys
from pprint import pprint as pprint
import datetime

from coffea.util import load, save

import numpy as np



####################################
# Dataset
####################################

# Data
datasets_Data = [
{
  "dataset_name": "Run2016B-17Jul2018_ver1",
  "sel_file": "input_files/sel_Run2016B-17Jul2018_ver1_2016.coffea",
  "year": "2016",
},
{
  "dataset_name": "Run2016B-17Jul2018_ver2",
  "sel_file": "input_files/sel_Run2016B-17Jul2018_ver2_2016.coffea",
  "year": "2016",
},
{
  "dataset_name": "Run2016C-17Jul2018",
  "sel_file": "input_files/sel_Run2016C-17Jul2018_2016.coffea",
  "year": "2016",
},
{
  "dataset_name": "Run2016D-17Jul2018",
  "sel_file": "input_files/sel_Run2016D-17Jul2018_2016.coffea",
  "year": "2016",
},
{
  "dataset_name": "Run2016E-17Jul2018",
  "sel_file": "input_files/sel_Run2016E-17Jul2018_2016.coffea",
  "year": "2016",
},
{
  "dataset_name": "Run2016F-17Jul2018",
  "sel_file": "input_files/sel_Run2016F-17Jul2018_2016.coffea",
  "year": "2016",
},
{
  "dataset_name": "Run2016G-17Jul2018",
  "sel_file": "input_files/sel_Run2016G-17Jul2018_2016.coffea",
  "year": "2016",
},
{
  "dataset_name": "Run2016H-17Jul2018",
  "sel_file": "input_files/sel_Run2016H-17Jul2018_2016.coffea",
  "year": "2016",
},
# {
#   "dataset_name": "Run2018C-17Sep2018-v1",
#   "sel_file": "input_files/sel_Run2018C-17Sep2018-v1_2018.coffea",
#   "year": "2018",
# },
]

# MC
datasets_MC = [
{
  "dataset_name": "ggH_HToUps1SG",
  "sel_file": "input_files/sel_ggH_HToUps1SG_2016.coffea",
  "year": "2016",
},
# {
#   "dataset_name": "ggH_HToUps1SG",
#   "sel_file": "input_files/sel_ggH_HToUps1SG_2018.coffea",
#   "year": "2018",
# },
{
  "dataset_name": "GluGluHToMuMuG_M125_MLL-0To60_Dalitz_012j",
  "sel_file": "input_files/sel_GluGluHToMuMuG_M125_MLL-0To60_Dalitz_012j_2016.coffea",
  "year": "2016",
},
# {
#   "dataset_name": "GluGluHToMuMuG_M125_MLL-0To60_Dalitz_012j",
#   "sel_file": "input_files/sel_GluGluHToMuMuG_M125_MLL-0To60_Dalitz_012j_2018.coffea",
#   "year": "2018",
# },

{
  "dataset_name": "ttH_HToUps1SG",
  "sel_file": "input_files/sel_ttH_HToUps1SG_2016.coffea",
  "year": "2016",
},
# {
#   "dataset_name": "ttH_HToUps1SG",
#   "sel_file": "input_files/sel_ttH_HToUps1SG_2018.coffea",
#   "year": "2018",
# },
{
  "dataset_name": "VBFH_HToUps1SG",
  "sel_file": "input_files/sel_VBFH_HToUps1SG_2016.coffea",
  "year": "2016",
},
{
  "dataset_name": "WmH_HToUps1SG",
  "sel_file": "input_files/sel_WmH_HToUps1SG_2016.coffea",
  "year": "2016",
},
# {
#   "dataset_name": "WmH_HToUps1SG",
#   "sel_file": "input_files/sel_WmH_HToUps1SG_2018.coffea",
#   "year": "2018",
# },
{
  "dataset_name": "WpH_HToUps1SG",
  "sel_file": "input_files/sel_WpH_HToUps1SG_2016.coffea",
  "year": "2016",
},
# {
#   "dataset_name": "WpH_HToUps1SG",
#   "sel_file": "input_files/sel_WpH_HToUps1SG_2018.coffea",
#   "year": "2018",
# },
{
  "dataset_name": "ZGTo2MuG_MMuMu-2To15",
  "sel_file": "input_files/sel_ZGTo2MuG_MMuMu-2To15_2016.coffea",
  "year": "2016",
},
# {
#   "dataset_name": "ZGTo2MuG_MMuMu-2To15",
#   "sel_file": "input_files/sel_ZGTo2MuG_MMuMu-2To15_2018.coffea",
#   "year": "2018",
# },
{
  "dataset_name": "ZH_HToUps1SG",
  "sel_file": "input_files/sel_ZH_HToUps1SG_2016.coffea",
  "year": "2016",
},
# {
#   "dataset_name": "ZH_HToUps1SG",
#   "sel_file": "input_files/sel_ZH_HToUps1SG_2018.coffea",
#   "year": "2018",
# },
{
  "dataset_name": "ZToUpsilon1SGamma",
  "sel_file": "input_files/sel_ZToUpsilon1SGamma_2016.coffea",
  "year": "2016",
},
# {
#   "dataset_name": "ZToUpsilon1SGamma",
#   "sel_file": "input_files/sel_ZToUpsilon1SGamma_2018.coffea",
#   "year": "2018",
# },
{
  "dataset_name": "ZToUpsilon2SGamma",
  "sel_file": "input_files/sel_ZToUpsilon2SGamma_2016.coffea",
  "year": "2016",
},
# {
#   "dataset_name": "ZToUpsilon2SGamma",
#   "sel_file": "input_files/sel_ZToUpsilon2SGamma_2018.coffea",
#   "year": "2018",
# },
{
  "dataset_name": "ZToUpsilon3SGamma",
  "sel_file": "input_files/sel_ZToUpsilon3SGamma_2016.coffea",
  "year": "2016",
},
# {
#   "dataset_name": "ZToUpsilon3SGamma",
#   "sel_file": "input_files/sel_ZToUpsilon3SGamma_2018.coffea",
#   "year": "2018",
# },
]


def selector_evaluator(dataset, selected_events):
  ####################################
  # get accumulator and load pre-selected events
  ####################################
  dataset_name = dataset['dataset_name']
  year = dataset["year"]

  print("--> Starting processing of: "+ dataset["dataset_name"] + " - "+ dataset["year"])

  df = load("input_files/sel_" + dataset_name + "_" + year + ".coffea")

  ####################################
  # get mc weight
  ####################################
  weight = 1.0
  xsec = 1.0 # one needs to define the xSecs per proccess, per sample (somehow...)
  int_lumi = 39.9 # pb^1 (2016)
  if not(dataset["dataset_name"].startswith("Run")): # if it is MC
    weight = xsec*int_lumi/df["cutflow"]["all_events"]

  selected_events = {
    "dataset_name" : dataset_name,
    "year": year,
    "all_events": df["cutflow"]["all_events"],
    "selected_events": 0,
  }

  ########################################
  # get some data
  ########################################

  # example (leadming muon pt):
  print("\n--> Number of pre-selected events:")
  pprint(df["leadmu_pt"].value.size)

  print("\n--> Array of leading muons Pt:")
  pprint(df["leadmu_pt"].value)

  ### other variables
  # # leading muon
  # leadmu_pt
  # leadmu_eta
  # leadmu_phi

  ### trailing muon
  # trailmu_pt
  # trailmu_eta
  # trailmu_phi        

  ### photon
  # photon_pt
  # photon_eta
  # photon_phi    
  # photon_calibEnergy    
  # photon_calibEt    
  # photon_etasc    

  ### upsilon
  # upsilon_pt
  # upsilon_eta
  # upsilon_phi
  # upsilon_mass

  ### boson
  # boson_pt
  # boson_eta
  # boson_phi
  # boson_mass


  ### kinematical cuts
  # delta_R_leading_photon
  # delta_R_trailing_photon
  # delta_R_upsilon_photon
  # delta_Phi_leading_photon
  # et_photon_over_M_boson
  # pt_upsilon_over_M_boson

  ########################################
  # do some selection
  ########################################

  # loop over events
  for evt in range(df["leadmu_pt"].value.size):
    if df["delta_R_leading_photon"].value[evt] >= 0.0: # dummy selection
      selected_events["selected_events"] += 1.0*weight

  ########################################
  # done
  ########################################
  return selected_events




if __name__ == '__main__':
  import time
  start_time = time.time()

  print("--> Starting ---")

  # loop over MC 
  selected_events_MC = []
  for d in datasets_MC:
    selected_events_MC.append(selector_evaluator(d, selected_events_MC))
  
  # loop over Data 
  selected_events_Data = []
  for d in datasets_Data:
    selected_events_Data.append(selector_evaluator(d, selected_events_Data))


  ########################################
  # print selected events
  ########################################

  print("\n\n--> Selected events for MC:")
  pprint(selected_events_MC)

  print("\n\n--> Selected events for Data:")
  pprint(selected_events_Data)

  print("--> Done in %s seconds ---" % datetime.timedelta(seconds=(time.time() - start_time)))

  