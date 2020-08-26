#!/usr/bin/env python3

import os, sys
from pprint import pprint as pprint
import datetime

from termcolor import colored

import config
config.datasets_data = config.get_datasets_data_analysis()
config.datasets_mc = config.get_datasets_mc_analysis()

import time

import getpass
username = getpass.getuser()

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-sub", "--submit", help="Submit jobs", action="store_true")
parser.add_argument("-har", "--harvest", help="Collect job results", action="store_true")
parser.add_argument("-d", "--dataset", help="Dataset to be analyzed. If not specified, all datasets will be submited.")
parser.add_argument("-y", "--year", help="Year for MC dataset. To be used only with --dataset (-d) option")
parser.add_argument("-c", "--check", help="Check jobs logs and print errors.", action="store_true")
args = parser.parse_args()

def clear(ds):
  os.system("rm -rf condor_buffer/"+ds["dataset_name"]+"_"+ds["year"]+"_"+ds["chunk"])
  os.system("mkdir -p condor_buffer/"+ds["dataset_name"]+"_"+ds["year"]+"_"+ds["chunk"])
  os.system("mkdir -p condor_buffer/"+ds["dataset_name"]+"_"+ds["year"]+"_"+ds["chunk"]+"/logs")

def submit(ds):
  print("\n\n--> Submitting H/Z --> Upsilon + Photon analyzer for: " + ds["dataset_name"] + " - "+ ds["year"]+" - Chunk: "+ds["chunk"])
  clear(ds)

  #######################
  # job file
  #######################
  jobFile = r"""universe = vanilla
Executable = condor_runner.sh
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
Output = logs/"""+ds["dataset_name"]+"_"+ds["year"]+"_"+ds["chunk"]+""".stdout
Error = logs/"""+ds["dataset_name"]+"_"+ds["year"]+"_"+ds["chunk"]+""".stderr
Log = logs/"""+ds["dataset_name"]+"_"+ds["year"]+"_"+ds["chunk"]+""".log
Arguments = """+ds["dataset_name"]+" "+ds["year"]+" "+ds["chunk"]+"""
x509userproxy = $ENV(X509_USER_PROXY)
transfer_input_files = ../CMSSW_WORKING_AREA.tar.gz
"""


#   if ds["dataset_name"].startswith("Run"):
#     jobFile = jobFile+"""
# request_cpus = 4
# """
  # if ds["dataset_name"].startswith("Run2016G"):
  #   jobFile = jobFile.replace("request_memory = 3072", "request_memory = 5120")

  # if ds["dataset_name"].startswith("Run2016H"):
  #   jobFile = jobFile.replace("request_memory = 6144", "request_memory = 5120")

  jobFile = jobFile+"""
Queue 1
"""
  with open('condor_buffer/'+ds["dataset_name"]+'_'+ds["year"]+'_'+ds["chunk"]+'/job.jdl', 'w') as f:
    f.write(jobFile)

  #######################
  # job executable
  #######################
  path = None
  if ds["data_or_mc"] == "mc":
    path = "MC/"+ds["year"]
  else:
    path = "Data/MuonEG"
  # executable file
  with open('condor_buffer/'+ds["dataset_name"]+'_'+ds["year"]+"_"+ds["chunk"]+'/condor_runner.sh', 'w') as f:
    f.write(
r"""#!/bin/bash
set -x
hostname
date
source /cvmfs/cms.cern.ch/cmsset_default.sh
eval `scramv1 project CMSSW CMSSW_10_2_15`
cd CMSSW_10_2_15/src
mv ../../CMSSW_WORKING_AREA.tar.gz .
tar -zxvf CMSSW_WORKING_AREA.tar.gz
ls -lha 
eval `scramv1 runtime -sh` # cmsenv is an alias not on the workers
python3 -m venv HZUpsilonPhotonRun2Env_ggntuples
source HZUpsilonPhotonRun2Env_ggntuples/bin/activate
source /cvmfs/sft.cern.ch/lcg/releases/LCG_92python3/ROOT/6.12.04/x86_64-centos7-gcc7-opt/bin/thisroot.sh
pip install --upgrade pip
pip3 install --upgrade --ignore-installed --force-reinstall coffea
pip3 install --upgrade --ignore-installed --force-reinstall pick
pip3 install --upgrade --ignore-installed --force-reinstall cppyy
pip3 install --upgrade --ignore-installed --force-reinstall -vvv 'xrootd==4.9.0'
cd HZUpsilonPhotonRun2Ana
mkdir plots
mkdir outputs
mkdir outputs/buffer
mkdir inputs

./ggntuples_analyzer.py -d $1 -y $2 -c $3
ls -lha outputs/buffer/ana_$1_$2_$3_*.coffea
cd outputs/buffer 
tar -zcvf ana_$1_$2_$3.tar.gz ana_$1_$2_$3_*.coffea
cd ..
cd ..
xrdcp --parallel 4 -r outputs/buffer/ana_$1_$2_$3.tar.gz  root://cmseos.fnal.gov//store/user/"""+username+"""/condor_buffer/.
""")
  os.system('chmod +x condor_buffer/'+ds["dataset_name"]+'_'+ds["year"]+'_'+ds["chunk"]+'/condor_runner.sh')

  #######################
  # submit job
  #######################
  os.system('cd condor_buffer/'+ds["dataset_name"]+'_'+ds["year"]+"_"+ds["chunk"]+'/ ; condor_submit job.jdl')

def check(ds):
  try:
    with open('condor_buffer/'+ds["dataset_name"]+'_'+ds["year"]+"_"+ds["chunk"]+'/logs/'+ds["dataset_name"]+'_'+ds["year"]+"_"+ds["chunk"]+'.log') as f:
      if 'return value 0' in f.read():
        print(colored(ds["dataset_name"]+'_'+ds["year"]+"_"+ds["chunk"] + ' :OK', 'green'))

      else:
        print(colored(ds["dataset_name"]+'_'+ds["year"]+"_"+ds["chunk"] + ' :ERROR', 'red'))
  except:
    print(colored(ds["dataset_name"]+'_'+ds["year"]+"_"+ds["chunk"] + ' :ERROR opening log file', 'red'))


def harvest(ds):
  # print("\n\n--> Collecting outputs for: " + ds["dataset_name"] + " - "+ ds["year"] + " - Chunk: "+ ds["chunk"])
  # os.system("rm -f outputs/buffer/ana_" + ds["dataset_name"] + "_"+ ds["year"] + "_"+ ds["chunk"] + "_*.coffea")
  # os.system("rm -f outputs/buffer/ana_" + ds["dataset_name"] + "_"+ ds["year"] + "_"+ ds["chunk"] + ".tar.gz")
  # os.system("rsync -har --info=progress2 /eos/uscms/store/user/"+username+"/condor_buffer/ana_" + ds["dataset_name"] + "_"+ ds["year"] + "_"+ ds["chunk"] + ".tar.gz outputs/buffer/.")
  # os.system("cd outputs/buffer ; tar-zxvf ana_" + ds["dataset_name"] + "_"+ ds["year"] + "_"+ ds["chunk"] + ".tar.gz")
  print("This should be fixed! Use no -d option...")



if __name__ == '__main__':
  start_time = time.time()

  if args.check:
    print("\n--> Checking logs")
    #############################
    # Check logs for Data
    #############################
    for ds in config.datasets_data:
      check(ds)

    #############################
    # Check logs for MC
    #############################
    for ds in config.datasets_mc:
      check(ds)    
  
  if args.submit:
    print("\n\n--> Starting ---")

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
          if d["dataset_name"] == args.dataset:
            _buff.append(d)
        if _buff == []:
          print("\n\nDataset + chunk not found!")
          exit()
        config.datasets_data = _buff
        config.datasets_mc = []
      else:
        _buff = []
        for d in config.datasets_mc:
          if d["dataset_name"] == args.dataset and d["year"] == args.year:
            _buff.append(d)
        if _buff == []:
          print("\n\nDataset + year + chunk not found!")
          exit()
        config.datasets_data = []
        config.datasets_mc = _buff
    else:    
      os.system("rm -rf /eos/uscms/store/user/"+username+"/condor_buffer/*")

    #############################
    # prepare the CMSSW tar file
    #############################
    with open('condor_buffer/excludes.txt', 'w') as f:
      f.write(
r"""*.coffea
.git
__pycache__
._*
*.pdf
*.png
condor_buffer
*.root
*.tar.gz
outputs/*"""
        )
    os.system(r'cd .. ; tar -zcvf HZUpsilonPhotonRun2Ana/condor_buffer/CMSSW_WORKING_AREA.tar.gz HZUpsilonPhotonRun2Ana/ --exclude-from=HZUpsilonPhotonRun2Ana/condor_buffer/excludes.txt')

    #############################
    # Submit for Data
    #############################
    for ds in config.datasets_data:
      submit(ds)

    #############################
    # Submit for MC
    #############################
    for ds in config.datasets_mc:
      submit(ds)

    print("\n\n--> Done in %s seconds ---" % datetime.timedelta(seconds=(time.time() - start_time)))
    exit()

  if args.harvest:
    #############################
    # collecting outputs
    #############################
    if args.dataset != None:
      # Copy data oputputs
      for ds in config.datasets_data:
        harvest(ds)
      # Copy mc oputputs
      for ds in config.datasets_mc:
        harvest(ds)
    # if no dataset was passed as argument, copy everything
    else:
      print("\n\n--> Collecting outputs...")
      os.system(r'find outputs/buffer/ -name "*" -exec rm {} \;')
       # os.system("rsync -har --info=progress2 /eos/uscms/store/user/"+username+"/condor_buffer/ana_*.coffea outputs/buffer/.")
      os.system("rsync -har --info=progress2 /eos/uscms/store/user/"+username+"/condor_buffer/ana*.tar.gz outputs/buffer/.")
      print("\n\n--> Unpacking outputs...")
      import tqdm
      for ds in tqdm.tqdm((config.get_datasets_data()+config.get_datasets_mc())):
        os.system("tar -C outputs/buffer/ -zxf outputs/buffer/ana_" + ds["dataset_name"] + "_"+ ds["year"]+"_"+ ds["chunk"]+".tar.gz")
        os.system("rm -f outputs/buffer/ana_" + ds["dataset_name"] + "_"+ ds["year"]+"_"+ ds["chunk"]+".tar.gz")


    print("\n\n--> Done in %s seconds ---" % datetime.timedelta(seconds=(time.time() - start_time)))






  if (args.submit == False) and (args.harvest == False) and (args.check == False):
    print("Try execute it with --help.")

  

  