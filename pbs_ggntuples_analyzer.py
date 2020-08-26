#!/usr/bin/env python3

import os, sys
import datetime

import config
config.datasets_data = config.get_datasets_data_analysis()
config.datasets_mc = config.get_datasets_mc_analysis()

import time

import getpass
username = getpass.getuser()

import subprocess

import hashlib

import yaml
config_doc = yaml.load(open("config/config.yaml", "r"), Loader=yaml.FullLoader)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-sub", "--submit", help="Submit jobs", action="store_true")
parser.add_argument("-mon", "--monitor", help="Monitor tasks", action="store_true")
parser.add_argument("-cle", "--clear", help="Clear workspace", action="store_true")
parser.add_argument("-d", "--dataset", help="Dataset to be analyzed. If not specified, all datasets will be submited.")
parser.add_argument("-y", "--year", help="Year for MC dataset. To be used only with --dataset (-d) option")
parser.add_argument("-c", "--check", help="Check jobs logs and print errors.", action="store_true")
# args = parser.parse_args()
args, unknown = parser.parse_known_args()





##############
# Celery App #
##############
from celery import Celery

app = Celery('pbs_ggntuples_analyzer', backend=f'redis://{config_doc["celery_master_node_ip"]}:{config_doc["redis_port"]}/1', broker=f'redis://{config_doc["celery_master_node_ip"]}:{config_doc["redis_port"]}/0')

app.conf.update(
    CELERY_TASK_RESULT_EXPIRES=0,
)

@app.task
def run_taks(ds):
  print("\n\n--> Starting H/Z --> Upsilon + Photon analyzer for: " + ds["dataset_name"] + " - "+ ds["year"]+" - Chunk: "+ds["chunk"])
  exec_command = f'./ggntuples_analyzer.py -d {ds["dataset_name"]} -y {ds["year"]} -c {ds["chunk"]} > pbs_buffer/logs/log_{ds["dataset_name"]}_{ds["year"]}_{ds["chunk"]}.log 2>&1 '
  exec_command = f'./ggntuples_analyzer.py -d {ds["dataset_name"]} -y {ds["year"]} -c {ds["chunk"]} > pbs_buffer/logs/log_{ds["dataset_name"]}_{ds["year"]}_{ds["chunk"]}.log 2>&1 '
  proc_child = subprocess.Popen(exec_command, stdout=subprocess.PIPE, shell=True)
  proc_child.communicate()[0]
  return proc_child.returncode

def do_submit(ds):
  # run_taks.delay(ds)
  run_taks.apply_async(args=(ds,), kwargs=None, task_id=ds["dataset_name"]+'_'+ds["year"]+'_'+ds["chunk"])
  
def do_monitor():
  try:
    print("\n\n--> Setup tunnel:")
    print(r"ssh -N -f -L 5556:localhost:5556 ftaraujo@planck")
    print("Access: http://localhost:5556\n\n")
    print("--> Press Ctrl+C to exit.")
    os.system("flower -A pbs_ggntuples_analyzer --port=5556")

  except KeyboardInterrupt:
      print ('--> Exiting...')


def start_redis():
  os.system("redis-server --bind 0.0.0.0 --port 4023 > pbs_buffer/redis.log 2>&1  &")

def clear_workspace():
  # os.system("rm -rf pbs_buffer/*")
  os.system(f"qselect -u {username} | xargs qdel")
  os.system("pkill -9 -f celery*")
  os.system("pkill -9 -f flower*")
  os.system("pkill -9 -f redis-server*")
  


pbs_sub_file = r"""
#!/bin/bash

# send email
#PBS -M ftaraujo@ifi.unicamp.br
#PBS -m abe

### Process name
#PBS -N pbs_ggntuples_analyzer___SUB_NUMBER__

### Output files
#PBS -e pbs_buffer/worker_log___SUB_NUMBER___err.log
#PBS -o pbs_buffer/worker_log___SUB_NUMBER___out.log

# queues
#PBS -q __QUEUE__
#PBS -l nodes=__NODES__:ppn=__PPN__


# enter working dir
cd $PBS_O_WORKDIR
echo "Location:"
pwd
# ls -d /cvmfs/cms.cern.ch/cmsset_default.sh

# load env
source ~/.bashrc
source /cvmfs/cms.cern.ch/cmsset_default.sh
conda deactivate
conda deactivate 
conda activate HZUpsilonPhotonRun2Env
eval `scramv1 runtime -sh`
export PYTHONPATH=$CONDA_PREFIX/lib/python3.6/site-packages/
source /cvmfs/sft.cern.ch/lcg/releases/LCG_92python3/ROOT/6.12.04/x86_64-centos7-gcc7-opt/bin/thisroot.sh # for bash


# start worker
celery -A pbs_ggntuples_analyzer worker --loglevel=info --concurrency=__CONC__ --hostname=worker___SUB_NUMBER__@%h___NAME_APPEND__

exit
"""

def launch_workers(n_jobs):
  print("\n\n--> To launch workers...")
  for w in range(n_jobs):
    with open(f"pbs_buffer/pbs_sub_file_{w}.sh", "w") as sub_file:
      pbs_sub_file_buff = pbs_sub_file.replace("__SUB_NUMBER__", str(w)).replace("__QUEUE__", config_doc["queue"]).replace("__PPN__", str(config_doc["ppn"])).replace("__NODES__", str(config_doc["nodes"])).replace("__CONC__", str(config_doc["concurrency"])) 
      pbs_sub_file_buff = pbs_sub_file_buff.replace("__NAME_APPEND__", str(hashlib.sha1(pbs_sub_file.encode("UTF-8")).hexdigest()[:10]))
      sub_file.write(pbs_sub_file_buff)
      os.system(f"chmod +x pbs_buffer/pbs_sub_file_{w}.sh")
      # os.system(f"qstat -q")
      # os.system(f"qsub -q {config_doc['queue']} pbs_buffer/pbs_sub_file_{w}.sh")
      print(f"qsub -q {config_doc['queue']} pbs_buffer/pbs_sub_file_{w}.sh")
  print("\n\n")

if __name__ == '__main__':
  start_time = time.time() 
  
  if args.submit:
    print("\n\n--> Starting ---")
    os.system("rm -rf pbs_buffer/*")
    os.system("mkdir -p pbs_buffer/logs")

    ############################
    # Redis Server
    ############################
    print("--> Launching Redis Server...")
    start_redis()

    ############################
    # launch workers
    ############################
    launch_workers(config_doc["n_jobs"])

    ############################
    # dataset selector
    ############################
    print("--> Filtering selected datasets...")
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


    #############################
    # Submit for Data
    #############################
    print("--> Submiting tasks - Data...")
    print(len(config.datasets_data))
    for ds in config.datasets_data:
      do_submit(ds)

    #############################
    # Submit for MC
    #############################
    print("--> Submiting tasks - MC...")
    for ds in config.datasets_mc:
      do_submit(ds)

    print("\n\n--> Done in %s seconds ---" % datetime.timedelta(seconds=(time.time() - start_time)))
    exit()

  if args.monitor:
    #############################
    # Monitoring
    #############################
    print("--> Starting Flower...")
    do_monitor()

  if args.clear:
    #############################
    # Clear Workspace
    #############################
    print("--> Clearing workspace...")
    clear_workspace()

  if (args.submit == False) and (args.monitor == False) and (args.clear == False):
    print("Try execute it with --help.")

  

  