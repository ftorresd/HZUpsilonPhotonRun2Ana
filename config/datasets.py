import subprocess
import os 
from pprint import pprint as pprint
import copy
from coffea.util import load, save
import yaml

config_doc = yaml.load(open("config/config.yaml", "r"), Loader=yaml.FullLoader)
datasets_doc = yaml.load(open("config/datasets.yaml", "r"), Loader=yaml.FullLoader)

datasets_mc_trigger = datasets_doc["datasets_mc_trigger"]
datasets_mc_analysis = datasets_doc["datasets_mc_analysis"]

datasets_data_SingleMuon = datasets_doc["datasets_data_SingleMuon"]
datasets_data_MuonEG = datasets_doc["datasets_data_MuonEG"]


def get_fileset(ds):
  fileset = {
    ds["dataset_name"]+"_"+ds["year"]: ds["files"]
  }
  return fileset


def file_pather(path):
  if path.startswith("root://"):
    result = subprocess.run("(eval `scram unsetenv -sh`; gfal-ls gsiftp://cmseos-gridftp.fnal.gov//eos/uscms/"+path.split('//')[2]+")", shell=True, stdout=subprocess.PIPE)
    # result = subprocess.run(["xrdfs",  path.split('//')[0]+"//"+path.split('//')[1]+"/", "ls",  "/"+path.split('//')[2]], stdout=subprocess.PIPE)
    # print([path.split('//')[0]+"//"+path.split('//')[1]+"//"+path.split('//')[2]+"/"+f for f in result.stdout.decode('utf-8').splitlines()])
    return [path.split('//')[0]+"//"+path.split('//')[1]+"//"+path.split('//')[2]+"/"+f for f in result.stdout.decode('utf-8').splitlines()]
  else:
    result = subprocess.run(['ls', path], stdout=subprocess.PIPE)
    return [path+'/'+f for f in result.stdout.decode('utf-8').splitlines()]


# Yield successive n-sized 
# chunks from l. 
def divide_chunks(l, n): 
  # looping till length l 
  for i in range(0, len(l), n):  
      yield l[i:i + n] 

# max_chunks_per_dataset = 400
# max_chunks_per_dataset = 20
max_chunks_per_dataset = config_doc["max_chunks_per_dataset"]

def build_datasets_data(primary_dataset, chunked=True):
  print("\n\n--> Building files list for Data...")
  datasets_data = None
  if primary_dataset == "MuonEG":
    datasets_data = datasets_data_MuonEG
  if primary_dataset == "SingleMuon":
    datasets_data = datasets_data_SingleMuon

  buff = []
  for d in datasets_data:
    d["files"] = file_pather(f"{config_doc['data_location']}/Data/"+d["primary_dataset"]+"/"+d["dataset_name"])
    # d["files"] = file_pather("/eos/uscms/store/user/ftorresd/HZUpsilonPhotonRun2/Data/MuonEG/"+d["dataset_name"])
    # d["files"] = file_pather("root://cmseos.fnal.gov//store/user/ftorresd/HZUpsilonPhotonRun2/Data/"+d["primary_dataset"]+"/"+d["dataset_name"])
    if d["files"] != []:
      if len(d["files"]) <= max_chunks_per_dataset:
        d["chunk"] = str(0)
        buff.append(d)
      else:
        if chunked:
          # chunks_files_list = list(divide_chunks(d["files"], max_chunks_per_dataset))
          chunks_files_list = [d["files"][i:i+max_chunks_per_dataset] for i  in range(0, len(d["files"]), max_chunks_per_dataset)]
          for index, c in enumerate(chunks_files_list):
            temp_dataset = copy.deepcopy(d)
            temp_dataset["chunk"] = str(index)
            temp_dataset["files"] = c
            buff.append(temp_dataset)
        else:
          d["chunk"] = str(0)
          buff.append(d)
  # pprint(buff)
  return buff

def build_datasets_mc(dataset_block, chunked=True):
  print("\n\n--> Building files list for MC...")
  datasets_mc = None
  if dataset_block == "analysis":
    datasets_mc = datasets_mc_analysis
  if dataset_block == "trigger":
    datasets_mc = datasets_mc_trigger

  

  buff = []
  for d in datasets_mc:
    d["files"] = file_pather(f"{config_doc['data_location']}/MC/"+d["year"]+"/"+d["dataset_name"])
    # d["files"] = file_pather("/eos/uscms/store/user/ftorresd/HZUpsilonPhotonRun2/MC/"+d["year"]+"/"+d["dataset_name"])
    # d["files"] = file_pather("root://cmseos.fnal.gov//store/user/ftorresd/HZUpsilonPhotonRun2/MC/"+d["year"]+"/"+d["dataset_name"])
    if d["files"] != []:
      if len(d["files"]) <= max_chunks_per_dataset:
        d["chunk"] = str(0)
        buff.append(d)
      else:
        if chunked:
          # chunks_files_list = list(divide_chunks(d["files"], max_chunks_per_dataset))
          chunks_files_list = [d["files"][i:i+max_chunks_per_dataset] for i  in range(0, len(d["files"]), max_chunks_per_dataset)]
          for index, c in enumerate(chunks_files_list):
            temp_dataset = copy.deepcopy(d)
            temp_dataset["chunk"] = str(index)
            temp_dataset["files"] = c
            buff.append(temp_dataset)
        else:
          d["chunk"] = str(0)
          buff.append(d)
  # pprint(buff)
  return buff

#########################
# Analysis
#########################
def get_datasets_data_analysis():
  return load("data/datasets/datasets_data_analysis.coff")

def get_datasets_mc_analysis():
  return load("data/datasets/datasets_mc_analysis.coff")

def get_datasets_data_no_chunks_analysis():
  return load("data/datasets/datasets_data_no_chunks_analysis.coff")

def get_datasets_mc_no_chunks_analysis():
  return load("data/datasets/datasets_mc_no_chunks_analysis.coff")

#########################
# Trigger
#########################
def get_datasets_data_trigger():
  return load("data/datasets/datasets_data_trigger.coff")

def get_datasets_mc_trigger():
  return load("data/datasets/datasets_mc_trigger.coff")

def get_datasets_data_no_chunks_trigger():
  return load("data/datasets/datasets_data_no_chunks_trigger.coff")

def get_datasets_mc_no_chunks_trigger():
  return load("data/datasets/datasets_mc_no_chunks_trigger.coff")


#########################
# Produce dataset objects
#########################
if __name__ == '__main__':
  save(build_datasets_data(primary_dataset="MuonEG"), "data/datasets/datasets_data_analysis.coff")
  save(build_datasets_data(primary_dataset="SingleMuon"), "data/datasets/datasets_data_trigger.coff")
  
  save(build_datasets_mc(dataset_block="analysis"), "data/datasets/datasets_mc_analysis.coff")
  save(build_datasets_mc(dataset_block="trigger"), "data/datasets/datasets_mc_trigger.coff")

  save(build_datasets_data(primary_dataset="MuonEG", chunked=False), "data/datasets/datasets_data_no_chunks_analysis.coff")
  save(build_datasets_data(primary_dataset="SingleMuon", chunked=False), "data/datasets/datasets_data_no_chunks_trigger.coff")
  
  save(build_datasets_mc(dataset_block="analysis", chunked=False), "data/datasets/datasets_mc_no_chunks_analysis.coff")
  save(build_datasets_mc(dataset_block="trigger", chunked=False), "data/datasets/datasets_mc_no_chunks_trigger.coff")


