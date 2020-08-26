# HZUpsilonPhoton - Run2 Analyzer

## Install Miniconda

Because of the CMSSW dependencies and how ROOT in brought to env, we need to install Miniconda: 

https://conda.io/projects/conda/en/latest/user-guide/install/linux.html

In summary (optimized for bash):

```
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b 
~/miniconda3/bin/conda init
echo 'conda deactivate ; conda deactivate' >> ~/.bashrc
rm Miniconda3-latest-Linux-x86_64.sh
```

It is better to reload your session.

## Setup ENV

Only once:

```
conda deactivate ; conda deactivate 
conda create -y -n HZUpsilonPhotonRun2Env python=3.6 xrootd -c conda-forge

conda activate HZUpsilonPhotonRun2Env

python3 -m pip install --upgrade pip
python3 -m pip install --upgrade --ignore-installed --force-reinstall coffea==0.6.14
python3 -m pip install --upgrade --ignore-installed --force-reinstall pick
python3 -m pip install --upgrade --ignore-installed --force-reinstall tabulate
python3 -m pip install --upgrade --ignore-installed --force-reinstall millify
python3 -m pip install --upgrade --ignore-installed --force-reinstall PyYAML
python3 -m pip install --upgrade --ignore-installed --force-reinstall celery
python3 -m pip install --upgrade --ignore-installed --force-reinstall flower
python3 -m pip install --upgrade --ignore-installed --force-reinstall redis

conda install -y -c anaconda redis
conda install -y gcc_linux-64 gxx_linux-64
python3 -m pip install --upgrade --ignore-installed --force-reinstall cppyy


export SCRAM_ARCH=slc7_amd64_gcc700 # using bash
#setenv SCRAM_ARCH "slc7_amd64_gcc700" # using csh

cmsrel CMSSW_10_2_15
cd CMSSW_10_2_15/src/
cmsenv

git clone git@github.com:mkovac/MuonMVAReader.git

scram b -j`nproc`

export PYTHONPATH=$CONDA_PREFIX/lib/python3.6/site-packages/

git clone ssh://git@gitlab.cern.ch:7999/hzupsilonphoton/hzupsilonphotonrun2ana.git HZUpsilonPhotonRun2Ana

#git clone https://gitlab.cern.ch/hzupsilonphoton/hzupsilonphotonrun2ana.git HZUpsilonPhotonRun2Ana


mkdir -p HZUpsilonPhotonRun2Ana/plots
mkdir -p HZUpsilonPhotonRun2Ana/outputs
mkdir -p HZUpsilonPhotonRun2Ana/outputs/buffer
mkdir -p HZUpsilonPhotonRun2Ana/condor_buffer
mkdir -p HZUpsilonPhotonRun2Ana/pbs_buffer
mkdir -p HZUpsilonPhotonRun2Ana/data/datasets
```

## Load ENV (after setup)

```
cd HZUpsilonPhotonRun2Ana
. quick_setup.sh
# . quick_setup.csh
```

## Running the analyzers

### Datasets

The produced [ggNtuples](https://github.com/cmkuo/ggAnalysis/tree/102X) are stored at:

- **Data:** ```root://cmseos.fnal.gov//store/user/ftorresd/HZUpsilonPhotonRun2/Data/MuonEG/```

- **MC:** ```root://cmseos.fnal.gov//store/user/ftorresd/HZUpsilonPhotonRun2/MC```

Each dataset is abstracted as a dict containg ```dataset_name```, ```is_data_or_mc```, ```year``` and ```files```. The files list is chunck in smaller subsets, in order to reduce the memory load. The chunk size can be configured (check **Configuration** section, bellow).

### Workflow

Once en ENV is setup and loaded, the canonical order of the analyzers is:

**1)** ```ggntuples_analyzer.py -b```: This one runs over all samples and filters: Events that pass the trigger selection and have at least 2 muons. The outputs (```outputs/ana_*.coffea```) contain of numpy arrays for the event-wide information, muons and photons. **This can be ran as HTCondor jobs, one per chunk. Check HTCondor section, bellow.**


```ggntuples_analyzer.py --help``` will give details on other possible configurations.

**1.1)** Once one have the, ```ana_*.coffea```, files (one per file for each dataset, stored in ```outputs/buffer```), one has to merger them. ```ggntuples_analyzer.py -m``` will do the job. Since this is very memory consuming, this may take a while... Once merged, they will be saved in ```outputs```.

**2)** ```ggntuples_selector.py ```: Object Reco/ID plus selection is defined here. The outputs (```outputs/sel_*.coffea```) contain of numpy arrays for the event-wide information, muons, photons, reconstructed upsilons and reconstructed bosons (for Higgs or Z analysis) with different sets of cuts are saved but the events are not filtered for them.

**3)** ```ggntuples_plotter.py ```: Runs over the selected events producing histograms for diferent quantities. Saves them at (```outputs/plot_*.coffea```).

**4)** ```ggntuples_harvester.py ```: Runs over the the histogram files and print plots in .png and .pdf. Saves them at (```plots/```).

**Note**: 
- ```ggntuples_genanalyzer.py ``` will run over generate level infromation and produce plots with these quantities. They will be saved at (```plots/gen/```).
- ```selector_evaluator/fom_study.ipynb ``` will run the significance (figure of merit) study to figure out the best cut values. It is a Jupiter Notebook.

### Configuration

Relevant files for configuration are:

- ```config/datasets.py```: A descripition of each MC and Data dateset of ntuples, specially, name, files path and year, plus helpers functions. It is also possible to configure the ```max_chunks_per_dataset``` for a different file chunking per dataset (default is 400 files per chunk).
-- If something has chenged in the dataset files (ggntuples), one has to rerun the dataset pickler. Execute: ```python config/datasets.py```. This will produce updated pickled datasets (```data/datasets/datasets_*.coff```).

- ```config/evtinfo.py```: A descripition ntuples content for each object (muons and photons) and event-wide information, plus helpers functions. Complete information about the ntuples content can be found at ```docs/ggntuples_content_gen.txt```.

### Pre-selection HTCondor

**This only works for LPC@FNAL cluster**:bangbang:

**1)** The file ```./condor_ggntuples_analyzer.py -sub``` will do the trick. This will submit one job per chunk. The output will be saved in a buffer zone located at ```/eos/uscms/store/user/ftorresd/condor_buffer```.

**2)** One can check status of the jobs with ```condor_q```. Once all (or some) of them are ready, ```./condor_ggntuples_analyzer.py -c``` will tell which ones finished returning 0. It would be good to also cross-check with the buffer zone. :wink:

**3)** To collect the outputs, run ```./condor_ggntuples_analyzer.py -har```, this will copy them packed outputs to the local buffer zone ```outputs/buffer``` and unpack them. The packed files (```*.tar.gz```) will be deleted.

From here on, the workflow is the same as the standard one.