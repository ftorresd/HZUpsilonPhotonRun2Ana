if [[ $_ == $0 ]]; then  
  echo "This script is meant to be sourced:"
  echo "  source $0"
  exit 0
fi

cd ..
conda deactivate ; conda deactivate 
conda activate HZUpsilonPhotonRun2Env
cmsenv
export PYTHONPATH=$CONDA_PREFIX/lib/python3.6/site-packages/
source /cvmfs/sft.cern.ch/lcg/releases/LCG_92python3/ROOT/6.12.04/x86_64-centos7-gcc7-opt/bin/thisroot.sh # for bash
cd HZUpsilonPhotonRun2Ana
voms-proxy-init --rfc --voms cms