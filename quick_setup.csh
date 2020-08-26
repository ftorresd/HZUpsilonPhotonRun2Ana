cd ..
conda deactivate ; conda deactivate 
conda activate HZUpsilonPhotonRun2Env
cmsenv
export PYTHONPATH=$CONDA_PREFIX/lib/python3.6/site-packages/
source /cvmfs/sft.cern.ch/lcg/releases/LCG_92python3/ROOT/6.12.04/x86_64-centos7-gcc7-opt/bin/thisroot.csh
cd HZUpsilonPhotonRun2Ana
voms-proxy-init --rfc --voms cms