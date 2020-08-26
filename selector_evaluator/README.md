# Selection Evaluator

## Setup

With the global ENV for the analyzer already set.

```
mkdir input_files
```

Copy the pre-selected events:
```
wget http://ftorresd.web.cern.ch/ftorresd/HZUpsilonPhotonRun2Ana_pre_selected_events.tar.gz
tar -zxvf HZUpsilonPhotonRun2Ana_pre_selected_events.tar.gz
```

## Running the evaluator

```
./ggntuples_sel_evaluator.py
```

## Running the FOM study

Open the file fom_study.ipynb using jupyter notebook.
