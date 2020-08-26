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
  
import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt

from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

import json

from tabulate import tabulate
from millify import millify
import collections

########################################
# plot tweaks
########################################
def get_tweaks(histogram):
  tweaks = {
    'log_y' : False,
  }
  # list histograms that starts with certain pattern to be ploted in log scale 
  histograms_for_log_scale = (
    "h_leadmu_pt", 
    "h_trailmu_pt", 
    "h_photon_pt", 
    "h_boson_pt", 
    "h_upsilon_pt", 
    )
  if histogram.startswith(histograms_for_log_scale):
    tweaks['log_y'] = True

  return tweaks


########################################
# plot options
########################################
fill_opts_signal = {
    'edgecolor': (0,0,0,0.3),
    'alpha': 0.4,
    'color': 'teal',
}
fill_opts_peaking_background = {
    'edgecolor': (0,0,0,0.3),
    'alpha': 0.4,
    'color': 'gray',
}
error_opts = {
    'label':'Stat. Unc.',
    'hatch':'///',
    'facecolor':'none',
    'edgecolor':(0,0,0,.5),
    'linewidth': 0,
}
data_err_opts = {
    'linestyle':'none',
    'marker': '.',
    'markersize': 10.,
    'color':'k',
    'elinewidth': 1,
    'emarker': '_',
}


########################################
# plotter function
########################################
def print_plot(plots, density=True):
  year = plots["year"]

  ########################################
  # clear plots directory
  ########################################
  if density:
    # os.system("rm -rf plots/"+year)
    os.system("mkdir -p plots/"+year+"/au")
    os.system("mkdir -p plots/"+year+"/n_evts")
  
  # load signal
  ds_signal = processor.dict_accumulator({})
  for ds in plots["datasets_signal"]:
    ds_signal += load("outputs/plot_"+ds+"_"+year+".coffea")

  # load peaking background
  ds_peakingbkg = processor.dict_accumulator({})
  for ds in plots["datasets_peakingbkg"]:
    ds_peakingbkg += load("outputs/plot_"+ds+"_"+year+".coffea")

  # load data
  ds_data = processor.dict_accumulator({})
  for ds in plots["datasets_data"]:
    ds_data += load("outputs/plot_"+ds+"_"+year+".coffea")

  # loop over histograms and plot them 
  normalization = "Density"
  scale_signal = ""
  scale_pbkg = ""
  if density == False:
    normalization = "N. of events"
    scale_signal = " (x100)"
    scale_pbkg = ""

  for histo in tqdm(ds_data.keys(), desc=year + " - " + normalization, unit=" histograms"):
    if histo.startswith("h_"):
      # group and rename
      dataset_axis = hist.Cat("grouped_dataset", "")
      
      datasets_signal = {
          'Signal - MC'+scale_signal: [ds+"_"+year for ds in plots["datasets_signal"]],
      }
      ds_signal[histo] = ds_signal[histo].group("dataset", dataset_axis, datasets_signal)

      datasets_peakingbkg = {
          'Peaking BKG - MC'+scale_pbkg: [ds+"_"+year for ds in plots["datasets_peakingbkg"]],
      }
      ds_peakingbkg[histo] = ds_peakingbkg[histo].group("dataset", dataset_axis, datasets_peakingbkg)

      datasets_data = {
          'Data - '+ year: [ds+"_"+year for ds in plots["datasets_data"]],
      }
      ds_data[histo] = ds_data[histo].group("dataset", dataset_axis, datasets_data)

      fig, ax = plt.subplots()

      # signal
      ds_signal[histo].scale(100.)
      hist.plot1d(ds_signal[histo],
                  overlay="grouped_dataset",
                  ax=ax,
                  clear=False,
                  # stack=True,
                  line_opts=None,
                  fill_opts=fill_opts_signal,
                  # error_opts=error_opts,
                  density=density,
                  )

      # peaking background
      # ds_peakingbkg[histo].scale(3.)
      hist.plot1d(ds_peakingbkg[histo],
                  overlay="grouped_dataset",
                  ax=ax,
                  clear=False,
                  # stack=True,
                  line_opts=None,
                  fill_opts=fill_opts_peaking_background,
                  # error_opts=error_opts,
                  density=density,
                  )

      # data
      hist.plot1d(ds_data[histo],
                  overlay="grouped_dataset",
                  ax=ax,
                  clear=False,
                  error_opts=data_err_opts,
                  density=density,
                 )

      # decoration
      # add legend
      leg = ax.legend()

      coffee = plt.text(0., 1., u"CMS Preliminary",
                        fontsize=10,
                        horizontalalignment='left',
                        verticalalignment='bottom',
                        transform=ax.transAxes
                       )

      lumi = None
      if year == "2016":
        lumi = "35.9"
      if year == "2017":
        lumi = "27.13"
      if year == "2018":
        lumi = "59.74"
      lumi = plt.text(1., 1., lumi+r" fb$^{-1}$ (13 TeV)",
                      fontsize=10,
                      horizontalalignment='right',
                      verticalalignment='bottom',
                      transform=ax.transAxes
                     )
      # tweaks
      # ax.set_ylim(bottom=0.001) # clear some crowded area in the bottom of the histogram, when we have two many zeros (empty bins)
      if density:
        plt.ylabel('a.u.')
      else:
        plt.ylabel('Events (Data) / Yields (MC)')

      # log scale
      if get_tweaks(histo)['log_y']:
        plt.yscale('log')

      # save plot
      filename = "plots/"+year+"/au/"+histo+"_"+year
      if density == False:
        filename = "plots/"+year+"/n_evts/"+histo+"_"+year
      fig.savefig(filename+".png")
      fig.savefig(filename+".pdf")
      
      # clear fig , ax
      plt.cla()
      plt.clf()
      plt.close()

  # re-print plots, normalized to number of events
  if density:
    print_plot(plots, density=False)

########################################
# yields printer function
########################################
def print_yeilds(plots):
  year = plots["year"]

  ########################################
  # clear plots directory
  ########################################
  os.system("mkdir -p plots/yields/tables")
  os.system("mkdir -p plots/yields/json")
  

  cutflow = {
    "year" : year
  }

  # load signal
  for ds in plots["datasets_signal"]:
    cutflow[ds] = load("outputs/plot_"+ds+"_"+year+".coffea")["cutflow"]

  # load peaking background
  ds_peakingbkg = processor.dict_accumulator({})
  for ds in plots["datasets_peakingbkg"]:
    ds_peakingbkg += load("outputs/plot_"+ds+"_"+year+".coffea")
  cutflow["peakingbkg"] = ds_peakingbkg['cutflow']

  # load data
  ds_data = processor.dict_accumulator({})
  for ds in plots["datasets_data"]:
    ds_data += load("outputs/plot_"+ds+"_"+year+".coffea")
  cutflow["data"] = ds_data['cutflow']

  # print json
  filename = "plots/yields/json/"+year
  with open(filename+'.json', 'w') as f:
      json.dump(cutflow, f)

  # print latex table
  # REF: https://github.com/astanin/python-tabulate
  header_latex = ['', r'$Z \rightarrow \Upsilon(1S) +  \gamma$', r'$Z \rightarrow \Upsilon(2S) +  \gamma$',  r'$Z \rightarrow \Upsilon(3S) +  \gamma$', r'$Z \rightarrow \mu\mu\gamma_{FSR}$', year+r' Data']
  header_simple = ['', '1S', '2S', '3S', 'RBKG', 'Data']

  cuts = collections.OrderedDict()
  cuts['all_events'] = "Total of events"
  cuts['trigger'] = "Trigger"
  # cuts['two_recomuons'] = r"At least 2 muons")
  # cuts['reco_opposite_charge'] = "Opposite charge"
  # cuts['reco_min_lead_pt'] = r"Min. leadming muon $p_{T}$")
  cuts['reco_min_lead_pt'] = r"At least 2 muons"
  cuts['one_recophoton'] = r"At least one photon"
  cuts['dimuon_mass_cut'] = r"$8 < m_{\mu\mu} < 11$ (GeV)"
  cuts['good_bosons_count'] = r"$70 < m_{\mu\mu\gamma} < 120$ (GeV)"
  # cuts['delta_R_leading_photon'] = r"$\Delta R(lead \mu, \gamma) > 1$"
  # cuts['delta_R_trailing_photon'] = r"$\Delta R(trail \mu, \gamma) > 1$"
  # cuts['delta_R_upsilon_photon'] = r"$\Delta R(\mu\mu, \gamma) > 2$"
  # cuts['delta_Phi_leading_photon'] = r"$|\Delta\phi(\mu\mu, \gamma)| > 1.5$"
  # cuts['et_photon_over_M_boson'] = r"$E_{T}^{\gamma}/m_{\mu\mu\gamma} > 35/91.2$"
  # cuts['pt_upsilon_over_M_boson'] = r"$\Delta R(lead \mu, \gamma) > 1$ \\ $\Delta R(trail \mu, \gamma) > 1$ \\ $\Delta R(\mu\mu, \gamma) > 2$ \\ $|\Delta\phi(\mu\mu, \gamma)| > 1.5$ \\ $p_{T}^{\mu\mu}/m_{\mu\mu\gamma} > 35/91.2$ \\ $E_{T}^{\gamma}/m_{\mu\mu\gamma} > 35/91.2$ ")
  cuts['pt_upsilon_over_M_boson'] = r"Full Selection"
  # cuts['one_photon'] = "one_photon"
  # cuts['opposite_charge'] = "opposite_charge"
  # cuts['all_events_unweighted'] = "all_events_unweighted"
  # cuts['two_muons'] = "two_muons"
        
  data = []
  for c in cuts.keys():
    line = []
    line.append(cuts[c])
    for ds in plots["datasets_signal"]:
      line.append(millify(cutflow[ds][c], precision=3, drop_nulls=False))
    line.append(millify(cutflow["peakingbkg"][c], precision=1, drop_nulls=False))
    if c == "good_bosons_count" or  c == "pt_upsilon_over_M_boson":
      line.append("XXX")
    else:
      line.append(millify(cutflow["data"][c], precision=0, drop_nulls=True))
    data.append(line)

  # print latex
  filename = "plots/yields/tables/"+year
  with open(filename+'.latex', 'w') as f:
      # f.write(tabulate(data, headers=header_latex, tablefmt="latex_raw", floatfmt=(".2E", ".2E", ".2E", ".2E", ".2E")))
      f.write(tabulate(data, headers=header_latex, tablefmt="latex_raw"))
  
  print("###################### " + year + " ######################")
  # print(tabulate(data, headers=header_simple, tablefmt="simple", floatfmt=(".4E", ".4E", ".4E", ".4E", ".0f")))
  print(tabulate(data, headers=header_simple, tablefmt="simple"))
  # print(tabulate(data, headers=header_simple, tablefmt="simple",))
  print("\n\n")

########################################
# define plots
########################################
plots_set = [
# Z - 2016
{
  "year" : "2016",
  "datasets_signal" : ["ZToUpsilon1SGamma", "ZToUpsilon2SGamma", "ZToUpsilon3SGamma", ],
  "datasets_peakingbkg" : ["ZGTo2MuG_MMuMu-2To15"],
  "datasets_data" : [
            "Run2016B-17Jul2018_ver1",
            "Run2016B-17Jul2018_ver2",
            "Run2016C-17Jul2018",
            "Run2016D-17Jul2018",
            "Run2016E-17Jul2018",
            "Run2016F-17Jul2018",
            "Run2016G-17Jul2018",
            "Run2016H-17Jul2018",
            ],
},

# Z - 2017 - Mu_Photon Trigger
{
  "year" : "2017",
  "datasets_signal" : ["ZToUpsilon1SGamma", "ZToUpsilon2SGamma", "ZToUpsilon3SGamma", ],
  "datasets_peakingbkg" : ["ZGTo2MuG_MMuMu-2To15"],
  "datasets_data" : [
            # "Run2017B-31Mar2018-v1",
            # "Run2017C-31Mar2018-v1",
            "Run2017D-31Mar2018-v1",
            "Run2017E-31Mar2018-v1",
            "Run2017F-31Mar2018-v1",
            ],
},

# Z - 2018
{
  "year" : "2018",
  "datasets_signal" : ["ZToUpsilon1SGamma", "ZToUpsilon2SGamma", "ZToUpsilon3SGamma", ],
  "datasets_peakingbkg" : ["ZGTo2MuG_MMuMu-2To15"],
  "datasets_data" : [
            "Run2018A-17Sep2018-v1",
            "Run2018B-17Sep2018-v1",
            "Run2018C-17Sep2018-v1",
            "Run2018D-PromptReco-v2",
            ],
},
]


if __name__ == '__main__':
  import time
  start_time = time.time()

  print("--> Starting ---")

  ########################################
  # loop over plots and print them
  ########################################
  os.system("rm -rf plots/*")

  for p in plots_set:
    print_plot(p)

  for p in plots_set:
    print_yeilds(p)

  print("--> Done in %s seconds ---" % datetime.timedelta(seconds=(time.time() - start_time)))

  