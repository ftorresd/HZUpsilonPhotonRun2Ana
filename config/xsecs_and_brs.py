def get_xsec(dataset):
  # REF: https://docs.google.com/spreadsheets/d/1zvvedRr1K4NFylNqxNdkWbFgoMPvh9E9BrqxywBpu-Y/edit?usp=sharing
  # define total xsecs 
  z_total_xsec = 5.7095E+04
  higgs_ggH_xsec = 4.8580E+01
  higgs_WpH_xsec = 8.4000E-01
  higgs_WmH_xsec = 5.3280E-01
  higgs_ZH_xsec = 8.8390E-01
  higgs_VBFH_xsec = 3.7820E+00
  higgs_ttH_xsec = 5.0710E-01
  higgs_total_xsec = 5.5710E+01


  # define BRs
  dy_br = 1.0
  z_1S_br = 4.8000E-08
  z_2S_br = 2.4400E-08
  z_3S_br = 1.8800E-08
  higgs_1S_br = 5.2200E-09
  higgs_2S_br = 1.4200E-09
  higgs_3S_br = 9.1000E-10
  mumugamma_fsr_br = 1.63E-6
  dalitz_decay_br = 3.8300E-05 
  upsilon_1S_to_mumu = 2.4800E-02
  upsilon_2S_to_mumu = 1.9300E-02
  upsilon_3S_to_mumu = 2.1800E-02

  effective_xsec = None
  # Z
  if dataset == "ZToUpsilon1SGamma":
    effective_xsec = z_total_xsec*z_1S_br*upsilon_1S_to_mumu

  if dataset == "ZToUpsilon2SGamma":
    effective_xsec = z_total_xsec*z_2S_br*upsilon_2S_to_mumu

  if dataset == "ZToUpsilon3SGamma":
    effective_xsec = z_total_xsec*z_3S_br*upsilon_3S_to_mumu

  # Higgs - ggH
  if dataset == "ggH_HToUps1SG":
    effective_xsec = higgs_ggH_xsec*higgs_1S_br*upsilon_1S_to_mumu

  if dataset == "ggH_HToUps2SG":
    effective_xsec = higgs_ggH_xsec*higgs_2S_br*upsilon_2S_to_mumu

  if dataset == "ggH_HToUps3SG":
    effective_xsec = higgs_ggH_xsec*higgs_3S_br*upsilon_3S_to_mumu

  # Higgs - WpH
  if dataset == "WpH_HToUps1SG":
    effective_xsec = higgs_WpH_xsec*higgs_1S_br*upsilon_1S_to_mumu

  if dataset == "WpH_HToUps2SG":
    effective_xsec = higgs_WpH_xsec*higgs_2S_br*upsilon_2S_to_mumu

  if dataset == "WpH_HToUps3SG":
    effective_xsec = higgs_WpH_xsec*higgs_3S_br*upsilon_3S_to_mumu

  # Higgs - WpH
  if dataset == "WmH_HToUps1SG":
    effective_xsec = higgs_WmH_xsec*higgs_1S_br*upsilon_1S_to_mumu

  if dataset == "WmH_HToUps2SG":
    effective_xsec = higgs_WmH_xsec*higgs_2S_br*upsilon_2S_to_mumu

  if dataset == "WmH_HToUps3SG":
    effective_xsec = higgs_WmH_xsec*higgs_3S_br*upsilon_3S_to_mumu

  # Higgs - ZH
  if dataset == "ZH_HToUps1SG":
    effective_xsec = higgs_ZH_xsec*higgs_1S_br*upsilon_1S_to_mumu

  if dataset == "ZH_HToUps2SG":
    effective_xsec = higgs_ZH_xsec*higgs_2S_br*upsilon_2S_to_mumu

  if dataset == "ZH_HToUps3SG":
    effective_xsec = higgs_ZH_xsec*higgs_3S_br*upsilon_3S_to_mumu

  # Higgs - VBFH
  if dataset == "VBFH_HToUps1SG":
    effective_xsec = higgs_VBFH_xsec*higgs_1S_br*upsilon_1S_to_mumu

  if dataset == "VBFH_HToUps2SG":
    effective_xsec = higgs_VBFH_xsec*higgs_2S_br*upsilon_2S_to_mumu

  if dataset == "VBFH_HToUps3SG":
    effective_xsec = higgs_VBFH_xsec*higgs_3S_br*upsilon_3S_to_mumu

  # Higgs - ttH
  if dataset == "ttH_HToUps1SG":
    effective_xsec = higgs_ttH_xsec*higgs_1S_br*upsilon_1S_to_mumu

  if dataset == "ttH_HToUps2SG":
    effective_xsec = higgs_ttH_xsec*higgs_2S_br*upsilon_2S_to_mumu

  if dataset == "ttH_HToUps3SG":
    effective_xsec = higgs_ttH_xsec*higgs_3S_br*upsilon_3S_to_mumu


  # peaking Backgrounds
  if dataset == "ZGTo2MuG_MMuMu-2To15":
    effective_xsec = z_total_xsec*mumugamma_fsr_br

  if dataset == "GluGluHToMuMuG_M125_MLL-0To60_Dalitz_012j":
    effective_xsec = higgs_total_xsec*dalitz_decay_br


  # drell-yan
  if dataset == "DYJetsToLL_M-50":
    effective_xsec = z_total_xsec*dy_br

  # return effective x-section
  return effective_xsec


def get_MC_weight(dataset, year, sign_weighted_total_of_events):
  # Data
  if dataset.startswith("Run"): 
    return 1.0

  # MC
  event_weight = 1.0

  # get integrated lumi
  lumi = None
  if year == "2016":
    lumi = 35.9*1E3
  if year == "2017":
    lumi = 27.13*1E3
  if year == "2018":
    lumi = 59.74*1E3

  # get cross section for the process
  xsec = get_xsec(dataset)

  event_weight = xsec*lumi/sign_weighted_total_of_events
  return event_weight



