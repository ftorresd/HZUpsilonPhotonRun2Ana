import numpy as np

muon_vars = ["muPt", "muEn", "muEta", "muPhi", "muCharge", "muType", "muIDbit", "muD0", "muDz", "muSIP", "muChi2NDF", "muInnerD0", "muInnerDz", "muTrkLayers", "muPixelLayers", "muPixelHits", "muMuonHits", "muStations", "muMatches", "muTrkQuality", "muIsoTrk", "muPFChIso", "muPFPhoIso", "muPFNeuIso", "muPFPUIso", "muPFChIso03", "muPFPhoIso03", "muPFNeuIso03", "muPFPUIso03", "muFiredTrgs", "muFiredL1Trgs", "muInnervalidFraction", "musegmentCompatibility", "muchi2LocalPosition", "mutrkKink", "muBestTrkPtError", "muBestTrkPt", "muBestTrkType"]
photon_vars = ["phoE", "phoSigmaE", "phoEt", "phoEta", "phoPhi", "phoCalibE", "phoCalibEt", "phoSCE", "phoSCRawE", "phoESEnP1", "phoESEnP2", "phoSCEta", "phoSCPhi", "phoSCEtaWidth", "phoSCPhiWidth", "phoSCBrem", "phohasPixelSeed", "phoEleVeto", "phoR9", "phoHoverE", "phoESEffSigmaRR", "phoSigmaIEtaIEtaFull5x5", "phoSigmaIEtaIPhiFull5x5", "phoSigmaIPhiIPhiFull5x5", "phoE2x2Full5x5", "phoE5x5Full5x5", "phoR9Full5x5", "phoPFChIso", "phoPFPhoIso", "phoPFNeuIso", "phoPFChWorstIso", "phoIDMVA", "phoFiredSingleTrgs", "phoFiredDoubleTrgs", "phoFiredTripleTrgs", "phoFiredL1Trgs", "phoSeedTime", "phoSeedEnergy", "phoxtalBits", "phoIDbit", "phoScale_stat_up", "phoScale_stat_dn", "phoScale_syst_up", "phoScale_syst_dn", "phoScale_gain_up", "phoScale_gain_dn", "phoResol_rho_up", "phoResol_rho_dn", "phoResol_phi_up", "phoResol_phi_dn"]
evt_vars = ["run", "event", "lumis", "isData", "nVtx", "nGoodVtx", "isPVGood", "vtx", "vty", "vtz", "rho", "rhoCentral", "L1ECALPrefire", "L1ECALPrefireUp", "L1ECALPrefireDown", "HLTEleMuX", "HLTPho", "HLTPhoRejectedByPS", "HLTJet", "HLTEleMuXIsPrescaled", "HLTPhoIsPrescaled", "HLTJetIsPrescaled", "pdf", "pthat", "processID", "genWeight", "genHT", "genPho1", "genPho2", "EventTag", "nPUInfo", "nPU", "puBX", "puTrue"]

def get_vars_dict(dataframe, vars_list, get_from_acc = False):
  dict = {}
  for v in vars_list:
    try:
      if get_from_acc:
        dict[v] = dataframe[v].value
      else:
        if dataframe[v].size == 0:
          dict[v] = np.array([])
        else:
          dict[v] = dataframe[v]
    except:
      pass
  return dict