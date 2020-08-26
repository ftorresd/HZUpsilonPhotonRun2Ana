from coffea.analysis_objects import JaggedCandidateArray
import numpy as np
from awkward import JaggedArray


def CutBasedIdTight_PFIsoTight(muons, year):
  _probe_muon_good_pt = muons.pt > 3 # pT > 3 GeV
  _probe_muon_good_abs_eta = np.absolute(muons.eta) < 2.4 # |eta| < 2.4
  # CutBasedIdSoft - 13
  # CutBasedIdTight - 3
  # PFIsoTight - 9
  _probe_muon_good_id = np.bitwise_and(np.right_shift(muons["muIDbit"], 3), 1).astype(bool) 
  _probe_muon_good_iso = np.bitwise_and(np.right_shift(muons["muIDbit"], 9), 1).astype(bool) # PFIsoTight
  # _probe_muon_good_dxy = (np.absolute(muons['muD0']) < 0.5)
  # _probe_muon_good_dz = (np.absolute(muons['muDz']) < 1.0)
  _is_good_probe_muon = (_probe_muon_good_pt & _probe_muon_good_abs_eta & _probe_muon_good_id & _probe_muon_good_iso) 
  # _is_good_probe_muon = (_probe_muon_good_pt & _probe_muon_good_abs_eta & _probe_muon_good_id) 
  muons = muons[_is_good_probe_muon]
  return muons


def CutBasedIdSoft_noISO(muons, year):
  _probe_muon_good_pt = muons.pt > 3 # pT > 3 GeV
  _probe_muon_good_abs_eta = np.absolute(muons.eta) < 2.4 # |eta| < 2.4
  # CutBasedIdSoft - 13
  # CutBasedIdTight - 3
  # PFIsoTight - 9
  _probe_muon_good_id = np.bitwise_and(np.right_shift(muons["muIDbit"], 13), 1).astype(bool) 
  # _probe_muon_good_iso = np.bitwise_and(np.right_shift(muons["muIDbit"], 9), 1).astype(bool) # PFIsoTight
  # _probe_muon_good_dxy = (np.absolute(muons['muD0']) < 0.5)
  # _probe_muon_good_dz = (np.absolute(muons['muDz']) < 1.0)
  # _is_good_probe_muon = (_probe_muon_good_pt & _probe_muon_good_abs_eta & _probe_muon_good_id & _probe_muon_good_iso) 
  _is_good_probe_muon = (_probe_muon_good_pt & _probe_muon_good_abs_eta & _probe_muon_good_id) 
  muons = muons[_is_good_probe_muon]
  return muons


def CutBasedIdSoft_PFIsoMedium(muons, year):
  _probe_muon_good_pt = muons.pt > 3 # pT > 3 GeV
  _probe_muon_good_abs_eta = np.absolute(muons.eta) < 2.4 # |eta| < 2.4
  # CutBasedIdSoft - 13
  # CutBasedIdTight - 3
  # PFIsoTight - 9
  # PFIsoMedium - 8
  _probe_muon_good_id = np.bitwise_and(np.right_shift(muons["muIDbit"], 13), 1).astype(bool) 
  _probe_muon_good_iso = np.bitwise_and(np.right_shift(muons["muIDbit"], 8), 1).astype(bool) # PFIsoTight
  # _probe_muon_good_dxy = (np.absolute(muons['muD0']) < 0.5)
  # _probe_muon_good_dz = (np.absolute(muons['muDz']) < 1.0)
  _is_good_probe_muon = (_probe_muon_good_pt & _probe_muon_good_abs_eta & _probe_muon_good_id & _probe_muon_good_iso) 
  # _is_good_probe_muon = (_probe_muon_good_pt & _probe_muon_good_abs_eta & _probe_muon_good_id) 
  muons = muons[_is_good_probe_muon]
  return muons



def CutBasedIdSoft_PFIsoLoose(muons, year):
  _probe_muon_good_pt = muons.pt > 3 # pT > 3 GeV
  _probe_muon_good_abs_eta = np.absolute(muons.eta) < 2.4 # |eta| < 2.4
  # CutBasedIdSoft - 13
  # CutBasedIdTight - 3
  # PFIsoTight - 9
  # PFIsoMedium - 8
  # PFIsoLoose - 7
  _probe_muon_good_id = np.bitwise_and(np.right_shift(muons["muIDbit"], 13), 1).astype(bool) 
  _probe_muon_good_iso = np.bitwise_and(np.right_shift(muons["muIDbit"], 7), 1).astype(bool) # PFIsoTight
  # _probe_muon_good_dxy = (np.absolute(muons['muD0']) < 0.5)
  # _probe_muon_good_dz = (np.absolute(muons['muDz']) < 1.0)
  _is_good_probe_muon = (_probe_muon_good_pt & _probe_muon_good_abs_eta & _probe_muon_good_id & _probe_muon_good_iso) 
  # _is_good_probe_muon = (_probe_muon_good_pt & _probe_muon_good_abs_eta & _probe_muon_good_id) 
  muons = muons[_is_good_probe_muon]
  return muons


def HZZ4l_MVABAsed(muons, year):
  # muon types
  # REF: https://github.com/cms-sw/cmssw/blob/CMSSW_9_4_X/DataFormats/MuonReco/interface/Muon.h#L284-L291
  # GlobalMuon --> 1;
  # TrackerMuon --> 2;
  # StandAloneMuon --> 3;
  # CaloMuon --> 4;
  # PFMuon --> 5;
  # RPCMuon --> 6; 
  # GEMMuon --> 7;
  # ME0Muon --> 8;
  # print (muons.pt.flatten().shape if ((dataset == "ZToUpsilon1SGamma" and year == "2016") or (dataset == "Run2016B-17Jul2018_ver")) else "") 
  # Loose Muons
  _loose_muon_good_min_pt = (muons.pt > 5.0)
  _loose_muon_good_eta = (np.absolute(muons.eta) < 2.4)
  _loose_muon_good_dxy = (np.absolute(muons['muD0']) < 0.5)
  _loose_muon_good_dz = (np.absolute(muons['muDz']) < 1.0)
  _loose_muon_good_global_muon = (np.bitwise_and(np.right_shift(muons["muType"].astype(int), 1), 1).astype(bool))
  _loose_muon_good_tracker_muon = (np.bitwise_and(np.right_shift(muons["muType"].astype(int), 2), 1).astype(bool))
  _loose_muon_good_mu_matches = (muons['muMatches'] > 0)
  _loose_muon_good_best_track_type = (muons['muBestTrkType'] != 2)

  _muon_good_loose_muons = (_loose_muon_good_min_pt & _loose_muon_good_min_pt & _loose_muon_good_eta & _loose_muon_good_dxy & _loose_muon_good_dz & ((_loose_muon_good_global_muon | (_loose_muon_good_tracker_muon & _loose_muon_good_mu_matches)) & _loose_muon_good_best_track_type))

  # Tracker High Pt
  _tracker_high_pt_muon_good_mu_matches = (muons['muMatches'] > 1)
  _tracker_high_pt_muon_good_rel_pt_error = (muons['muBestTrkPtError']/muons['muBestTrkPt'] < 0.3)
  _tracker_high_pt_muon_good_dxy = (np.absolute(muons['muD0']) < 0.2)
  _tracker_high_pt_muon_good_dz = (np.absolute(muons['muDz']) < 0.5)
  _tracker_high_pt_muon_good_pixel_hits = (muons['muPixelHits'] > 0)
  _tracker_high_pt_muon_good_trk_layers = (muons['muTrkLayers'] > 5)

  _muon_good_tracker_high_pt = (_tracker_high_pt_muon_good_mu_matches  & _tracker_high_pt_muon_good_rel_pt_error  & _tracker_high_pt_muon_good_dxy & _tracker_high_pt_muon_good_dz & _tracker_high_pt_muon_good_pixel_hits & _tracker_high_pt_muon_good_trk_layers)

  # Tight Muons
  _tight_muon_good_high_pt = (muons.pt > 200.0)
  _tight_muon_good_bdt = (muons.pt < 0.0)
  if year == "2016":
      _tight_muon_good_bdt = (((muons.pt <= 10.0) & (muons['bdt_score'] > 0.8847169876098633)) | ((muons.pt > 10)  & (muons['bdt_score'] > -0.19389629721641488)))
  if year == "2017":
      _tight_muon_good_bdt = (((muons.pt <= 10.0) & (muons['bdt_score'] > 0.883555161952972)) | ((muons.pt > 10)  & (muons['bdt_score'] > -0.3830992293357821)))
  if year == "2018":
      _tight_muon_good_bdt = (((muons.pt <= 10.0) & (muons['bdt_score'] > 0.9506129026412962)) | ((muons.pt > 10)  & (muons['bdt_score'] > -0.3629065185785282)))

  _is_good_muon_tight = (_muon_good_loose_muons & (_tight_muon_good_bdt | (_muon_good_tracker_high_pt & _tight_muon_good_high_pt)))

  muons = muons[_is_good_muon_tight]

  return muons
