from coffea import hist
from coffea.analysis_objects import JaggedCandidateArray
import coffea.processor as processor
from coffea import hist
import numpy as np
from awkward import JaggedArray
from pprint import pprint as pprint
from coffea.util import load, save
import awkward
import config
import cppyy
import utils
import uproot
from tabulate import tabulate
import glob
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

import yaml
config_doc = yaml.load(open("config/config.yaml", "r"), Loader=yaml.FullLoader)

import utils.accumulator_merger
import utils.muon_identifiers


# Look at ProcessorABC to see the expected methods and what they are supposed to do
class TriggerSelectorProcessor(processor.ProcessorABC):
  def __init__(self,):
    ####################################
    # accumulator
    ####################################
    self._accumulator = processor.dict_accumulator({     
        # weight per triplet
        'triplet_weight': processor.column_accumulator(np.array([])),

        # tag muon
        'tag_muon_pt': processor.column_accumulator(np.array([])),
        'tag_muon_eta': processor.column_accumulator(np.array([])),
        'tag_muon_phi': processor.column_accumulator(np.array([])),

        # probe muon
        'probe_muon_pt': processor.column_accumulator(np.array([])),
        'probe_muon_eta': processor.column_accumulator(np.array([])),
        'probe_muon_phi': processor.column_accumulator(np.array([])),        

        # photon
        'probe_photon_et': processor.column_accumulator(np.array([])),
        'probe_photon_eta': processor.column_accumulator(np.array([])),
        'probe_photon_phi': processor.column_accumulator(np.array([])),    
        # 'photon_energy': processor.column_accumulator(np.array([])),    
        # 'photon_calibEnergy': processor.column_accumulator(np.array([])),    
        # 'photon_calibEt': processor.column_accumulator(np.array([])),    
        # 'photon_etasc': processor.column_accumulator(np.array([])),   
        
        # passing probe flag
        'good_probe': processor.column_accumulator(np.array([])),
        'good_probe_muon': processor.column_accumulator(np.array([])),
        'good_probe_photon': processor.column_accumulator(np.array([])),

        # cutflow
        'cutflow': processor.defaultdict_accumulator(float),
    })

  @property
  def accumulator(self):
    return self._accumulator

  def looper(self, file_data):
    # clean accumulator
    output_buff = self.accumulator.identity()

    # configure looper
    dataset = file_data[0]
    year = file_data[1]
    data_or_mc = file_data[2]
    file_name = file_data[3]
    total_sign_weight_events = file_data[4]

    # load dataframe
    df = load(file_name)

    ####################################
    # load cutflow
    ####################################
    output_buff['cutflow'] += df['cutflow']
    ####################################
    # load objects
    ####################################

    if df['nMu'].value.size > 0: # process only datasets with at least one event selected
        #event info
        event_info = awkward.Table(**config.get_vars_dict(df, config.evt_vars, get_from_acc=True))
        event_info['one_photon'] = df['one_photon'].value
        event_info['evtWeight_sign'] = df['evtWeight_sign'].value

        # define total event weight
        # initiate this array with MC xSec weights
        event_info['total_weight'] = np.ones(df['evtWeight_sign'].value.size).astype(float)
        # total_sign_weight_events = output_buff['cutflow']['all_events']
        event_info['total_weight'] = event_info['total_weight']*config.get_MC_weight(dataset, year, total_sign_weight_events)

        # get pu weight
        if not(dataset.startswith("Run")):
            weight_pu_nominal, weight_pu_up, weight_pu_down  = utils.get_pu_weight(event_info['puTrue'], dataset, year)
            event_info['total_weight'] = event_info['total_weight']*weight_pu_nominal

        # correct the total weighted and triggered events by the x-sec
        output_buff['cutflow']['all_events'] = output_buff['cutflow']['all_events']*config.get_MC_weight(dataset, year, total_sign_weight_events)
        output_buff['cutflow']['trigger'] = output_buff['cutflow']['trigger']*config.get_MC_weight(dataset, year, total_sign_weight_events)
        output_buff['cutflow']['two_muons'] = output_buff['cutflow']['two_muons']*config.get_MC_weight(dataset, year, total_sign_weight_events)
        output_buff['cutflow']['opposite_charge'] = output_buff['cutflow']['opposite_charge']*config.get_MC_weight(dataset, year, total_sign_weight_events)
        output_buff['cutflow']['one_photon'] = output_buff['cutflow']['one_photon']*config.get_MC_weight(dataset, year, total_sign_weight_events)
        
        # correct 
        event_info['total_weight'] = event_info['total_weight']*event_info['evtWeight_sign']


        ####################################
        # muon rochester corrections
        ####################################
        # load rochester correctors
        if year == "2016":
            from cppyy.gbl import rc2016 as rc
        if year == "2017":
            from cppyy.gbl import rc2017 as rc
        if year == "2018":
            from cppyy.gbl import rc2018 as rc

        get_muon_scale_factor = None
        muon_scale_factor = None

        # correction if it is MC
        if data_or_mc == "mc":
            get_muon_scale_factor = lambda a : rc.kSmearMC(int(a[0]),a[1],a[2],a[3],int(a[4]),a[5]) # rc.kSmearMC(charge, pt, eta, phi, "number of trackerLayersWithMeasurement", "random number distributed uniformly between 0 and 1")
            muon_scale_factor = np.apply_along_axis(get_muon_scale_factor, 0, 
            np.vstack((
                df['muCharge'].value,
                df['muPt'].value,
                df['muEta'].value,
                df['muPhi'].value,
                df['muTrkLayers'].value,
                np.random.uniform(0,1,df['muPt'].value.size),
                ))
            )
        # correction if it is Data
        else:
            get_muon_scale_factor = lambda a : rc.kScaleDT(int(a[0]),a[1],a[2],a[3]) # rc->kScaleDT(charge, pt, eta, phi);
            muon_scale_factor = np.apply_along_axis(get_muon_scale_factor, 0, 
                np.vstack((
                    df['muCharge'].value,
                    df['muPt'].value,
                    df['muEta'].value,
                    df['muPhi'].value,
                    ))
                )

        # get the corrected muon pt
        # muon_corrected_pt = df['muPt'].value*muon_scale_factor
        muon_corrected_pt = df['muPt'].value

        # load muons
        muons = JaggedCandidateArray.candidatesfromcounts(
            df['nMu'].value,
            # pt=df['muPt'].value,
            pt=muon_corrected_pt,
            eta=df['muEta'].value,
            phi=df['muPhi'].value,
            mass=105.6583745/1000.,
            **config.get_vars_dict(df, config.muon_vars, get_from_acc=True)
            )

        # load muond BDT score
        if year == "2016":
            from cppyy.gbl import reader_2016 as reader
        if year == "2017":
            from cppyy.gbl import reader_2017 as reader
        if year == "2018":
            from cppyy.gbl import reader_2018 as reader

        # MVA reader arguments - in this order
        # pt
        # eta
        # mu_N_hits_ 
        # mu_N_pixel_hits_
        # mu_N_tracker_hits_
        # mu_chi_square_
        # PFPhotonIso
        # PFChargedHadIso
        # PFNeutralHadIso
        # rho
        # SIP
        # dxy
        # dz
        
        get_muon_MVA_score = lambda a : reader.Get_MVA_value(a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7],a[8],a[9],a[10],a[11],a[12])
        rho_for_BDT = np.repeat(event_info['rho'].flatten(), muons.counts)
        muons_bdt_score = np.apply_along_axis(get_muon_MVA_score, 0, 
            np.vstack((
                muons.pt.flatten(),
                muons.eta.flatten(),
                muons['muMuonHits'].flatten(),
                muons['muPixelHits'].flatten(),
                muons['muTrkLayers'].flatten(),
                muons['muChi2NDF'].flatten(),
                muons['muPFPhoIso03'].flatten(),
                muons['muPFChIso03'].flatten(),
                muons['muPFNeuIso03'].flatten(),
                rho_for_BDT, 
                muons['muSIP'].flatten(),
                np.absolute(muons['muD0'].flatten()),
                np.absolute(muons['muDz'].flatten()),
                ))
            )
        # pprint(muons_bdt_score)
        # pprint(muons_bdt_score.shape)
        # pprint(muons.shape)
        # pprint(muons.pt.flatten().shape)
        muons.add_attributes(bdt_score=muons_bdt_score)
        # pprint(photons.offsets())
        # muons.add_attributes(bdt_score=np.ones(muons.pt.flatten().shape[0]))


        ####################################
        # muon fired trigger selection
        # muFiredTrgs - BITS
        ####################################

        # Tag Triggers
        # BIT: 1 - HLT_IsoMu24
        # BIT: 17 - HLT_IsoMu27
        # BIT: 18 - HLT_IsoTkMu22
        # BIT: 19 - HLT_IsoTkMu24
        # BIT: 20 - HLT_IsoTkMu27
        # BIT: 30 - HLT_IsoMu24 (2017)
        # BIT: 31 - HLT_IsoMu27 (2017)

        # if fired HLT_IsoMu24 || HLT_IsoTkMu24 - 2016
        tag_muon_fired_trigger_bit_2016_IsoMu24 =  1
        tag_muon_fired_trigger_2016_IsoMu24 = np.bitwise_and(np.right_shift(muons["muFiredTrgs"], tag_muon_fired_trigger_bit_2016_IsoMu24), 1).astype(bool) 

        tag_muon_fired_trigger_bit_2016_IsoTkMu24 =  19
        tag_muon_fired_trigger_2016_IsoTkMu24 = np.bitwise_and(np.right_shift(muons["muFiredTrgs"], tag_muon_fired_trigger_bit_2016_IsoTkMu24), 1).astype(bool) 

        # if fired HLT_IsoMu27 || HLT_IsoTkMu27 - 2017
        tag_muon_fired_trigger_bit_2017_IsoMu27 =  31
        tag_muon_fired_trigger_2017_IsoMu27 = np.bitwise_and(np.right_shift(muons["muFiredTrgs"], tag_muon_fired_trigger_bit_2017_IsoMu27), 1).astype(bool) 

        tag_muon_fired_trigger_bit_2017_IsoTkMu27 =  20
        tag_muon_fired_trigger_2017_IsoTkMu27 = np.bitwise_and(np.right_shift(muons["muFiredTrgs"], tag_muon_fired_trigger_bit_2017_IsoTkMu27), 1).astype(bool) 

        # if fired HLT_IsoMu24 || HLT_IsoTkMu24 - 2018
        tag_muon_fired_trigger_bit_2018_IsoMu24 =  30
        tag_muon_fired_trigger_2018_IsoMu24 = np.bitwise_and(np.right_shift(muons["muFiredTrgs"], tag_muon_fired_trigger_bit_2018_IsoMu24), 1).astype(bool) 

        tag_muon_fired_trigger_bit_2018_IsoTkMu24 =  19
        tag_muon_fired_trigger_2018_IsoTkMu24 = np.bitwise_and(np.right_shift(muons["muFiredTrgs"], tag_muon_fired_trigger_bit_2018_IsoTkMu24), 1).astype(bool) 

        # # if fired HLT_IsoMu27 - 2017 or 2018
        # tag_muon_fired_trigger_bit_2017_or_2018 =  31
        # tag_muon_fired_trigger_2017_or_2018 = np.bitwise_and(np.right_shift(muons["muFiredTrgs"], tag_muon_fired_trigger_bit_2017_or_2018), 1).astype(bool) 

        # merge the results from all years and add as atribute of the muons collection
        tag_muon_fired_trigger_bit = tag_muon_fired_trigger_2016_IsoMu24 | tag_muon_fired_trigger_2016_IsoTkMu24
        if year == "2017":
            tag_muon_fired_trigger_bit = tag_muon_fired_trigger_2017_IsoMu27 | tag_muon_fired_trigger_2017_IsoTkMu27
        
        if year == "2018":
            tag_muon_fired_trigger_bit = tag_muon_fired_trigger_2018_IsoMu24 | tag_muon_fired_trigger_2018_IsoTkMu24

        muons.add_attributes(tag_muon_fired_trigger_bit=tag_muon_fired_trigger_bit.flatten())

        # Signal Triggers
        # 2016
        # The filters checked for the muon and photon legs of the trigger are different
        # between runs B to E and F to H.

        # -- For runs B,C,D,and E:
        # muon leg: hltL3fL1sL1Mu5IsoEG18L1f5L2f7L3Filtered17 
        # photon leg: hltMu17Photon30CaloIdLL1ISOHEFilter
        # BIT: 2 - HLT_Mu17_Photon30_CaloIdL_L1ISO_v*

        # -- For runs F, G and H, as well as for the MC samples:
        # muon leg: hltL3fL1sL1Mu5IsoEG18ORL1Mu5IsoEG20L1f5L2f7L3Filtered17 
        # photon leg: hltMu17Photon30CaloIdLL1ISOORHEFilter
        # BIT: 21 - HLT_Mu17_Photon30_CaloIdL_L1ISO_v*

        # 2017 and 2018
        # muon leg: hltMu17Photon30IsoCaloIdMuonlegL3Filtered17Q
        # photon leg : hltMu17Photon30IsoCaloIdPhotonlegTrackIsoFilter
        # BIT: 0 - HLT_Mu17_Photon30_IsoCaloId_v* - muon leg

        # HLT_DoubleMu20_Pho23 - ?????
        # BIT: 29 - HLT_DoubleMu20_Pho23* - muon leg ????

        # if fired HLT_Mu17_Photon30_CaloIdL_L1ISO_v - 2016 - runs B,C,D,and E
        probe_muon_fired_trigger_bit_2016BCDE =  2
        probe_muon_fired_trigger_2016BCDE = np.bitwise_and(np.right_shift(muons["muFiredTrgs"], probe_muon_fired_trigger_bit_2016BCDE), 1).astype(bool) 
        # print("\nprobe_muon_fired_trigger_2016BCDE:")
        # pprint(probe_muon_fired_trigger_2016BCDE.flatten().sum())

        # if fired HLT_Mu17_Photon30_CaloIdL_L1ISO_v - 2016 - runs F, G and H, or for the MC samples
        probe_muon_fired_trigger_bit_2016GFH_or_MC =  21
        probe_muon_fired_trigger_2016GFH_or_MC = np.bitwise_and(np.right_shift(muons["muFiredTrgs"], probe_muon_fired_trigger_bit_2016GFH_or_MC), 1).astype(bool) 
        # print("\nprobe_muon_fired_trigger_2016GFH_or_MC:")
        # pprint(probe_muon_fired_trigger_2016GFH_or_MC.flatten().sum())

        # if fired HLT_Mu17_Photon30_IsoCaloId_v* - 2017 or 2018
        probe_muon_fired_trigger_bit_2017_or_2018 =  0
        probe_muon_fired_trigger_2017_or_2018 = np.bitwise_and(np.right_shift(muons["muFiredTrgs"], probe_muon_fired_trigger_bit_2017_or_2018), 1).astype(bool) 
        # print("\nprobe_muon_fired_trigger_2017_or_2018:")
        # pprint(probe_muon_fired_trigger_2017_or_2018.flatten().sum())

        # merge the results from all years and add as atribute of the muons collection
        probe_muon_fired_trigger_bit = probe_muon_fired_trigger_2016BCDE 
        
        if dataset.startswith("Run2016F") or dataset.startswith("Run2016G") or dataset.startswith("Run2016H") or not(dataset.startswith("Run")):
            probe_muon_fired_trigger_bit = probe_muon_fired_trigger_2016GFH_or_MC 
        
        if year == "2017" or year == "2018":
            probe_muon_fired_trigger_bit = probe_muon_fired_trigger_2017_or_2018
        
        muons.add_attributes(probe_muon_fired_trigger_bit=probe_muon_fired_trigger_bit)


        # load photons
        photons = JaggedCandidateArray.candidatesfromcounts(
            df['nPho'].value,
            pt=df['phoCalibEt'].value,
            eta=df['phoEta'].value,
            phi=df['phoPhi'].value,
            energy=df['phoCalibE'].value,
            **config.get_vars_dict(df, config.photon_vars, get_from_acc=True)
            )


        ####################################
        # photon fired trigger selection
        # phoFiredDoubleTrgs - BITS
        ####################################

        # 2016
        # -- For runs B,C,D,and E:
        # muon leg: hltL3fL1sL1Mu5IsoEG18L1f5L2f7L3Filtered17 
        # photon leg: hltMu17Photon30CaloIdLL1ISOHEFilter
        # BIT: 28 - HLT_Mu17_Photon30_CaloIdL_L1ISO_v*

        # -- For runs F, G, and H, as well as for the MC sample:
        # muon leg: hltL3fL1sL1Mu5IsoEG18ORL1Mu5IsoEG20L1f5L2f7L3Filtered17 
        # photon leg: hltMu17Photon30CaloIdLL1ISOORHEFilter
        # BIT: 29 - HLT_Mu17_Photon30_CaloIdL_L1ISO_v*

        # 2017 and 2018
        # muon leg: hltMu17Photon30IsoCaloIdMuonlegL3Filtered17Q
        # photon leg : hltMu17Photon30IsoCaloIdPhotonlegTrackIsoFilter
        # BIT: 27 - HLT_Mu17_Photon30_IsoCaloId_v - photon leg

        # HLT_DoubleMu20_Pho23 - ?????
        # BIT: ????? - HLT_DoubleMu20_Pho23* - muon leg ????

        # if fired HLT_Mu17_Photon30_CaloIdL_L1ISO_v - 2016 - runs B,C,D,and E
        probe_photon_fired_trigger_bit_2016BCDE =  28
        probe_photon_fired_trigger_2016BCDE = np.bitwise_and(np.right_shift(photons["phoFiredDoubleTrgs"], probe_photon_fired_trigger_bit_2016BCDE), 1).astype(bool) 
        # print("\nprobe_photon_fired_trigger_2016BCDE:")
        # pprint(probe_photon_fired_trigger_2016BCDE.flatten().sum())

        # if fired HLT_Mu17_Photon30_CaloIdL_L1ISO_v - 2016 - runs F, G and H, or for the MC samples
        probe_photon_fired_trigger_bit_2016GFH_or_MC =  29
        probe_photon_fired_trigger_2016GFH_or_MC = np.bitwise_and(np.right_shift(photons["phoFiredDoubleTrgs"], probe_photon_fired_trigger_bit_2016GFH_or_MC), 1).astype(bool) 
        # print("\nprobe_photon_fired_trigger_2016GFH_or_MC:")
        # pprint(probe_photon_fired_trigger_2016GFH_or_MC.flatten().sum())

        # if fired HLT_Mu17_Photon30_IsoCaloId_v* - 2017 or 2018
        probe_photon_fired_trigger_bit_2017_or_2018 =  27
        probe_photon_fired_trigger_2017_or_2018 = np.bitwise_and(np.right_shift(photons["phoFiredDoubleTrgs"], probe_photon_fired_trigger_bit_2017_or_2018), 1).astype(bool) 
        # print("\nprobe_photon_fired_trigger_2017_or_2018:")
        # pprint(probe_photon_fired_trigger_2017_or_2018.flatten().sum())

        # # merge the results from all years and add as atribute of the photons collection
        # probe_photon_fired_trigger_bit = (probe_photon_fired_trigger_2016BCDE) | (probe_photon_fired_trigger_2016GFH_or_MC) | (probe_photon_fired_trigger_2017_or_2018)
        # photons.add_attributes(probe_photon_fired_trigger_bit=probe_photon_fired_trigger_bit)

        # merge the results from all years and add as atribute of the muons collection
        probe_photon_fired_trigger_bit = probe_photon_fired_trigger_2016BCDE 
        
        if dataset.startswith("Run2016F") or dataset.startswith("Run2016G") or dataset.startswith("Run2016H") or not(dataset.startswith("Run")):
            probe_photon_fired_trigger_bit = probe_photon_fired_trigger_2016GFH_or_MC 
        
        if year == "2017" or year == "2018":
            probe_photon_fired_trigger_bit = probe_photon_fired_trigger_2017_or_2018

        photons.add_attributes(probe_photon_fired_trigger_bit=probe_photon_fired_trigger_bit)


        ###########################################
        # create (candidate) tag muons
        ###########################################
        tag_muons = JaggedCandidateArray.candidatesfromcounts(
            muons.counts,
            pt=muons.pt.flatten(),
            eta=muons.eta.flatten(),
            phi=muons.phi.flatten(),
            mass=muons.mass.flatten(),
            muCharge=muons['muCharge'].flatten(),
            probe_muon_fired_trigger_bit=muons['probe_muon_fired_trigger_bit'].flatten(),
            tag_muon_fired_trigger_bit=muons['tag_muon_fired_trigger_bit'].flatten(),
            muIDbit=muons['muIDbit'].flatten(),
            muon_evt_weight=np.repeat(event_info['total_weight'].flatten(), muons.counts),
            )

        _tag_muon_good_pt = tag_muons.pt > 26.0 # pT > 26 GeV
        if year == "2017":
            _tag_muon_good_pt = tag_muons.pt > 29.0 # pT > 29 GeV
        _tag_muon_good_abs_eta = np.absolute(tag_muons.eta) < 2.4 # |eta| < 2.4
        _tag_muon_good_fired_trigger = tag_muons["tag_muon_fired_trigger_bit"] # fired the HLT_IsoMu27_v* 
        _tag_muon_good_id = np.bitwise_and(np.right_shift(tag_muons["muIDbit"], 3), 1).astype(bool) # CutBasedIdTight
        _tag_muon_good_iso = np.bitwise_and(np.right_shift(tag_muons["muIDbit"], 9), 1).astype(bool) # PFIsoTight
        _is_good_tag_muon = (_tag_muon_good_pt & _tag_muon_good_abs_eta & _tag_muon_good_fired_trigger & _tag_muon_good_id & _tag_muon_good_iso) 

        tag_muons = tag_muons[_is_good_tag_muon]

        # at least one good tag muon
        _at_least_one_tag = (tag_muons.counts >= 1)
        output_buff['cutflow']['at_least_one_tag'] += (_at_least_one_tag*event_info['total_weight']).sum()
        tag_muons = tag_muons[_at_least_one_tag]
        muons = muons[_at_least_one_tag]
        photons = photons[_at_least_one_tag]
        event_info = event_info[_at_least_one_tag]



        ####################################
        # probe muon selection
        ####################################      
        # # muon reco
        # # REF: https://twiki.cern.ch/twiki/bin/viewauth/CMS/HiggsZZ4lRunIILegacy#Muons
        
        # # muon types
        # # REF: https://github.com/cms-sw/cmssw/blob/CMSSW_9_4_X/DataFormats/MuonReco/interface/Muon.h#L284-L291
        # # GlobalMuon --> 1;
        # # TrackerMuon --> 2;
        # # StandAloneMuon --> 3;
        # # CaloMuon --> 4;
        # # PFMuon --> 5;
        # # RPCMuon --> 6; 
        # # GEMMuon --> 7;
        # # ME0Muon --> 8;
        # # print (muons.pt.flatten().shape if ((dataset == "ZToUpsilon1SGamma" and year == "2016") or (dataset == "Run2016B-17Jul2018_ver")) else "") 
        # # Loose Muons
        # _loose_muon_good_min_pt = (muons.pt > 5.0)
        # _loose_muon_good_eta = (np.absolute(muons.eta) < 2.4)
        # _loose_muon_good_dxy = (np.absolute(muons['muD0']) < 0.5)
        # _loose_muon_good_dz = (np.absolute(muons['muDz']) < 1.0)
        # _loose_muon_good_global_muon = (np.bitwise_and(np.right_shift(muons["muType"].astype(int), 1), 1).astype(bool))
        # _loose_muon_good_tracker_muon = (np.bitwise_and(np.right_shift(muons["muType"].astype(int), 2), 1).astype(bool))
        # _loose_muon_good_mu_matches = (muons['muMatches'] > 0)
        # _loose_muon_good_best_track_type = (muons['muBestTrkType'] != 2)

        # _muon_good_loose_muons = (_loose_muon_good_min_pt & _loose_muon_good_min_pt & _loose_muon_good_eta & _loose_muon_good_dxy & _loose_muon_good_dz & ((_loose_muon_good_global_muon | (_loose_muon_good_tracker_muon & _loose_muon_good_mu_matches)) & _loose_muon_good_best_track_type))

        # # Tracker High Pt
        # _tracker_high_pt_muon_good_mu_matches = (muons['muMatches'] > 1)
        # _tracker_high_pt_muon_good_rel_pt_error = (muons['muBestTrkPtError']/muons['muBestTrkPt'] < 0.3)
        # _tracker_high_pt_muon_good_dxy = (np.absolute(muons['muD0']) < 0.2)
        # _tracker_high_pt_muon_good_dz = (np.absolute(muons['muDz']) < 0.5)
        # _tracker_high_pt_muon_good_pixel_hits = (muons['muPixelHits'] > 0)
        # _tracker_high_pt_muon_good_trk_layers = (muons['muTrkLayers'] > 5)

        # _muon_good_tracker_high_pt = (_tracker_high_pt_muon_good_mu_matches  & _tracker_high_pt_muon_good_rel_pt_error  & _tracker_high_pt_muon_good_dxy & _tracker_high_pt_muon_good_dz & _tracker_high_pt_muon_good_pixel_hits & _tracker_high_pt_muon_good_trk_layers)

        # # Tight Muons
        # _tight_muon_good_high_pt = (muons.pt > 200.0)
        # _tight_muon_good_bdt = (muons.pt < 0.0)
        # if year == "2016":
        #     _tight_muon_good_bdt = (((muons.pt <= 10.0) & (muons['bdt_score'] > 0.8847169876098633)) | ((muons.pt > 10)  & (muons['bdt_score'] > -0.19389629721641488)))
        # if year == "2017":
        #     _tight_muon_good_bdt = (((muons.pt <= 10.0) & (muons['bdt_score'] > 0.883555161952972)) | ((muons.pt > 10)  & (muons['bdt_score'] > -0.3830992293357821)))
        # if year == "2018":
        #     _tight_muon_good_bdt = (((muons.pt <= 10.0) & (muons['bdt_score'] > 0.9506129026412962)) | ((muons.pt > 10)  & (muons['bdt_score'] > -0.3629065185785282)))

        # _is_good_muon_tight = (_muon_good_loose_muons & (_tight_muon_good_bdt | (_muon_good_tracker_high_pt & _tight_muon_good_high_pt)))

        # muons = muons[_is_good_muon_tight]


        # _probe_muon_good_pt = muons.pt > 3 # pT > 5 GeV
        # _probe_muon_good_abs_eta = np.absolute(muons.eta) < 2.4 # |eta| < 2.4
        # # CutBasedIdSoft - 13
        # # CutBasedIdTight - 3
        # # PFIsoTight - 9
        # _probe_muon_good_id = np.bitwise_and(np.right_shift(muons["muIDbit"], 3), 1).astype(bool) 
        # _probe_muon_good_iso = np.bitwise_and(np.right_shift(muons["muIDbit"], 9), 1).astype(bool) # PFIsoTight
        # # _probe_muon_good_dxy = (np.absolute(muons['muD0']) < 0.5)
        # # _probe_muon_good_dz = (np.absolute(muons['muDz']) < 1.0)
        # _is_good_probe_muon = (_probe_muon_good_pt & _probe_muon_good_abs_eta & _probe_muon_good_id & _probe_muon_good_iso) 
        # # _is_good_probe_muon = (_probe_muon_good_pt & _probe_muon_good_abs_eta & _probe_muon_good_id) 

        # muons = muons[_is_good_probe_muon]

        # run muon identifier
        muons = getattr(utils.muon_identifiers, config_doc['muon_id'])(muons, year)

        # at least one good probe muons
        _at_least_two_probe_muons = (muons.counts >= 1)
        output_buff['cutflow']['at_least_two_probe_muons'] += (_at_least_two_probe_muons*event_info['total_weight']).sum()
        muons = muons[_at_least_two_probe_muons]
        tag_muons = tag_muons[_at_least_two_probe_muons]
        photons = photons[_at_least_two_probe_muons]
        event_info = event_info[_at_least_two_probe_muons]

        probe_muons = JaggedCandidateArray.candidatesfromcounts(
            muons.counts,
            pt=muons.pt.flatten(),
            eta=muons.eta.flatten(),
            phi=muons.phi.flatten(),
            mass=muons.mass.flatten(),
            muCharge=muons['muCharge'].flatten(),
            probe_muon_fired_trigger_bit=muons['probe_muon_fired_trigger_bit'].flatten(),
            tag_muon_fired_trigger_bit=muons['tag_muon_fired_trigger_bit'].flatten(),
            muIDbit=muons['muIDbit'].flatten(),
            )

        # pprint(tag_muons.shape)
        # pprint(probe_muons.shape)
        # pprint(photons.shape)

        # # dimuons with oposite muCharge
        # dimuons = muons.distincts()
        # opposite_muCharge_dimuons = (dimuons.i0['muCharge'] * dimuons.i1['muCharge'] == -1)
        # dimuons = dimuons[opposite_muCharge_dimuons]
        # opposite_charge = (dimuons.counts >= 1)
        # output_buff['cutflow']['reco_opposite_charge'] += (opposite_charge*event_info['total_weight']).sum()
        # muons = muons[opposite_charge]
        # photons = photons[opposite_charge]
        # dimuons = dimuons[opposite_charge]
        # event_info = event_info[opposite_charge]

        # # dimuons with lead muon pt > 20.0 GeV (right above the trigger)
        # dimuons_min_lead_pt = ((dimuons.i0.pt > 20.) | (dimuons.i1.pt > 5.))
        # dimuons = dimuons[dimuons_min_lead_pt]
        # min_lead_pt = (dimuons.counts >= 1)
        # output_buff['cutflow']['reco_min_lead_pt'] += (min_lead_pt*event_info['total_weight']).sum()
        # muons = muons[min_lead_pt]
        # photons = photons[min_lead_pt]
        # dimuons = dimuons[min_lead_pt]
        # event_info = event_info[min_lead_pt]

        ####################################
        # photon selection
        ####################################

        # photon RECO
        _photon_good_eta = (np.absolute(photons['phoSCEta']) < 2.5)
        _photon_good_EleVeto = (photons['phoEleVeto'] > 0)
        _photon_good_MVA_id_EB = ((photons['phoIDMVA'] > -0.02) & (np.absolute(photons['phoSCEta']) < 1.4442))
        _photon_good_MVA_id_EE = ((photons['phoIDMVA'] > -0.26) & (np.absolute(photons['phoSCEta']) > 1.566))
        _photon_good_MVA_id_EB_EE = (_photon_good_MVA_id_EB | _photon_good_MVA_id_EE)

        # _photon_good_pt = (photons.pt > 33.0) # HLT_Mu17_Photon30_*_v

        # _is_good_photon = (_photon_good_eta & _photon_good_EleVeto & _photon_good_MVA_id_EB_EE & _photon_good_pt)
        _is_good_photon = (_photon_good_eta & _photon_good_EleVeto & _photon_good_MVA_id_EB_EE )
        # _is_good_photon = (_photon_good_pt)
        photons = photons[_is_good_photon]

        # at least one good photon
        one_photon = (photons.counts >= 1)
        output_buff['cutflow']['one_recophoton'] += (one_photon*event_info['total_weight']).sum()
        # dimuons = dimuons[one_photon]
        # muons = muons[one_photon]
        tag_muons = tag_muons[one_photon]
        probe_muons = probe_muons[one_photon]
        photons = photons[one_photon]
        event_info = event_info[one_photon]

        probe_photons = photons

        # ####################################
        # # dimuon selection
        # ####################################
        # # apply dimuon mass cut
        # muon_mass_cut_dimuons = (dimuons.mass > 8.0) & (dimuons.mass < 11.0)
        # dimuons = dimuons[muon_mass_cut_dimuons]
        # dimuon_mass_cut = (dimuons.counts >= 1)
        # output_buff['cutflow']['dimuon_mass_cut'] += (dimuon_mass_cut*event_info['total_weight']).sum()
        # dimuons = dimuons[dimuon_mass_cut]
        # muons = muons[dimuon_mass_cut]
        # photons = photons[dimuon_mass_cut]
        # event_info = event_info[dimuon_mass_cut]

        ########################################
        # triplets selection
        ########################################
        # build triplets
        triplets = None
        if (probe_photons.counts.size != 0):
            # define muon pairs
            muon_pairs = tag_muons.cross(probe_muons)
            ########################################
            # get muon_pairs components
            # muon_pairs.i0 ==> tag_muons
            # muon_pairs.i1 ==> probe_muons
            ########################################

            # muon pairs with oposite charge
            opposite_charge_muon_pairs = (muon_pairs.i0['muCharge'] * muon_pairs.i1['muCharge'] < 0)
            muon_pairs = muon_pairs[opposite_charge_muon_pairs]

            opposite_charge = (muon_pairs.counts >= 1)

            muon_pairs = muon_pairs[opposite_charge]
            tag_muons = tag_muons[opposite_charge]
            probe_muons = probe_muons[opposite_charge]
            probe_photons = probe_photons[opposite_charge]  

            
            # # good tags
            # good_tags_muon_pairs = (muon_pairs.i0['tag_muon_fired_trigger_bit'] == True)
            # muon_pairs = muon_pairs[good_tags_muon_pairs]

            # good_tags = (muon_pairs.counts >= 1)

            # muon_pairs = muon_pairs[good_tags]
            # tag_muons = tag_muons[good_tags]
            # probe_muons = probe_muons[good_tags]
            # probe_photons = probe_photons[good_tags]  

            # define the triples
            triplets = muon_pairs.cross(probe_photons)
            ########################################
            # get triples components
            # triplets.i0 ==> muon_pairs
            # triplets.i0.i0 ==> tag_muons
            # triplets.i0.i1 ==> probe_muons
            # triplets.i1 ==> probe_photons
            ########################################

            ###################
            # feed accumulator 
            ###################

            # 'triplet_weight': processor.column_accumulator(np.array([])),
            output_buff['triplet_weight'] += processor.column_accumulator((triplets.i0.i0["muon_evt_weight"]).flatten())


            # leading muon
            output_buff['tag_muon_pt'] += processor.column_accumulator((triplets.i0.i0.pt).flatten())
            output_buff['tag_muon_eta'] += processor.column_accumulator((triplets.i0.i0.eta).flatten())
            output_buff['tag_muon_phi'] += processor.column_accumulator((triplets.i0.i0.phi).flatten())

            # trailing muon
            output_buff['probe_muon_pt'] += processor.column_accumulator((triplets.i0.i1.pt).flatten())
            output_buff['probe_muon_eta'] += processor.column_accumulator((triplets.i0.i1.eta).flatten())
            output_buff['probe_muon_phi'] += processor.column_accumulator((triplets.i0.i1.phi).flatten())

            # photon
            output_buff['probe_photon_eta'] += processor.column_accumulator((triplets.i1.eta).flatten())
            output_buff['probe_photon_phi'] += processor.column_accumulator((triplets.i1.phi).flatten())
            output_buff['probe_photon_et'] += processor.column_accumulator((triplets.i1.pt).flatten())


            # passing probe flag
            is_good_probe = ((triplets.i0.i1.probe_muon_fired_trigger_bit == True) & (triplets.i1.probe_photon_fired_trigger_bit == True))

            # pprint((triplets.i0.pt).flatten().shape)
            # pprint((triplets.i0.i1.probe_muon_fired_trigger_bit).flatten().shape)
            # pprint((triplets.i1.probe_photon_fired_trigger_bit).flatten().shape)
            # pprint(is_good_probe.flatten().shape)

            # is_good_probe =  (triplets.i1.probe_photon_fired_trigger_bit == True)
            # output_buff['good_probe'] += processor.column_accumulator(is_good_probe.flatten())

            # pprint("\nis_good_probe:")
            # pprint(is_good_probe.flatten().sum())
            # pprint(is_good_probe.shape)
            output_buff['good_probe'] += processor.column_accumulator(is_good_probe.flatten().astype(int))
            output_buff['good_probe_muon'] += processor.column_accumulator(triplets.i0.i1.probe_muon_fired_trigger_bit.flatten().astype(int))
            output_buff['good_probe_photon'] += processor.column_accumulator(triplets.i1.probe_photon_fired_trigger_bit.flatten().astype(int))
        else:
            output_buff['cutflow']['triplets_count'] += 0
    else:
        print("---> This chunck has no pre-selected events!")

    ########################################
    # done
    ########################################
    # save event info
    evt_info_acc = processor.dict_accumulator({})
    if df['nMu'].value.size > 0: # process only datasets with at least one event selected
        for var in event_info.columns:
            evt_info_acc[var] = processor.column_accumulator(np.array(event_info[var]))
    output_buff += evt_info_acc  

    return output_buff


  def process(self, ds):
    ####################################
    # get accumulator and load data
    ####################################
    output = self.accumulator.identity()
    dataset = ds['dataset_name']
    year = ds['year']
    data_or_mc = ds['data_or_mc'] 

    print("\n\n--> Starting processing of: "+ ds["dataset_name"] + " - "+ ds["year"])    

    # get all files that follow the pattern
    files_from_buffer_zone = glob.glob("outputs/buffer/trigger_ana_" + dataset + "_" + year + "*.coffea")

    # get total total_sign_weight_events
    total_sign_weight_events = 0.0
    for f in files_from_buffer_zone:
        total_sign_weight_events += load(f)['cutflow']['all_events']

    # loop over files
    with ProcessPoolExecutor(max_workers = config_doc['n_processors'] ) as executor:
        results = list(tqdm(executor.map(self.looper, [(dataset, year, data_or_mc, f, total_sign_weight_events) for f in files_from_buffer_zone]), unit=" files", total=len(files_from_buffer_zone)))

    # merge outputs
    print("\n\n--> Merging accumulators for: "+ ds["dataset_name"] + " - "+ ds["year"])
    output = utils.accumulator_merger(results)
    # for res in tqdm(results, unit=" files"):
    #     output = output + res
    # pprint(output)

    print("--> Saving output for: "+ ds["dataset_name"] + " - "+ ds["year"])
    with uproot.recreate("outputs/trigger_sel_" + ds["dataset_name"] + "_"+ ds["year"] + ".root") as f:
        f["triplets"] = uproot.newtree({
                                        "triplet_weight": "float",
                                        "tag_muon_pt": "float",
                                        "tag_muon_eta": "float",
                                        "tag_muon_phi": "float",
                                        "probe_muon_pt": "float",
                                        "probe_muon_eta": "float",
                                        "probe_muon_phi": "float",
                                        "probe_photon_pt": "float",
                                        "probe_photon_eta": "float",
                                        "probe_photon_phi": "float",
                                        "good_probe": "int",
                                        "good_probe_muon": "int",
                                        "good_probe_photon": "int",
                                        })


        f["triplets"].extend({
                            "triplet_weight": np.array(output["triplet_weight"].value),
                            "tag_muon_pt": np.array(output["tag_muon_pt"].value),
                            "tag_muon_eta": np.array(output["tag_muon_eta"].value),
                            "tag_muon_phi": np.array(output["tag_muon_phi"].value),
                            "probe_muon_pt": np.array(output["probe_muon_pt"].value),
                            "probe_muon_eta": np.array(output["probe_muon_eta"].value),
                            "probe_muon_phi": np.array(output["probe_muon_phi"].value),
                            "probe_photon_pt": np.array(output["probe_photon_et"].value),
                            "probe_photon_eta": np.array(output["probe_photon_eta"].value),
                            "probe_photon_phi": np.array(output["probe_photon_phi"].value),
                            "good_probe": np.array(output["good_probe"].value),
                            "good_probe_muon": np.array(output["good_probe_muon"].value),
                            "good_probe_photon": np.array(output["good_probe_photon"].value),
                            })


    # print triplets
    if(False):
        print("--> Triplets for: "+ ds["dataset_name"] + " - "+ ds["year"])
        print(tabulate({"triplet_weight": np.array(output["triplet_weight"].value),
                            "tag_muon_pt": np.array(output["tag_muon_pt"].value),
                            "tag_muon_eta": np.array(output["tag_muon_eta"].value),
                            "tag_muon_phi": np.array(output["tag_muon_phi"].value),
                            "probe_muon_pt": np.array(output["probe_muon_pt"].value),
                            "probe_muon_eta": np.array(output["probe_muon_eta"].value),
                            "probe_muon_phi": np.array(output["probe_muon_phi"].value),
                            "probe_photon_pt": np.array(output["probe_photon_et"].value),
                            "probe_photon_eta": np.array(output["probe_photon_eta"].value),
                            "probe_photon_phi": np.array(output["probe_photon_phi"].value),
                            "good_probe": np.array(output["good_probe"].value),
                            "good_probe_muon": np.array(output["good_probe_muon"].value),
                            "good_probe_photon": np.array(output["good_probe_photon"].value),
                        }, headers="keys"))

    # return output - dummy
    return processor.dict_accumulator({
        'foo': processor.defaultdict_accumulator(float),
    })

  def postprocess(self, accumulator):
    # return accumulator - dummy
    return processor.dict_accumulator({
        'foo': processor.defaultdict_accumulator(float),
    })



