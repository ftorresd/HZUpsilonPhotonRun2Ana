from coffea import hist
from coffea.analysis_objects import JaggedCandidateArray
import coffea.processor as processor
from coffea import hist
import numpy as np
import uproot
from awkward import JaggedArray
from pprint import pprint as pprint
from coffea.util import load, save
import awkward
import config
import cppyy
import utils
import glob
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

import yaml
config_doc = yaml.load(open("config/config.yaml", "r"), Loader=yaml.FullLoader)

import utils.accumulator_merger
import utils.muon_identifiers

# Look at ProcessorABC to see the expected methods and what they are supposed to do
class SelectorProcessor(processor.ProcessorABC):
  def __init__(self,):
    ####################################
    # accumulator
    ####################################

    # axis
    dataset_axis = hist.Cat("dataset", "")
    mass_axis = hist.Bin("mass", r"$m_{\mu\mu}$ [GeV]", 70, 8.0, 11.0)
    # pt_axis = hist.Bin("pt", r"$p_{T,\mu}$ [GeV]", 3000, 0.25, 300)
    self._accumulator = processor.dict_accumulator({
        # dimuon mass array for Upsilon mass fit
        'dimuon_mass_for_upsilon_fit': processor.column_accumulator(np.array([])),
        'dimuon_mass_for_upsilon_fit_weights_sign': processor.column_accumulator(np.array([])),
        
        # leading muon
        'leadmu_pt': processor.column_accumulator(np.array([])),
        'leadmu_eta': processor.column_accumulator(np.array([])),
        'leadmu_phi': processor.column_accumulator(np.array([])),

        # trailing muon
        'trailmu_pt': processor.column_accumulator(np.array([])),
        'trailmu_eta': processor.column_accumulator(np.array([])),
        'trailmu_phi': processor.column_accumulator(np.array([])),        

        # photon
        'photon_pt': processor.column_accumulator(np.array([])),
        'photon_eta': processor.column_accumulator(np.array([])),
        'photon_phi': processor.column_accumulator(np.array([])),    
        # 'photon_energy': processor.column_accumulator(np.array([])),    
        'photon_calibEnergy': processor.column_accumulator(np.array([])),    
        'photon_calibEt': processor.column_accumulator(np.array([])),    
        'photon_etasc': processor.column_accumulator(np.array([])),    

        # upsilon
        'upsilon_pt': processor.column_accumulator(np.array([])),
        'upsilon_eta': processor.column_accumulator(np.array([])),
        'upsilon_phi': processor.column_accumulator(np.array([])),
        'upsilon_mass': processor.column_accumulator(np.array([])),

        # boson
        'boson_pt': processor.column_accumulator(np.array([])),
        'boson_eta': processor.column_accumulator(np.array([])),
        'boson_phi': processor.column_accumulator(np.array([])),
        'boson_mass': processor.column_accumulator(np.array([])),


        'delta_R_leading_photon': processor.column_accumulator(np.array([])),
        'delta_R_leading_photon_mask': processor.column_accumulator(np.array([])),
        
        'delta_R_trailing_photon': processor.column_accumulator(np.array([])),
        'delta_R_trailing_photon_mask': processor.column_accumulator(np.array([])),
        
        'delta_R_upsilon_photon': processor.column_accumulator(np.array([])),
        'delta_R_upsilon_photon_mask': processor.column_accumulator(np.array([])),
        
        'delta_Phi_leading_photon': processor.column_accumulator(np.array([])),
        'delta_Phi_leading_photon_mask': processor.column_accumulator(np.array([])),
        
        'et_photon_over_M_boson': processor.column_accumulator(np.array([])),
        'et_photon_over_M_boson_mask': processor.column_accumulator(np.array([])),

        'pt_upsilon_over_M_boson': processor.column_accumulator(np.array([])),
        'pt_upsilon_over_M_boson_mask': processor.column_accumulator(np.array([])),

        # histograms
        # 'muon_leading_pt': hist.Hist("Counts", dataset_axis, mass_axis),
        # 'muon_leading_eta': hist.Hist("Counts", dataset_axis, mass_axis),
        # 'muon_leading_phi': hist.Hist("Counts", dataset_axis, mass_axis),

        # 'muon_trailing_pt': hist.Hist("Counts", dataset_axis, mass_axis),
        # 'muon_trailing_eta': hist.Hist("Counts", dataset_axis, mass_axis),
        # 'muon_trailing_phi': hist.Hist("Counts", dataset_axis, mass_axis),

        # 'photon_pt': hist.Hist("Counts", dataset_axis, mass_axis),
        # 'photon_eta': hist.Hist("Counts", dataset_axis, mass_axis),
        # 'photon_phi': hist.Hist("Counts", dataset_axis, mass_axis),

        # 'upsilon_pt': hist.Hist("Counts", dataset_axis, mass_axis),
        # 'upsilon_eta': hist.Hist("Counts", dataset_axis, mass_axis),
        # 'upsilon_phi': hist.Hist("Counts", dataset_axis, mass_axis),
        # 'upsilon_mass': hist.Hist("Counts", dataset_axis, mass_axis),

        # 'boson_pt': hist.Hist("Counts", dataset_axis, mass_axis),
        # 'boson_eta': hist.Hist("Counts", dataset_axis, mass_axis),
        # 'boson_phi': hist.Hist("Counts", dataset_axis, mass_axis),
        # 'boson_mass': hist.Hist("Counts", dataset_axis, mass_axis),

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
            weight_pu_nominal, weight_pu_up, weight_pu_down = utils.get_pu_weight(event_info['puTrue'], dataset, year)
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
        muon_corrected_pt = df['muPt'].value*muon_scale_factor

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
        # muon selection
        ####################################
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
        muons.add_attributes(bdt_score=muons_bdt_score)

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

        # run muon identifier
        muons = getattr(utils.muon_identifiers, config_doc['muon_id'])(muons, year)

        # at least two good muons
        two_recomuons = (muons.counts >= 2)
        output_buff['cutflow']['two_recomuons'] += (two_recomuons*event_info['total_weight']).sum()
        muons = muons[two_recomuons]
        photons = photons[two_recomuons]
        event_info = event_info[two_recomuons]

        # dimuons with oposite muCharge
        dimuons = muons.distincts()
        opposite_muCharge_dimuons = (dimuons.i0['muCharge'] * dimuons.i1['muCharge'] == -1)
        dimuons = dimuons[opposite_muCharge_dimuons]
        opposite_charge = (dimuons.counts >= 1)
        output_buff['cutflow']['reco_opposite_charge'] += (opposite_charge*event_info['total_weight']).sum()
        muons = muons[opposite_charge]
        photons = photons[opposite_charge]
        dimuons = dimuons[opposite_charge]
        event_info = event_info[opposite_charge]

        # dimuons with lead muon pt > 20.0 GeV (right above the trigger)
        dimuons_min_lead_pt = ((dimuons.i0.pt > 20.) | (dimuons.i1.pt > 5.))
        dimuons = dimuons[dimuons_min_lead_pt]
        min_lead_pt = (dimuons.counts >= 1)
        output_buff['cutflow']['reco_min_lead_pt'] += (min_lead_pt*event_info['total_weight']).sum()
        muons = muons[min_lead_pt]
        photons = photons[min_lead_pt]
        dimuons = dimuons[min_lead_pt]
        event_info = event_info[min_lead_pt]

        # save dimuon mass ntuples for the upsilon fit
        output_buff['dimuon_mass_for_upsilon_fit'] += processor.column_accumulator((dimuons.mass).flatten())
        output_buff['dimuon_mass_for_upsilon_fit_weights_sign'] += processor.column_accumulator(np.repeat(event_info['total_weight'].flatten(), muons.counts))

        ####################################
        # photon selection
        ####################################

        # photon RECO
        _photon_good_eta = (np.absolute(photons['phoSCEta']) < 2.5)
        _photon_good_EleVeto = (photons['phoEleVeto'] > 0)
        _photon_good_MVA_id_EB = ((photons['phoIDMVA'] > -0.02) & (np.absolute(photons['phoSCEta']) < 1.4442))
        _photon_good_MVA_id_EE = ((photons['phoIDMVA'] > -0.26) & (np.absolute(photons['phoSCEta']) > 1.566))
        _photon_good_MVA_id_EB_EE = (_photon_good_MVA_id_EB | _photon_good_MVA_id_EE)

        _photon_good_pt = (photons.pt > 33.0) # HLT_Mu17_Photon30_*_v

        _is_good_photon = (_photon_good_eta & _photon_good_EleVeto & _photon_good_MVA_id_EB_EE & _photon_good_pt)
        # _is_good_photon = (_photon_good_pt)
        photons = photons[_is_good_photon]

        # at least one good photon
        one_photon = (photons.counts >= 1)
        output_buff['cutflow']['one_recophoton'] += (one_photon*event_info['total_weight']).sum()
        dimuons = dimuons[one_photon]
        muons = muons[one_photon]
        photons = photons[one_photon]
        event_info = event_info[one_photon]

        ####################################
        # dimuon selection
        ####################################
        # apply dimuon mass cut
        muon_mass_cut_dimuons = (dimuons.mass > 8.0) & (dimuons.mass < 11.0)
        dimuons = dimuons[muon_mass_cut_dimuons]
        dimuon_mass_cut = (dimuons.counts >= 1)
        output_buff['cutflow']['dimuon_mass_cut'] += (dimuon_mass_cut*event_info['total_weight']).sum()
        dimuons = dimuons[dimuon_mass_cut]
        muons = muons[dimuon_mass_cut]
        photons = photons[dimuon_mass_cut]
        event_info = event_info[dimuon_mass_cut]

        ########################################
        # boson (Z or Higgs) reconstruction
        ########################################

        # build best boson
        bosons = None
        if (photons.counts.size != 0):
            bosons = dimuons.cross(photons)
            # print(bosons.counts.size)
            # print(bosons.mass)

            # boson_pdg_mass = 125.0
            # if dataset.startswith("Z")
            #     boson_pdg_mass = 91.1876
            near_bosons = np.abs(bosons.mass - 91.1876).argmin() # z_mass = 91.1876
            # near_bosons = np.abs(bosons.mass - 125.0).argmin() # higgs_mass = 125.0
            bosons = bosons[near_bosons]

            # apply boson mass cut
            mass_cut_boson = (bosons.mass > 70) & (bosons.mass < 120) # z_mass cuts
            # mass_cut_boson = (bosons.mass > 100) & (bosons.mass < 150) # higgs_mass cuts
            bosons = bosons[mass_cut_boson]
            good_bosons_count = (bosons.counts >= 1)
            output_buff['cutflow']['good_bosons_count'] += (good_bosons_count*event_info['total_weight']).sum()
            dimuons = dimuons[good_bosons_count]
            muons = muons[good_bosons_count]
            photons = photons[good_bosons_count]
            bosons = bosons[good_bosons_count]
            event_info = event_info[good_bosons_count]

            ########################################
            # get boson components
            # bosons.i0 ==> dimuon
            # bosons.i1 ==> photon
            ########################################

            is_leading_muon = (bosons.i0.i0.pt.content > bosons.i0.i1.pt.content)

            muon_leading_pt = np.where(is_leading_muon, bosons.i0.i0.pt.content, bosons.i0.i1.pt.content)
            muon_leading_eta = np.where(is_leading_muon, bosons.i0.i0.eta.content, bosons.i0.i1.eta.content)
            muon_leading_phi = np.where(is_leading_muon, bosons.i0.i0.phi.content, bosons.i0.i1.phi.content)
            muon_leading = JaggedCandidateArray.candidatesfromcounts(
                np.ones(muon_leading_pt.size),
                pt=muon_leading_pt,
                eta=muon_leading_eta,
                phi=muon_leading_phi,
                mass=105.6583745/1000.,
                )

            muon_trailing_pt = np.where(~is_leading_muon, bosons.i0.i0.pt.content, bosons.i0.i1.pt.content)
            muon_trailing_eta = np.where(~is_leading_muon, bosons.i0.i0.eta.content, bosons.i0.i1.eta.content)
            muon_trailing_phi = np.where(~is_leading_muon, bosons.i0.i0.phi.content, bosons.i0.i1.phi.content)
            muon_trailing = JaggedCandidateArray.candidatesfromcounts(
                np.ones(muon_leading_pt.size),
                pt=muon_leading_pt,
                eta=muon_leading_eta,
                phi=muon_leading_phi,
                mass=105.6583745/1000.,
                )

            upsilon = bosons.i0
            photon = bosons.i1

            ########################################
            # feed accumulator - no kin cuts
            ########################################

            # leading muon
            output_buff['leadmu_pt'] += processor.column_accumulator((muon_leading.pt).flatten())
            output_buff['leadmu_eta'] += processor.column_accumulator((muon_leading.eta).flatten())
            output_buff['leadmu_phi'] += processor.column_accumulator((muon_leading.phi).flatten())

            # trailing muon
            output_buff['trailmu_pt'] += processor.column_accumulator((muon_trailing.pt).flatten())
            output_buff['trailmu_eta'] += processor.column_accumulator((muon_trailing.eta).flatten())
            output_buff['trailmu_phi'] += processor.column_accumulator((muon_trailing.phi).flatten())

            # photon
            output_buff['photon_pt'] += processor.column_accumulator((photon.pt).flatten())
            output_buff['photon_eta'] += processor.column_accumulator((photon.eta).flatten())
            output_buff['photon_phi'] += processor.column_accumulator((photon.phi).flatten())
            # output_buff['photon_energy'] += processor.column_accumulator((photon['energy']).flatten())
            output_buff['photon_calibEnergy'] += processor.column_accumulator((photon['phoCalibE']).flatten())
            output_buff['photon_calibEt'] += processor.column_accumulator((photon['phoCalibEt']).flatten())
            output_buff['photon_etasc'] += processor.column_accumulator((photon['phoSCEta']).flatten())

            # upsilon
            output_buff['upsilon_pt'] += processor.column_accumulator((upsilon.pt).flatten())
            output_buff['upsilon_eta'] += processor.column_accumulator((upsilon.eta).flatten())
            output_buff['upsilon_phi'] += processor.column_accumulator((upsilon.phi).flatten())
            output_buff['upsilon_mass'] += processor.column_accumulator((upsilon.mass).flatten())

            # boson
            output_buff['boson_pt'] += processor.column_accumulator((bosons.pt).flatten())
            output_buff['boson_eta'] += processor.column_accumulator((bosons.eta).flatten())
            output_buff['boson_phi'] += processor.column_accumulator((bosons.phi).flatten())
            output_buff['boson_mass'] += processor.column_accumulator((bosons.mass).flatten())

            ########################################
            # kinematical vars
            ########################################
            delta_R_leading_photon = muon_leading.p4.delta_r(photon.p4)
            delta_R_leading_photon_mask = delta_R_leading_photon > 1.0
            
            delta_R_trailing_photon = muon_trailing.p4.delta_r(photon.p4)
            delta_R_trailing_photon_mask = delta_R_trailing_photon > 1.0
            
            delta_R_upsilon_photon = upsilon.p4.delta_r(photon.p4)
            delta_R_upsilon_photon_mask = delta_R_upsilon_photon > 2.0
            
            delta_Phi_leading_photon = np.absolute(muon_leading.p4.delta_phi(photon.p4))
            delta_Phi_leading_photon_mask = delta_Phi_leading_photon > 1.5
            
            et_photon_over_M_boson = photon.pt/bosons.mass
            et_photon_over_M_boson_mask = et_photon_over_M_boson > 35.0/91.1876
            
            pt_upsilon_over_M_boson = upsilon.pt/bosons.mass
            pt_upsilon_over_M_boson_mask = pt_upsilon_over_M_boson > 35.0/91.1876

            output_buff['delta_R_leading_photon'] += processor.column_accumulator((delta_R_leading_photon).flatten())
            output_buff['delta_R_leading_photon_mask'] += processor.column_accumulator((delta_R_leading_photon_mask).flatten())

            output_buff['delta_R_trailing_photon'] += processor.column_accumulator((delta_R_trailing_photon).flatten())
            output_buff['delta_R_trailing_photon_mask'] += processor.column_accumulator((delta_R_trailing_photon_mask).flatten())

            output_buff['delta_R_upsilon_photon'] += processor.column_accumulator((delta_R_upsilon_photon).flatten())
            output_buff['delta_R_upsilon_photon_mask'] += processor.column_accumulator((delta_R_upsilon_photon_mask).flatten())

            output_buff['delta_Phi_leading_photon'] += processor.column_accumulator((delta_Phi_leading_photon).flatten())
            output_buff['delta_Phi_leading_photon_mask'] += processor.column_accumulator((delta_Phi_leading_photon_mask).flatten())

            output_buff['et_photon_over_M_boson'] += processor.column_accumulator((et_photon_over_M_boson).flatten())
            output_buff['et_photon_over_M_boson_mask'] += processor.column_accumulator((et_photon_over_M_boson_mask).flatten())

            output_buff['pt_upsilon_over_M_boson'] += processor.column_accumulator((pt_upsilon_over_M_boson).flatten())
            output_buff['pt_upsilon_over_M_boson_mask'] += processor.column_accumulator((pt_upsilon_over_M_boson_mask).flatten())
        else:
            output_buff['cutflow']['good_bosons_count'] += 0
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
    files_from_buffer_zone = glob.glob("outputs/buffer/ana_" + dataset + "_" + year + "*.coffea")

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


    print("--> Saving output for: "+ ds["dataset_name"] + " - "+ ds["year"])
    save(output, "outputs/sel_" + ds["dataset_name"] + "_"+ ds["year"] + ".coffea")

    ####################################
    # save invariant masses for 2D fit
    ####################################
    print("--> Saving invariant masses for: "+ ds["dataset_name"] + " - "+ ds["year"])
    with uproot.recreate("outputs/mass_" + ds["dataset_name"] + "_"+ ds["year"] + ".root") as f:
        f["invariant_masses"] = uproot.newtree({
                                        "evt_weight": "float",
                                        "boson_mass": "float",
                                        "upsilon_mass": "float",
                                        })

        f["dimuon_mass"] = uproot.newtree({
                                        "dimuon_mass_for_upsilon_fit": "float",
                                        })

        try:
            full_selection_mask = output['delta_R_leading_photon_mask'].value.astype(bool) & output['delta_R_trailing_photon_mask'].value.astype(bool) & output['delta_R_upsilon_photon_mask'].value.astype(bool) & output['delta_Phi_leading_photon_mask'].value.astype(bool) & output['et_photon_over_M_boson_mask'].value.astype(bool) & output['pt_upsilon_over_M_boson_mask'].value.astype(bool)
            f["invariant_masses"].extend({
                        "evt_weight": np.array(output['total_weight'].value[full_selection_mask]),
                        "boson_mass": np.array(output["boson_mass"].value[full_selection_mask]),
                        "upsilon_mass": np.array(output["upsilon_mass"].value[full_selection_mask]),
                        })

            f["dimuon_mass"].extend({
                                "dimuon_mass_for_upsilon_fit": np.array(output["dimuon_mass_for_upsilon_fit"].value),
                                })
        except KeyError:
            pass

        


    # return output - dummy
    return processor.dict_accumulator({
        'foo': processor.defaultdict_accumulator(float),
    })

  def postprocess(self, accumulator):
    # return accumulator - dummy
    return processor.dict_accumulator({
        'foo': processor.defaultdict_accumulator(float),
    })



