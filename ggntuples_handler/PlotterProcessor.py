from coffea import hist
from coffea.analysis_objects import JaggedCandidateArray
import coffea.processor as processor
import numpy as np
from awkward import JaggedArray
from pprint import pprint as pprint
from coffea.util import load, save
import awkward
import config
import uproot
from pprint import pprint as pprint
# import sys



# Look at ProcessorABC to see the expected methods and what they are supposed to do
class PlotterProcessor(processor.ProcessorABC):
  def __init__(self,):
    # # leading muon
    # leadmu_pt
    # leadmu_eta
    # leadmu_phi

    ### trailing muon
    # trailmu_pt
    # trailmu_eta
    # trailmu_phi        

    ### photon
    # photon_pt
    # photon_eta
    # photon_phi    
    # photon_calibEnergy    
    # photon_calibEt    
    # photon_etasc    

    ### upsilon
    # upsilon_pt
    # upsilon_eta
    # upsilon_phi
    # upsilon_mass

    ### boson
    # boson_pt
    # boson_eta
    # boson_phi
    # boson_mass


    ### kinematical cuts
    # delta_R_leading_photon
    # delta_R_trailing_photon
    # delta_R_upsilon_photon
    # delta_Phi_leading_photon
    # et_photon_over_M_boson
    # pt_upsilon_over_M_boson

    #dimuon_mass_for_upsilon_fit
    #dimuon_mass_for_upsilon_fit_weights_sign

    ####################################
    # accumulator
    ####################################

    # cats and axis
    dataset_axis = hist.Cat("dataset", "")

    leadmu_pt_axis_noKinCuts = hist.Bin("pt", r"$p_{T}^{\mu}$ (GeV)", 160, 0.0, 160.0)
    leadmu_pt_axis_withKinCuts = hist.Bin("pt", r"$p_{T}^{\mu}$ (GeV)", 160, 0.0, 160.0)
    leadmu_eta_axis = hist.Bin("eta", r"$\eta_{\mu}$", 50, -2.5, 2.5)
    leadmu_phi_axis = hist.Bin("phi", r"$\phi_{\mu}$", 70, -3.2, 3.2)

    trailmu_pt_axis_noKinCuts = hist.Bin("pt", r"$p_{T}^{\mu}$ (GeV)", 160, 0.0, 160.0)
    trailmu_pt_axis_withKinCuts = hist.Bin("pt", r"$p_{T}^{\mu}$ (GeV)", 160, 0.0, 160.0)
    trailmu_eta_axis = hist.Bin("eta", r"$\eta^{\mu}$", 50, -2.5, 2.5)
    trailmu_phi_axis = hist.Bin("phi", r"$\phi^{\mu}$", 70, -3.2, 3.2)

    photon_et_axis_noKinCuts = hist.Bin("pt", r"$E_{T}^{\gamma}$ (GeV)",  100, 0.0, 160.0)
    photon_et_axis_withKinCuts = hist.Bin("pt", r"$E_{T}^{\gamma}$ (GeV)",  100, 0.0, 160.0)
    photon_eta_axis = hist.Bin("eta", r"$\eta^{SC}_{\gamma}$", 50, -2.5, 2.5)
    photon_phi_axis = hist.Bin("phi", r"$\phi_{\gamma}$", 70, -3.2, 3.2)

    upsilon_pt_axis_noKinCuts = hist.Bin("pt", r"$p_{T}^{\mu\mu}$ (GeV)", 160, 0.0, 160.0)
    upsilon_pt_axis_withKinCuts = hist.Bin("pt", r"$p_{T}^{\mu\mu}$ (GeV)", 160, 0.0, 160.0)
    upsilon_eta_axis = hist.Bin("eta", r"$\eta_{\mu\mu}$", 50, -2.5, 2.5)
    upsilon_phi_axis = hist.Bin("phi", r"$\phi_{\mu\mu}$", 70, -3.2, 3.2)
    upsilon_mass_axis = hist.Bin("mass", r"$m_{\mu\mu}$ (GeV)", 50, 8.0, 11.0)

    boson_pt_axis_noKinCuts = hist.Bin("pt", r"$p_{T}^{\mu\mu\gamma}$ (GeV)", 160, 0.0, 160.0)
    boson_pt_axis_withKinCuts = hist.Bin("pt", r"$p_{T}^{\mu\mu\gamma}$ (GeV)", 160, 0.0, 160.0)
    boson_eta_axis = hist.Bin("eta", r"$\eta_{\mu\mu\gamma}$", 50, -2.5, 2.5)
    boson_phi_axis = hist.Bin("phi", r"$\phi_{\mu\mu\gamma}$", 70, -3.2, 3.2)
    boson_mass_axis = hist.Bin("mass", r"$m_{\mu\mu\gamma}$ (GeV)", 60, 70., 120.)

    delta_R_leading_photon_axis = hist.Bin("delta_R", r"$\Delta R(lead \mu, \gamma)$", 100, 0.0, 5.0)
    delta_R_trailing_photon_axis = hist.Bin("delta_R", r"$\Delta R(trail \mu, \gamma)$", 100, 0.0, 5.0)
    delta_R_upsilon_photon_axis = hist.Bin("delta_R", r"$\Delta R(\mu\mu, \gamma)$", 100, 0.0, 5.0)
    delta_Phi_leading_photon_axis = hist.Bin("delta_Phi", r"$|\Delta\phi(\mu\mu, \gamma)|$", 100, 0.0, 4.0)

    et_photon_over_M_boson_axis_noKinCuts = hist.Bin("et_photon_over_M_boson", r"$E_{T}^{\gamma}/m_{\mu\mu\gamma}$", 100, 0.0, 1.0)
    et_photon_over_M_boson_axis_withKinCuts = hist.Bin("et_photon_over_M_boson", r"$E_{T}^{\gamma}/m_{\mu\mu\gamma}$", 100, 0.0, 1.0)
    
    pt_upsilon_over_M_boson_axis_noKinCuts = hist.Bin("pt_upsilon_over_M_boson", r"$p_{T}^{\mu\mu}/m_{\mu\mu\gamma}$", 100, 0.0, 1.0)
    pt_upsilon_over_M_boson_axis_withKinCuts = hist.Bin("pt_upsilon_over_M_boson", r"$p_{T}^{\mu\mu}/m_{\mu\mu\gamma}$", 100, 0.0, 1.0)

    # dimuon_mass_for_upsilon_fit = hist.Bin("dimuon_mass_for_upsilon_fit", r"$m_{\mu\mu}$ (GeV)", 50, 8.0, 11.0)
    # dimuon_mass_for_upsilon_fit_weights_sign = hist.Bin("dimuon_mass_for_upsilon_fit_weights_sign", r"$m_{\mu\mu}$ (GeV)", 50, 8.0, 11.0)


    self._accumulator = processor.dict_accumulator({

        # cutflow
        'cutflow': processor.defaultdict_accumulator(float),

        # histograms - No Kinematical cuts
        'h_leadmu_pt_noKinCuts': hist.Hist("Counts", dataset_axis, leadmu_pt_axis_noKinCuts),
        'h_leadmu_eta_noKinCuts': hist.Hist("Counts", dataset_axis, leadmu_eta_axis),
        'h_leadmu_phi_noKinCuts': hist.Hist("Counts", dataset_axis, leadmu_phi_axis),

        'h_trailmu_pt_noKinCuts': hist.Hist("Counts", dataset_axis, trailmu_pt_axis_noKinCuts),
        'h_trailmu_eta_noKinCuts': hist.Hist("Counts", dataset_axis, trailmu_eta_axis),
        'h_trailmu_phi_noKinCuts': hist.Hist("Counts", dataset_axis, trailmu_phi_axis),

        'h_photon_pt_noKinCuts': hist.Hist("Counts", dataset_axis, photon_et_axis_noKinCuts),
        'h_photon_eta_noKinCuts': hist.Hist("Counts", dataset_axis, photon_eta_axis),
        'h_photon_phi_noKinCuts': hist.Hist("Counts", dataset_axis, photon_phi_axis),

        'h_upsilon_pt_noKinCuts': hist.Hist("Counts", dataset_axis, upsilon_pt_axis_noKinCuts),
        'h_upsilon_eta_noKinCuts': hist.Hist("Counts", dataset_axis, upsilon_eta_axis),
        'h_upsilon_phi_noKinCuts': hist.Hist("Counts", dataset_axis, upsilon_phi_axis),
        'h_upsilon_mass_noKinCuts': hist.Hist("Counts", dataset_axis, upsilon_mass_axis),

        'h_boson_pt_noKinCuts': hist.Hist("Counts", dataset_axis, boson_pt_axis_noKinCuts),
        'h_boson_eta_noKinCuts': hist.Hist("Counts", dataset_axis, boson_eta_axis),
        'h_boson_phi_noKinCuts': hist.Hist("Counts", dataset_axis, boson_phi_axis),
        'h_boson_mass_noKinCuts': hist.Hist("Counts", dataset_axis, boson_mass_axis),

        'h_delta_R_leading_photon_noKinCuts': hist.Hist("Counts", dataset_axis, delta_R_leading_photon_axis),
        'h_delta_R_trailing_photon_noKinCuts': hist.Hist("Counts", dataset_axis, delta_R_trailing_photon_axis),
        'h_delta_R_upsilon_photon_noKinCuts': hist.Hist("Counts", dataset_axis, delta_R_upsilon_photon_axis),
        'h_delta_Phi_leading_photon_noKinCuts': hist.Hist("Counts", dataset_axis, delta_Phi_leading_photon_axis),
        'h_et_photon_over_M_boson_noKinCuts': hist.Hist("Counts", dataset_axis, et_photon_over_M_boson_axis_noKinCuts),
        'h_pt_upsilon_over_M_boson_noKinCuts': hist.Hist("Counts", dataset_axis, pt_upsilon_over_M_boson_axis_noKinCuts),


        # histograms - With Kinematical cuts
        'h_leadmu_pt_withKinCuts': hist.Hist("Counts", dataset_axis, leadmu_pt_axis_withKinCuts),
        'h_leadmu_eta_withKinCuts': hist.Hist("Counts", dataset_axis, leadmu_eta_axis),
        'h_leadmu_phi_withKinCuts': hist.Hist("Counts", dataset_axis, leadmu_phi_axis),

        'h_trailmu_pt_withKinCuts': hist.Hist("Counts", dataset_axis, trailmu_pt_axis_withKinCuts),
        'h_trailmu_eta_withKinCuts': hist.Hist("Counts", dataset_axis, trailmu_eta_axis),
        'h_trailmu_phi_withKinCuts': hist.Hist("Counts", dataset_axis, trailmu_phi_axis),

        'h_photon_pt_withKinCuts': hist.Hist("Counts", dataset_axis, photon_et_axis_withKinCuts),
        'h_photon_eta_withKinCuts': hist.Hist("Counts", dataset_axis, photon_eta_axis),
        'h_photon_phi_withKinCuts': hist.Hist("Counts", dataset_axis, photon_phi_axis),

        'h_upsilon_pt_withKinCuts': hist.Hist("Counts", dataset_axis, upsilon_pt_axis_withKinCuts),
        'h_upsilon_eta_withKinCuts': hist.Hist("Counts", dataset_axis, upsilon_eta_axis),
        'h_upsilon_phi_withKinCuts': hist.Hist("Counts", dataset_axis, upsilon_phi_axis),
        'h_upsilon_mass_withKinCuts': hist.Hist("Counts", dataset_axis, upsilon_mass_axis),

        'h_boson_pt_withKinCuts': hist.Hist("Counts", dataset_axis, boson_pt_axis_withKinCuts),
        'h_boson_eta_withKinCuts': hist.Hist("Counts", dataset_axis, boson_eta_axis),
        'h_boson_phi_withKinCuts': hist.Hist("Counts", dataset_axis, boson_phi_axis),
        'h_boson_mass_withKinCuts': hist.Hist("Counts", dataset_axis, boson_mass_axis),

        'h_delta_R_leading_photon_withKinCuts': hist.Hist("Counts", dataset_axis, delta_R_leading_photon_axis),
        'h_delta_R_trailing_photon_withKinCuts': hist.Hist("Counts", dataset_axis, delta_R_trailing_photon_axis),
        'h_delta_R_upsilon_photon_withKinCuts': hist.Hist("Counts", dataset_axis, delta_R_upsilon_photon_axis),
        'h_delta_Phi_leading_photon_withKinCuts': hist.Hist("Counts", dataset_axis, delta_Phi_leading_photon_axis),
        'h_et_photon_over_M_boson_withKinCuts': hist.Hist("Counts", dataset_axis, et_photon_over_M_boson_axis_withKinCuts),
        'h_pt_upsilon_over_M_boson_withKinCuts': hist.Hist("Counts", dataset_axis, pt_upsilon_over_M_boson_axis_withKinCuts),
        })

  @property
  def accumulator(self):
    return self._accumulator

  def process(self, ds):
    ####################################
    # get accumulator and load data
    ####################################
    output = self.accumulator.identity()
    dataset = ds['dataset_name']
    year = ds['year']


    df = load("outputs/sel_" + dataset + "_" + year + ".coffea")
    output['cutflow'] += df['cutflow']

    if df['leadmu_pt'].value.size > 0: # process only datasets with at least one event selected
        #event info
        event_info = awkward.Table(**config.get_vars_dict(df, config.evt_vars, get_from_acc=True))
        event_info['one_photon'] = df['one_photon'].value
        event_info['evtWeight_sign'] = df['evtWeight_sign'].value
        event_info['total_weight'] = df['total_weight'].value

        ####################################
        # load data
        ####################################
        leadmu_pt = df['leadmu_pt'].value
        leadmu_eta = df['leadmu_eta'].value
        leadmu_phi = df['leadmu_phi'].value

        trailmu_pt = df['trailmu_pt'].value
        trailmu_eta = df['trailmu_eta'].value
        trailmu_phi = df['trailmu_phi'].value

        photon_pt = df['photon_pt'].value
        photon_eta = df['photon_eta'].value
        photon_phi = df['photon_phi'].value    
        photon_calibEnergy = df['photon_calibEnergy'].value    
        photon_calibEt = df['photon_calibEt'].value    
        photon_etasc = df['photon_etasc'].value    

        upsilon_pt = df['upsilon_pt'].value
        upsilon_eta = df['upsilon_eta'].value
        upsilon_phi = df['upsilon_phi'].value
        upsilon_mass = df['upsilon_mass'].value

        boson_pt = df['boson_pt'].value
        boson_eta = df['boson_eta'].value
        boson_phi = df['boson_phi'].value
        boson_mass = df['boson_mass'].value

        delta_R_leading_photon = df['delta_R_leading_photon'].value
        delta_R_trailing_photon = df['delta_R_trailing_photon'].value
        delta_R_upsilon_photon = df['delta_R_upsilon_photon'].value
        delta_Phi_leading_photon = df['delta_Phi_leading_photon'].value
        et_photon_over_M_boson = df['et_photon_over_M_boson'].value
        pt_upsilon_over_M_boson = df['pt_upsilon_over_M_boson'].value   

        delta_R_leading_photon_mask = df['delta_R_leading_photon_mask'].value.astype(bool)
        delta_R_trailing_photon_mask = df['delta_R_trailing_photon_mask'].value.astype(bool)
        delta_R_upsilon_photon_mask = df['delta_R_upsilon_photon_mask'].value.astype(bool)
        delta_Phi_leading_photon_mask = df['delta_Phi_leading_photon_mask'].value.astype(bool)
        et_photon_over_M_boson_mask = df['et_photon_over_M_boson_mask'].value.astype(bool)
        pt_upsilon_over_M_boson_mask = df['pt_upsilon_over_M_boson_mask'].value.astype(bool)

        ###########################################
        # fill histos - without kinematical cuts
        ###########################################

        output['h_leadmu_pt_noKinCuts'].fill(dataset=dataset+"_"+year, pt=leadmu_pt, weight=event_info['total_weight']) 
        output['h_leadmu_eta_noKinCuts'].fill(dataset=dataset+"_"+year, eta=leadmu_eta, weight=event_info['total_weight'])
        output['h_leadmu_phi_noKinCuts'].fill(dataset=dataset+"_"+year, phi=leadmu_phi, weight=event_info['total_weight'])

        output['h_trailmu_pt_noKinCuts'].fill(dataset=dataset+"_"+year, pt=trailmu_pt, weight=event_info['total_weight'])
        output['h_trailmu_eta_noKinCuts'].fill(dataset=dataset+"_"+year, eta=trailmu_eta, weight=event_info['total_weight'])
        output['h_trailmu_phi_noKinCuts'].fill(dataset=dataset+"_"+year, phi=trailmu_phi, weight=event_info['total_weight'])

        output['h_photon_pt_noKinCuts'].fill(dataset=dataset+"_"+year, pt=photon_pt, weight=event_info['total_weight'])
        output['h_photon_eta_noKinCuts'].fill(dataset=dataset+"_"+year, eta=photon_etasc, weight=event_info['total_weight'])
        output['h_photon_phi_noKinCuts'].fill(dataset=dataset+"_"+year, phi=photon_phi, weight=event_info['total_weight'])

        output['h_upsilon_pt_noKinCuts'].fill(dataset=dataset+"_"+year, pt=upsilon_pt, weight=event_info['total_weight'])
        output['h_upsilon_eta_noKinCuts'].fill(dataset=dataset+"_"+year, eta=upsilon_eta, weight=event_info['total_weight'])
        output['h_upsilon_phi_noKinCuts'].fill(dataset=dataset+"_"+year, phi=upsilon_phi, weight=event_info['total_weight'])
        output['h_upsilon_mass_noKinCuts'].fill(dataset=dataset+"_"+year, mass=upsilon_mass, weight=event_info['total_weight'])

        output['h_boson_pt_noKinCuts'].fill(dataset=dataset+"_"+year, pt=boson_pt, weight=event_info['total_weight'])
        output['h_boson_eta_noKinCuts'].fill(dataset=dataset+"_"+year, eta=boson_eta, weight=event_info['total_weight'])
        output['h_boson_phi_noKinCuts'].fill(dataset=dataset+"_"+year, phi=boson_phi, weight=event_info['total_weight'])
        output['h_boson_mass_noKinCuts'].fill(dataset=dataset+"_"+year, mass=boson_mass, weight=event_info['total_weight'])

        output['h_delta_R_leading_photon_noKinCuts'].fill(dataset=dataset+"_"+year, delta_R=delta_R_leading_photon, weight=event_info['total_weight'])
        output['h_delta_R_trailing_photon_noKinCuts'].fill(dataset=dataset+"_"+year, delta_R=delta_R_trailing_photon, weight=event_info['total_weight'])
        output['h_delta_R_upsilon_photon_noKinCuts'].fill(dataset=dataset+"_"+year, delta_R=delta_R_upsilon_photon, weight=event_info['total_weight'])
        output['h_delta_Phi_leading_photon_noKinCuts'].fill(dataset=dataset+"_"+year, delta_Phi=delta_Phi_leading_photon, weight=event_info['total_weight'])
        output['h_et_photon_over_M_boson_noKinCuts'].fill(dataset=dataset+"_"+year, et_photon_over_M_boson=et_photon_over_M_boson, weight=event_info['total_weight'])
        output['h_pt_upsilon_over_M_boson_noKinCuts'].fill(dataset=dataset+"_"+year, pt_upsilon_over_M_boson=pt_upsilon_over_M_boson, weight=event_info['total_weight'])

        ###########################################
        # fill histos - with kinematical cuts
        ###########################################

        kinematical_cuts_mask = (delta_R_leading_photon_mask) & (delta_R_trailing_photon_mask) & (delta_R_upsilon_photon_mask) & (delta_Phi_leading_photon_mask) & (et_photon_over_M_boson_mask) & (pt_upsilon_over_M_boson_mask) 

        leadmu_pt = leadmu_pt[kinematical_cuts_mask]
        leadmu_eta = leadmu_eta[kinematical_cuts_mask]
        leadmu_phi = leadmu_phi[kinematical_cuts_mask]

        trailmu_pt = trailmu_pt[kinematical_cuts_mask]
        trailmu_eta = trailmu_eta[kinematical_cuts_mask]
        trailmu_phi = trailmu_phi[kinematical_cuts_mask]

        photon_pt = photon_pt[kinematical_cuts_mask]
        photon_eta = photon_eta[kinematical_cuts_mask]
        photon_phi = photon_phi[kinematical_cuts_mask]
        photon_calibEnergy = photon_calibEnergy[kinematical_cuts_mask]
        photon_calibEt = photon_calibEt[kinematical_cuts_mask]
        photon_etasc = photon_etasc[kinematical_cuts_mask]

        upsilon_pt = upsilon_pt[kinematical_cuts_mask]
        upsilon_eta = upsilon_eta[kinematical_cuts_mask]
        upsilon_phi = upsilon_phi[kinematical_cuts_mask]
        upsilon_mass = upsilon_mass[kinematical_cuts_mask]

        boson_pt = boson_pt[kinematical_cuts_mask]
        boson_eta = boson_eta[kinematical_cuts_mask]
        boson_phi = boson_phi[kinematical_cuts_mask]
        boson_mass = boson_mass[kinematical_cuts_mask]

        delta_R_leading_photon = delta_R_leading_photon[kinematical_cuts_mask]
        delta_R_trailing_photon = delta_R_trailing_photon[kinematical_cuts_mask]
        delta_R_upsilon_photon = delta_R_upsilon_photon[kinematical_cuts_mask]
        delta_Phi_leading_photon = delta_Phi_leading_photon[kinematical_cuts_mask]
        et_photon_over_M_boson = et_photon_over_M_boson[kinematical_cuts_mask]
        pt_upsilon_over_M_boson = pt_upsilon_over_M_boson[kinematical_cuts_mask]

        # delta_R_leading_photon_mask = delta_R_leading_photon_mask[kinematical_cuts_mask]
        # delta_R_trailing_photon_mask = delta_R_trailing_photon_mask[kinematical_cuts_mask]
        # delta_R_upsilon_photon_mask = delta_R_upsilon_photon_mask[kinematical_cuts_mask]
        # delta_Phi_leading_photon_mask = delta_Phi_leading_photon_mask[kinematical_cuts_mask]
        # et_photon_over_M_boson_mask = et_photon_over_M_boson_mask[kinematical_cuts_mask]
        # pt_upsilon_over_M_boson_mask = pt_upsilon_over_M_boson_mask[kinematical_cuts_mask]

        ########################################
        # cut flow for kinematical cuts
        ########################################
        output['cutflow']['delta_R_leading_photon'] += (kinematical_cuts_mask*event_info['total_weight']).sum()
        output['cutflow']['delta_R_trailing_photon'] += (kinematical_cuts_mask*event_info['total_weight']).sum()
        output['cutflow']['delta_R_upsilon_photon'] += (kinematical_cuts_mask*event_info['total_weight']).sum()
        output['cutflow']['delta_Phi_leading_photon'] += (kinematical_cuts_mask*event_info['total_weight']).sum()
        output['cutflow']['et_photon_over_M_boson'] += (kinematical_cuts_mask*event_info['total_weight']).sum()
        output['cutflow']['pt_upsilon_over_M_boson'] += (kinematical_cuts_mask*event_info['total_weight']).sum()

        event_info = event_info[kinematical_cuts_mask]

        # with kinematical cuts
        output['h_leadmu_pt_withKinCuts'].fill(dataset=dataset+"_"+year, pt=leadmu_pt, weight=event_info['total_weight']) 
        output['h_leadmu_eta_withKinCuts'].fill(dataset=dataset+"_"+year, eta=leadmu_eta, weight=event_info['total_weight'])
        output['h_leadmu_phi_withKinCuts'].fill(dataset=dataset+"_"+year, phi=leadmu_phi, weight=event_info['total_weight'])

        output['h_trailmu_pt_withKinCuts'].fill(dataset=dataset+"_"+year, pt=trailmu_pt, weight=event_info['total_weight'])
        output['h_trailmu_eta_withKinCuts'].fill(dataset=dataset+"_"+year, eta=trailmu_eta, weight=event_info['total_weight'])
        output['h_trailmu_phi_withKinCuts'].fill(dataset=dataset+"_"+year, phi=trailmu_phi, weight=event_info['total_weight'])

        output['h_photon_pt_withKinCuts'].fill(dataset=dataset+"_"+year, pt=photon_pt, weight=event_info['total_weight'])
        output['h_photon_eta_withKinCuts'].fill(dataset=dataset+"_"+year, eta=photon_etasc, weight=event_info['total_weight'])
        output['h_photon_phi_withKinCuts'].fill(dataset=dataset+"_"+year, phi=photon_phi, weight=event_info['total_weight'])

        output['h_upsilon_pt_withKinCuts'].fill(dataset=dataset+"_"+year, pt=upsilon_pt, weight=event_info['total_weight'])
        output['h_upsilon_eta_withKinCuts'].fill(dataset=dataset+"_"+year, eta=upsilon_eta, weight=event_info['total_weight'])
        output['h_upsilon_phi_withKinCuts'].fill(dataset=dataset+"_"+year, phi=upsilon_phi, weight=event_info['total_weight'])
        output['h_upsilon_mass_withKinCuts'].fill(dataset=dataset+"_"+year, mass=upsilon_mass, weight=event_info['total_weight'])

        output['h_boson_pt_withKinCuts'].fill(dataset=dataset+"_"+year, pt=boson_pt, weight=event_info['total_weight'])
        output['h_boson_eta_withKinCuts'].fill(dataset=dataset+"_"+year, eta=boson_eta, weight=event_info['total_weight'])
        output['h_boson_phi_withKinCuts'].fill(dataset=dataset+"_"+year, phi=boson_phi, weight=event_info['total_weight'])
        output['h_boson_mass_withKinCuts'].fill(dataset=dataset+"_"+year, mass=boson_mass, weight=event_info['total_weight'])

        output['h_delta_R_leading_photon_withKinCuts'].fill(dataset=dataset+"_"+year, delta_R=delta_R_leading_photon, weight=event_info['total_weight'])
        output['h_delta_R_trailing_photon_withKinCuts'].fill(dataset=dataset+"_"+year, delta_R=delta_R_trailing_photon, weight=event_info['total_weight'])
        output['h_delta_R_upsilon_photon_withKinCuts'].fill(dataset=dataset+"_"+year, delta_R=delta_R_upsilon_photon, weight=event_info['total_weight'])
        output['h_delta_Phi_leading_photon_withKinCuts'].fill(dataset=dataset+"_"+year, delta_Phi=delta_Phi_leading_photon, weight=event_info['total_weight'])
        output['h_et_photon_over_M_boson_withKinCuts'].fill(dataset=dataset+"_"+year, et_photon_over_M_boson=et_photon_over_M_boson, weight=event_info['total_weight'])
        output['h_pt_upsilon_over_M_boson_withKinCuts'].fill(dataset=dataset+"_"+year, pt_upsilon_over_M_boson=pt_upsilon_over_M_boson, weight=event_info['total_weight'])

    else:
        ########################################
        # cut flow for kinematical cuts in case of no selected events in the source file
        ########################################
        output['cutflow']['delta_R_leading_photon'] += 0
        output['cutflow']['delta_R_trailing_photon'] += 0
        output['cutflow']['delta_R_upsilon_photon'] += 0
        output['cutflow']['delta_Phi_leading_photon'] += 0
        output['cutflow']['et_photon_over_M_boson'] += 0
        output['cutflow']['pt_upsilon_over_M_boson'] += 0    

        ########################################
        # mass array in case of no selected events in the source file
        ########################################
        boson_mass = np.array([])
        upsilon_mass = np.array([])


    ########################################
    # done
    ########################################
    print("--> Saving output for: "+ ds["dataset_name"] + " - "+ ds["year"])
    save(output, "outputs/plot_" + ds["dataset_name"] + "_"+ ds["year"] + ".coffea")

    print("--> Saving mass dataset for: "+ ds["dataset_name"] + " - "+ ds["year"])
    with uproot.recreate("outputs/mass_" + ds["dataset_name"] + "_"+ ds["year"] + ".root") as f:
        f["mass_tree"] = uproot.newtree({"upsilon_mass": "float64", "boson_mass": "float64", })
        f["mass_tree"].extend({"upsilon_mass": upsilon_mass, "boson_mass": boson_mass, })

    # return output - dummy
    return processor.dict_accumulator({
        'foo': processor.defaultdict_accumulator(float),
    })

  def postprocess(self, accumulator):
    # return accumulator - dummy
    return processor.dict_accumulator({
        'foo': processor.defaultdict_accumulator(float),
    })



