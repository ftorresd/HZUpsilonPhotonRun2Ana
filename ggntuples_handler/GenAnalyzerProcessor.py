from coffea import hist
from coffea.analysis_objects import JaggedCandidateArray
import coffea.processor as processor
from coffea import hist
import numpy as np
from awkward import JaggedArray

# Look at ProcessorABC to see the expected methods and what they are supposed to do
class GenAnalyzerProcessor(processor.ProcessorABC):
  def __init__(self, analyzer_name):
    ####################################
    # accumulator
    ####################################
    self._accumulator = processor.dict_accumulator({
        'cutflow': processor.defaultdict_accumulator(float),

        ####################################
        # load objects
        ####################################
        'dimuon_mass_for_upsilon_fit': processor.column_accumulator(np.array([])),

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
    })

  @property
  def accumulator(self):
    return self._accumulator

  def process(self, df):
    ####################################
    # get accumulator and dataframe
    ####################################
    output = self.accumulator.identity()
    dataset = df['dataset']

    ####################################
    # load objects
    ####################################

    # load muons
    muons = JaggedCandidateArray.candidatesfromcounts(
        df['nMu'],
        pt=df['muPt'],
        eta=df['muEta'],
        phi=df['muPhi'],
        mass=105.6583745/1000.,
        charge=df['muCharge'],
        muType=df['muType'],
        muIDbit=df['muIDbit'],
        muD0=df['muD0'],
        muDz=df['muDz'],
        muSIP=df['muSIP'],
        muChi2NDF=df['muChi2NDF'],
        muInnerD0=df['muInnerD0'],
        muInnerDz=df['muInnerDz'],
        muTrkLayers=df['muTrkLayers'],
        muPixelLayers=df['muPixelLayers'],
        muPixelHits=df['muPixelHits'],
        muMuonHits=df['muMuonHits'],
        muStations=df['muStations'],
        muMatches=df['muMatches'],
        muTrkQuality=df['muTrkQuality'],
        muIsoTrk=df['muIsoTrk'],
        muPFChIso=df['muPFChIso'],
        muPFPhoIso=df['muPFPhoIso'],
        muPFNeuIso=df['muPFNeuIso'],
        muPFPUIso=df['muPFPUIso'],
        muPFChIso03=df['muPFChIso03'],
        muPFPhoIso03=df['muPFPhoIso03'],
        muPFNeuIso03=df['muPFNeuIso03'],
        muPFPUIso03=df['muPFPUIso03'],
        muFiredTrgs=df['muFiredTrgs'],
        muFiredL1Trgs=df['muFiredL1Trgs'],
        muInnervalidFraction=df['muInnervalidFraction'],
        musegmentCompatibility=df['musegmentCompatibility'],
        muchi2LocalPosition=df['muchi2LocalPosition'],
        mutrkKink=df['mutrkKink'],
        muBestTrkPtError=df['muBestTrkPtError'],
        muBestTrkPt=df['muBestTrkPt'],
        muBestTrkType=df['muBestTrkType'],
        )

    # load photons
    photons = JaggedCandidateArray.candidatesfromcounts(
        df['nPho'],
        pt=df['phoCalibEt'],
        eta=df['phoEta'],
        phi=df['phoPhi'],
        energy=df['phoCalibE'],
        phoE=df['phoE'],
        phoSigmaE=df['phoSigmaE'],
        phoEt=df['phoEt'],
        phoEta=df['phoEta'],
        phoPhi=df['phoPhi'],
        phoCalibE=df['phoCalibE'],
        phoCalibEt=df['phoCalibEt'],
        phoSCE=df['phoSCE'],
        phoSCRawE=df['phoSCRawE'],
        phoESEnP1=df['phoESEnP1'],
        phoESEnP2=df['phoESEnP2'],
        phoSCEta=df['phoSCEta'],
        phoSCPhi=df['phoSCPhi'],
        phoSCEtaWidth=df['phoSCEtaWidth'],
        phoSCPhiWidth=df['phoSCPhiWidth'],
        phoSCBrem=df['phoSCBrem'],
        phohasPixelSeed=df['phohasPixelSeed'],
        phoEleVeto=df['phoEleVeto'],
        phoR9=df['phoR9'],
        phoHoverE=df['phoHoverE'],
        phoESEffSigmaRR=df['phoESEffSigmaRR'],
        phoSigmaIEtaIEtaFull5x5=df['phoSigmaIEtaIEtaFull5x5'],
        phoSigmaIEtaIPhiFull5x5=df['phoSigmaIEtaIPhiFull5x5'],
        phoSigmaIPhiIPhiFull5x5=df['phoSigmaIPhiIPhiFull5x5'],
        phoE2x2Full5x5=df['phoE2x2Full5x5'],
        phoE5x5Full5x5=df['phoE5x5Full5x5'],
        phoR9Full5x5=df['phoR9Full5x5'],
        phoPFChIso=df['phoPFChIso'],
        phoPFPhoIso=df['phoPFPhoIso'],
        phoPFNeuIso=df['phoPFNeuIso'],
        phoPFChWorstIso=df['phoPFChWorstIso'],
        phoIDMVA=df['phoIDMVA'],
        phoFiredSingleTrgs=df['phoFiredSingleTrgs'],
        phoFiredDoubleTrgs=df['phoFiredDoubleTrgs'],
        phoFiredTripleTrgs=df['phoFiredTripleTrgs'],
        phoFiredL1Trgs=df['phoFiredL1Trgs'],
        phoSeedTime=df['phoSeedTime'],
        phoSeedEnergy=df['phoSeedEnergy'],
        phoxtalBits=df['phoxtalBits'],
        phoIDbit=df['phoIDbit'],
        phoScale_stat_up=df['phoScale_stat_up'],
        phoScale_stat_dn=df['phoScale_stat_dn'],
        phoScale_syst_up=df['phoScale_syst_up'],
        phoScale_syst_dn=df['phoScale_syst_dn'],
        phoScale_gain_up=df['phoScale_gain_up'],
        phoScale_gain_dn=df['phoScale_gain_dn'],
        phoResol_rho_up=df['phoResol_rho_up'],
        phoResol_rho_dn=df['phoResol_rho_dn'],
        phoResol_phi_up=df['phoResol_phi_up'],
        phoResol_phi_dn=df['phoResol_phi_dn'],
        )

    ####################################
    # get total of events
    ####################################
    output['cutflow']['all_events'] += df.size

    ####################################
    # trigger selection
    ####################################
    trigger_mask = np.bitwise_and(np.right_shift(df["HLTEleMuX"], 8), 1).astype(bool) # HLT_Mu17_Photon30_* - Run2016
    output['cutflow']['trigger'] += trigger_mask.sum()
    muons = muons[trigger_mask]
    photons = photons[trigger_mask]

    ####################################
    # dimuon selection
    ####################################

    # muon id + iso should go here!!!

    # at least two good muons
    two_muons = (muons.counts >= 2)
    output['cutflow']['two_muons'] += two_muons.sum()
    muons = muons[two_muons]
    photons = photons[two_muons]

    # load dimuons with oposite charge
    dimuons = muons.distincts()
    opposite_charge_dimuons = (dimuons.i0['charge'] * dimuons.i1['charge'] == -1)
    dimuons = dimuons[opposite_charge_dimuons]
    opposite_charge = (dimuons.counts >= 1)
    output['cutflow']['opposite_charge'] += opposite_charge.sum()
    muons = muons[opposite_charge]
    photons = photons[opposite_charge]
    dimuons = dimuons[opposite_charge]

    # apply dimuon mass cut
    muon_mass_cut_dimuons = (dimuons.mass > 8.0) & (dimuons.mass < 11.0)
    dimuons = dimuons[muon_mass_cut_dimuons]
    dimuon_mass_cut = (dimuons.counts >= 1)
    output['cutflow']['dimuon_mass_cut'] += dimuon_mass_cut.sum()
    dimuons = dimuons[dimuon_mass_cut]
    muons = muons[dimuon_mass_cut]
    photons = photons[dimuon_mass_cut]

    # save dimuon mass ntuples for the upsilon fit
    output['dimuon_mass_for_upsilon_fit'] += processor.column_accumulator((dimuons.mass).flatten())

    ####################################
    # photon selection
    ####################################

    # photon id + iso should go here!!!

    # at least one good photon
    one_photon = (photons.counts >= 1)
    output['cutflow']['one_photon'] += one_photon.sum()
    dimuons = dimuons[one_photon]
    muons = muons[one_photon]
    photons = photons[one_photon]

    ########################################
    # boson (Z or Higgs) reconstruction
    ########################################

    # build best boson
    bosons = dimuons.cross(photons)
    near_bosons = np.abs(bosons.mass - 91.1876).argmin() # z_mass = 91.1876
    bosons = bosons[near_bosons]

    # apply boson mas cut
    mass_cut_boson = (bosons.mass > 70) & (bosons.mass < 120) # z_mass cuts
    # boson_mass_cut = (bosons.mass > 100) & (bosons.mass < 150) # higgs_mass cuts
    bosons = bosons[mass_cut_boson]
    good_bosons_count = (bosons.counts >= 1)
    output['cutflow']['good_bosons_count'] += good_bosons_count.sum()
    dimuons = dimuons[good_bosons_count]
    muons = muons[good_bosons_count]
    photons = photons[good_bosons_count]
    bosons = bosons[good_bosons_count]

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
    # feed accumulator
    ########################################

    # leading muon
    output['leadmu_pt'] += processor.column_accumulator((muon_leading.pt).flatten())
    output['leadmu_eta'] += processor.column_accumulator((muon_leading.eta).flatten())
    output['leadmu_phi'] += processor.column_accumulator((muon_leading.phi).flatten())

    # trailing muon
    output['trailmu_pt'] += processor.column_accumulator((muon_trailing.pt).flatten())
    output['trailmu_eta'] += processor.column_accumulator((muon_trailing.eta).flatten())
    output['trailmu_phi'] += processor.column_accumulator((muon_trailing.phi).flatten())

    # photon
    output['photon_pt'] += processor.column_accumulator((photon.pt).flatten())
    output['photon_eta'] += processor.column_accumulator((photon.eta).flatten())
    output['photon_phi'] += processor.column_accumulator((photon.phi).flatten())
    # output['photon_energy'] += processor.column_accumulator((photon['energy']).flatten())
    output['photon_calibEnergy'] += processor.column_accumulator((photon['phoCalibE']).flatten())
    output['photon_calibEt'] += processor.column_accumulator((photon['phoCalibEt']).flatten())
    output['photon_etasc'] += processor.column_accumulator((photon['phoSCEta']).flatten())

    # upsilon
    output['upsilon_pt'] += processor.column_accumulator((upsilon.pt).flatten())
    output['upsilon_eta'] += processor.column_accumulator((upsilon.eta).flatten())
    output['upsilon_phi'] += processor.column_accumulator((upsilon.phi).flatten())
    output['upsilon_mass'] += processor.column_accumulator((upsilon.mass).flatten())

    # boson
    output['boson_pt'] += processor.column_accumulator((bosons.pt).flatten())
    output['boson_eta'] += processor.column_accumulator((bosons.eta).flatten())
    output['boson_phi'] += processor.column_accumulator((bosons.phi).flatten())
    output['boson_mass'] += processor.column_accumulator((bosons.mass).flatten())



    ########################################
    # done
    ########################################
    return output

  def postprocess(self, accumulator):
    return accumulator



