from coffea import hist
from coffea.analysis_objects import JaggedCandidateArray
import coffea.processor as processor
from coffea import hist
import numpy as np
from awkward import JaggedArray
import awkward
import config
from coffea.util import load, save
import random
# import subprocess
# import os 

class AnalyzerProcessor(processor.ProcessorABC):
    def __init__(self, analyzer_name):
        self._analyzer_name = analyzer_name
        ####################################
        # accumulator
        ####################################
        self._accumulator = processor.dict_accumulator({
            'cutflow': processor.defaultdict_accumulator(float),
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

        #event info
        event_info = awkward.Table(**config.get_vars_dict(df, config.evt_vars))

        # load muons
        muons = None
        if df['nMu'].size != 0:
            muons = JaggedCandidateArray.candidatesfromcounts(
                df['nMu'],
                pt=df['muPt'],
                eta=df['muEta'],
                phi=df['muPhi'],
                mass=105.6583745/1000.,
                **config.get_vars_dict(df, config.muon_vars)
                )
        else:
            muons = JaggedCandidateArray.candidatesfromcounts(
                np.array([]),
                pt=np.array([]),
                eta=np.array([]),
                phi=np.array([]),
                mass=np.array([]),
                **config.get_vars_dict(df, config.muon_vars)
                )
        # load photons
        photons = None
        if df['nPho'].size != 0:
            photons = JaggedCandidateArray.candidatesfromcounts(
                df['nPho'],
                pt=df['phoCalibEt'],
                eta=df['phoEta'],
                phi=df['phoPhi'],
                energy=df['phoCalibE'],
                **config.get_vars_dict(df, config.photon_vars)
                )
        else:
            photons = JaggedCandidateArray.candidatesfromcounts(
                np.array([]),
                pt=np.array([]),
                eta=np.array([]),
                phi=np.array([]),
                energy=np.array([]),
                **config.get_vars_dict(df, config.photon_vars)
                )
        ####################################
        # get total of events
        ####################################
        # evtWeight_sign mask
        evtWeight_sign = None
        if dataset.startswith("Run"): # Data
            evtWeight_sign = np.ones(df.size) 
        else: # MC
            evtWeight_sign = np.sign(df["genWeight"])
            
        output['cutflow']['all_events'] += np.sum(evtWeight_sign)
        output['cutflow']['all_events_unweighted'] += df.size

        ####################################
        # trigger selection
        ####################################
        trigger_mask = np.bitwise_and(np.right_shift(df["HLTEleMuX"], 8), 1).astype(bool) # HLT_Mu17_Photon30_* - Run2016
        output['cutflow']['trigger'] += (trigger_mask*evtWeight_sign).sum()
        muons = muons[trigger_mask]
        photons = photons[trigger_mask]
        event_info = event_info[trigger_mask]
        evtWeight_sign = evtWeight_sign[trigger_mask]

        ####################################
        # dimuon selection
        ####################################

        # at least two good muons
        two_muons = (muons.counts >= 2)
        output['cutflow']['two_muons'] += (two_muons*evtWeight_sign).sum()
        muons = muons[two_muons]
        photons = photons[two_muons]
        event_info = event_info[two_muons]
        evtWeight_sign = evtWeight_sign[two_muons]


        # load dimuons with oposite charge
        dimuons = muons.distincts()
        opposite_charge_dimuons = (dimuons.i0['muCharge'] * dimuons.i1['muCharge'] == -1)
        dimuons = dimuons[opposite_charge_dimuons]
        opposite_charge = (dimuons.counts >= 1)
        output['cutflow']['opposite_charge'] += (opposite_charge*evtWeight_sign).sum()
        muons = muons[opposite_charge]
        photons = photons[opposite_charge]
        event_info = event_info[opposite_charge]
        evtWeight_sign = evtWeight_sign[opposite_charge]


        ####################################
        # photon selection
        ####################################

        # at least one good photon
        one_photon = (photons.counts >= 1)
        output['cutflow']['one_photon'] += (one_photon*evtWeight_sign).sum()
        # muons = muons[one_photon]
        # photons = photons[one_photon]
        # event_info = event_info[one_photon] # one can not filter on one_photon otherwise, loose generality for the dimuon mas distribution
        # evtWeight_sign = evtWeight_sign[one_photon]

        ########################################
        # save data
        ########################################


        # event info
        evt_info_acc = processor.dict_accumulator({})
        for var in event_info.columns:
            evt_info_acc[var] = processor.column_accumulator(np.array(event_info[var]))
        evt_info_acc['one_photon'] = processor.column_accumulator(np.array(one_photon)) #save the 'one_photon' mask
        evt_info_acc['evtWeight_sign'] = processor.column_accumulator(np.array(evtWeight_sign)) #save the 'evtWeight_sign' mask
        output += evt_info_acc    


        # muons
        muon_acc = processor.dict_accumulator({})
        for var in config.muon_vars:
            muon_acc[var] = processor.column_accumulator(muons.flatten()[var])
        muon_acc["nMu"] = processor.column_accumulator(muons.counts)
        output += muon_acc


        # photons
        photon_acc = processor.dict_accumulator({})
        for var in config.photon_vars:
            photon_acc[var] = processor.column_accumulator(photons.flatten()[var])
        photon_acc["nPho"] = processor.column_accumulator(photons.counts)
        output += photon_acc

        ########################################
        # save output per job
        ########################################
        file_hash = str(random.getrandbits(128)) + str(df.size)
        save(output, "outputs/buffer/ana_" + self._analyzer_name + "_"+file_hash+".coffea")

        ########################################
        # done
        ########################################
        # return output - dummy
        return processor.dict_accumulator({
            'foo': processor.defaultdict_accumulator(float),
        })
        # return output

    def postprocess(self, accumulator):
        # # list saved outputs
        # result = subprocess.run("ls outputs/buffer/ana_" + ds["dataset_name"] + "_"+ ds["year"] + "_*.coffea", shell=True, stdout=subprocess.PIPE)

        # merged_output = processor.dict_accumulator({})
        # # memory save outputs merging
        # for f in result.stdout.decode('utf-8').splitlines():
        #     merged_output = merged_output + load(f)
        #     os.system("rm -f"+f)

        return accumulator



