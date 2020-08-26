import uproot
import numpy as np

def get_pu_weight(pu_true, dataset, year):
  weight_nominal = 1.0
  weight_up = 1.0
  weight_down = 1.0

  if dataset.startswith("Run"):
    return weight_nominal, weight_up, weight_down

  pu_file = {
        "2016" : "data/pu/pu_weights_2016_from_HZZ4L.root",
        "2017" : "data/pu/pu_weights_2017_from_HZZ4L.root",
        "2018" : "data/pu/pu_weights_2018_from_HZZ4L.root",
        }

  histo, bins = uproot.open(pu_file[year])["weights"].allnumpy()
  weight_nominal = histo[np.digitize(pu_true, bins)-1]

  histo, bins = uproot.open(pu_file[year])["weights_varUp"].allnumpy()
  weight_up = histo[np.digitize(pu_true, bins)-1]

  histo, bins = uproot.open(pu_file[year])["weights_varDn"].allnumpy()
  weight_down = histo[np.digitize(pu_true, bins)-1]

  return weight_nominal, weight_up, weight_down