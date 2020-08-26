import coffea
import coffea.processor as processor
import numpy as np
from pprint import pprint


# def accumulator_merger(accum_list):
#   out = accum_list[0]
#   for key in out.keys():
#     if type(out[key]) != coffea.processor.accumulator.column_accumulator:
#       for acc in accum_list[1:]:
#         out[key] += acc[key]
#     else:
#       out[key] = processor.column_accumulator(np.concatenate([acc[key].value for acc in accum_list]))
#   return out




def accumulator_merger(accum_list):
  # define a buff as the fist accumulator
  out = accum_list[0]
  keys_to_merge = {}
  # loop over the keys of the first accumulator and fill the keys_to_merge with the columns founded
  for key in out.keys():
    if type(out[key]) == coffea.processor.accumulator.column_accumulator:
      keys_to_merge[key] = [0]

  # loop over the rest of the accumulators and, if key is NOT a a column, add to out (buffer)
  # else add key and accum. index to dict of columns to be merged
  for idx_acc, acc in enumerate(accum_list[1:]):
    for key, value in acc.items():
      if type(acc[key]) != coffea.processor.accumulator.column_accumulator:
        if key not in out:
            if isinstance(value, coffea.processor.AccumulatorABC):
              out[key] = value.identity()
            else:
              raise ValueError
        out[key] += acc[key]
      else:
        if key in keys_to_merge:
          keys_to_merge[key].append(idx_acc+1)
        else:
          keys_to_merge[key] = [idx_acc+1]
  # loop over columns to be merged and concatenate them
  for key in keys_to_merge.keys():
    buff_acc_list = [acc[key].value for idx, acc in enumerate(accum_list) if idx in keys_to_merge[key]]
    out[key] = processor.column_accumulator(np.concatenate(buff_acc_list))
  
  return out




#########################
# Test
#########################
if __name__ == '__main__':
  acc_a = processor.dict_accumulator({
    'def1': processor.defaultdict_accumulator(float),
    'col': processor.column_accumulator(np.array([])),
  })
  acc_a["def1"]["dummy"] += 10
  acc_a["col"] += processor.column_accumulator(np.array([1,2,3]))

  acc_b = processor.dict_accumulator({
    'def1': processor.defaultdict_accumulator(float),
    'def2': processor.defaultdict_accumulator(float),
  })
  acc_b["def1"]["dummy"] += 7
  acc_b["def2"]["dummy"] += 4

  acc_c = processor.dict_accumulator({
    'def2': processor.defaultdict_accumulator(float),
    'col': processor.column_accumulator(np.array([])),
  })
  acc_c["def2"]["dummy"] += 23
  acc_c["col"] += processor.column_accumulator(np.array([7,8,9]))


  abc = accumulator_merger([acc_c, acc_a, acc_b])

  print("Merged acc:")
  pprint(abc)

  print("\nExpected result:")
  print(r"abc['col'] = [1,2,3,7,8,9]")
  print(r"abc['def1']['dummy'] = 17")
  print(r"abc['def2']['dummy'] = 27")

  
   