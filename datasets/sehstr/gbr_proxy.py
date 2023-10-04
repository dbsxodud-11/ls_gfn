"""
  Gradient boosted regressor sEH proxy model.
  Trained on neural net proxy's predictions on
  34M molecules from block18, stop6.
  Attains pearsonr=0.90 on data set.
"""

import pickle
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


class sEH_GBR_Proxy:
  def __init__(self, args):
    with open('datasets/sehstr/sehstr_gbtr.pkl', 'rb') as f:
      self.model = pickle.load(f)

    assert args.blocks_file == 'datasets/sehstr/block_18.json'
    blocks = pd.read_json(args.blocks_file)
    self.num_blocks = len(blocks)
    
    self.symbols = '0123456789abcdefghijklmnopqrstuvwxyz' + \
              'ABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\()*+,-./:;<=>?@[\]^_`{|}~'

  def predict_state(self, state):
    x_ft = self.featurize(state.content)
    return self.model.predict(x_ft)[0]

  def featurize(self, string):
    x_ft = np.concatenate([self.symbol_ohe(c) for c in string])
    return x_ft.reshape(1, -1)

  def symbol_ohe(self, symbol):
    zs = np.zeros(self.num_blocks)
    zs[self.symbols.index(symbol)] = 1.0
    return zs


def test():
  from attrdict import AttrDict
  args = {'blocks_file': 'datasets/sehstr/block_18.json'}
  args = AttrDict(args)
  model = sEH_GBR_Proxy(args)
  
  test_string = '012345'
  pred = model.model.predict(model.featurize(test_string))
  print(pred)
  return

if __name__ == '__main__':
  test()