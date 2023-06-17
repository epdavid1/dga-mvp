from tsai.all import *
import pandas as pd
import numpy as np

tst_noise = load_learner('tst_noise')
tst_test = pd.read_csv('test.csv')

start = time.time()
tst_noise.get_X_preds(tst_test.iloc[0].to_numpy().reshape(1,1,10001))[2].astype(int)
print(time.time()-start)
