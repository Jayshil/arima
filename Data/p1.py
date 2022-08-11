import numpy as np
import matplotlib.pyplot as plt
import os
import juliet

dataset = juliet.load(input_folder=os.getcwd() + '/Data/juliet')
res = dataset.fit(sampler='dynesty')

model, model_uerr, model_derr, comps = res.lc.evaluate('CHEOPS', return_err=True,\
     return_components=True, all_samples=True)
errs = (model_uerr-model_derr)/2

tim, fl, fle = dataset.times_lc['CHEOPS'], dataset.data_lc['CHEOPS'], dataset.errors_lc['CHEOPS']

resid = fl-model
resid_err = np.sqrt((errs**2) + (fle**2))

plt.errorbar(tim, resid, yerr=resid_err, fmt='.')
plt.axhline(0.0, ls='--', zorder=10)
plt.show()

f1 = open(os.getcwd() + '/Data/residuals.dat', 'w')
for i in range(len(tim)):
    f1.write(str(tim[i]) + '\t' + str(resid[i]) + '\t' + str(resid_err[i]) + '\n')
f1.close()