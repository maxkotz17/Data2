import os
import numpy as np


models=['GFDL-ESM4','IPSL-CM6A-LR','MPI-ESM1-2-HR','MRI-ESM2-0','UKESM1-0-LL','CanESM5','CNRM-CM6-1','CNRM-ESM2-1','EC-Earth3','MIROC6']
ssps=['ssp126','ssp585']

model_is=[1,2,3,4,5,6]

for i in range(1):
	ssp=ssps[i+1]
	for j in range(len(model_is)):
		model=models[model_is[j]]

		scrape="scp 'maxkotz@cluster.pik-potsdam.de:~/Data2/measures/temp/" + ssp + "/" + model + "/" + "T.M_" + ssp + "_" + model + "_???_measure.npy' " 
		
		scrape=scrape + "measures/" + ssp + "/" + model + "/"

		os.system(scrape)
 






