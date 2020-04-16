import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
import sys
import os
import glob
from scipy import stats
import matplotlib.colors as mcolors

#map plotting function
def make_map(mask_c,df,column,my_cmap,vmin,vmax,title,save_name):

	divnorm = mcolors.DivergingNorm(vmin=vmin, vcenter=0, vmax=vmax)
	
	fig, ax = plt.subplots(1, 1,figsize=(20,10))
	maskc.plot(ax=ax,color='white',edgecolor='black')
	df.loc[~df[column].isnull()].plot(ax=ax,column=column,cmap=my_cmap,norm=divnorm)
	ax.set_ylim([-60,85])
	ax.set_xlim([-180,180])
	plt.title(title)
	sm = plt.cm.ScalarMappable(cmap=my_cmap, norm=divnorm)
	sm._A = []
	cbar = fig.colorbar(sm,ax = ax,orientation='horizontal',fraction=0.046, pad=0.08)
	plt.savefig(save_name)
	plt.close()

experiment="T.M"	
ssps=['ssp126','ssp585']
models=['GFDL-ESM4','IPSL-CM6A-LR','MPI-ESM1-2-HR','MRI-ESM2-0','UKESM1-0-LL','CanESM5','CNRM-CM6-1','CNRM-ESM2-1','EC-Earth3','MIROC6']

#importing world maps
mask=gpd.read_file('/Users/maximiliankotz/iCloud Drive (Archive)/Documents/PIK/Data/masks/gadm36_levels_gpkg/gadm36_levels.gpkg',layer=1)
maskc=gpd.read_file('/Users/maximiliankotz/iCloud Drive (Archive)/Documents/PIK/Data/masks/gadm36_levels_gpkg/gadm36_levels.gpkg')

#divergent colormap
colors1 = plt.cm.Blues_r(np.linspace(0., 1, 100))
colors3 = plt.cm.Reds(np.linspace(0, 1, 100))
colors2 = np.array([1,1,1,1])
colors = np.vstack((colors1, colors2, colors3))
my_cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)

#list of 3 letter country identifiers (iso)
isolist=pd.unique(mask.GID_0)

#calculate regional averages and temporal standard deviation of var_T 
model_is=[0,1,2,3,4,5,6,7,8,9]

#get averages over 4 decades (2015-2025, 2035-2045, 2045-2055, 2090-2100), make plots of differences
decades=[2015,2035,2045,2090]
decaden=[(str(x) + '-' + str(x+10)) for x in decades]

#array to hold data for each region for each year in a decade of all models (10models*10years=100)
aggreg=np.zeros((2,len(decades),len(mask),100))

for y in range(2):
	ssp=ssps[y]
	folder_save='maps/' + ssp + "/"  
	print(ssp)

	for z in range(len(model_is)):
		model_i=model_is[z]
		model=models[model_i]
		print(model)
		folder="measures/" + ssp + "/" + model + '/'		
		count_reg=0

		for i in range(len(isolist)):
			iso=isolist[i]
			no_regions=len(mask.loc[mask.GID_0==iso])
			file_name=folder + 'T.M_' + ssp + '_' + model + '_' + iso + '_measure.npy'	
			file_list=glob.glob(file_name)

			if len(file_list)!=0:
				measure=np.load(file_list[0])
				for x in range(no_regions):
					for j in range(len(decades)):
						d1=decades[j]-2015	
						aggreg[y,j,count_reg+x,z*10:(z+1)*10]=np.mean(measure[1,x,d1:d1+10,:],axis=-1)
			else:
				for x in range(no_regions):
					for j in range(len(decades)):
						aggreg[y,j,count_reg+x,z*10:(z+1)*10]=np.nan

			count_reg+=no_regions

for y in range(2):
	ssp=ssps[y]
	#plot var_T changes
	vmax=0.4
	vmin=-0.4
	folder_save='maps/' + ssp + '/' + 'model_aggr/'
	for i in range(3):
		mask['var_T_delta' + str(i)]=np.mean(aggreg[y,i+1,:,:],axis=-1)-np.mean(aggreg[y,0,:,:],axis=-1)
		column='var_T_delta' + str(i)
		title='Regional d2d T variability difference between ' + decaden[i+1] + ' and ' + decaden[0]
		save_name=folder_save + 'delta' + str(i) + '_var_T_' + ssp + '_' + experiment + '.png'
		make_map(maskc,mask,column,my_cmap,vmin,vmax,title,save_name)

	

