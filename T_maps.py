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

#get averages over 4 decades (2015-2025, 2035-2045, 2045-2055, 2090-2100)
decades=['2015-2025','2035-2045','2045-2055','2090-2100']

#list of 3 letter country identifiers (iso)
isolist=pd.unique(mask.GID_0)

#calculate regional averages and temporal standard deviation of var_T 
model_is=[0,7,8,9]

for y in range(2):
	ssp=ssps[y]
	folder_save='maps/' + ssp + "/"
	
	for z in range(len(model_is)):
		model=models[model_is[z]]
		folder="measures/" + ssp + "/" + model + '/'
		print(folder)
		mean_T_mean=[]
		mean_T_std=[]
		var_T_mean=[]
		var_T_std=[]
	
		for i in range(len(isolist)):
			iso=isolist[i]
			no_regions=len(mask.loc[mask.GID_0==iso])
			file_name=folder + 'T.M_' + ssp + '_' + model + '_' + iso + '_measure.npy'      
			file_list=glob.glob(file_name)
			if len(file_list)!=0:
				measure=np.load(file_list[0])
				for x in range(no_regions):
					mean_T_mean.append(np.mean(measure[0,x,:,:]))	
					mean_T_std.append(np.std(np.mean(measure[0,x,:,:],axis=1)))
					var_T_mean.append(np.mean(measure[1,x,:,:]))
					var_T_std.append(np.std(np.mean(measure[1,x,:,:],axis=1)))
			else:
				for x in range(no_regions):
					mean_T_mean.append(np.nan)
					mean_T_std.append(np.nan)
					var_T_mean.append(np.nan)
					var_T_std.append(np.nan)

		mask['mean_T_mean']=mean_T_mean
		mask['mean_T_std']=mean_T_std
		mask['var_T_mean']=var_T_mean
		mask['var_T_std']=var_T_std

		
		#plot mean_T
		vmax=30
		vmin=-10
		column='mean_T_mean'
		title=''
		save_name=folder_save + 'mean_T_mean/' + model + '_' + ssp + '_' + experiment + '_mean_T_mean.png'
		make_map(maskc,mask,column,my_cmap,vmin,vmax,title,save_name)

		vmax=5
		vmin=-5
		column='mean_T_std'
		title=''
		save_name=folder_save + 'mean_T_std/' + model + '_' + ssp + '_' + experiment + '_mean_T_std.png'

		#plot var_T
		vmax=6
		vmin=-0.00000000001
		column='var_T_mean'
		title=''
		save_name=folder_save + 'var_T_mean/' + model + '_' + ssp + '_' + experiment + '_var_T_mean.png'
		make_map(maskc,mask,column,my_cmap,vmin,vmax,title,save_name)

		vmax=0.5
		vmin=-0.00000000001
		column='var_T_std'
		title=''
		save_name=folder_save + 'var_T_std/'+ model + '_' + ssp + '_' + experiment + '_var_T_std.png'
		make_map(maskc,mask,column,my_cmap,vmin,vmax,title,save_name)

#get averages over 4 decades (2015-2025, 2035-2045, 2045-2055, 2090-2100), make plots of differences
model_is=[0,7,8,9]
decades=[2015,2035,2045,2090]
decaden=[(str(x) + '-' + str(x+10)) for x in decades]

for y in range(2):
	ssp=ssps[y]
	folder_save='maps/' + ssp + "/"  
	print(ssp)
		
	for z in range(len(model_is)):
		model_i=model_is[z]
		model=models[model_i]
		print(model)
		folder="measures/" + ssp + "/" + model + '/'		

		mean_T=[]
		var_T=[]
		var_JFM=[]
		var_JAS=[]		
		for i in range(4):
			mean_T.append([])
			var_T.append([])
			var_JFM.append([])
			var_JAS.append([])

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
						mean_T[j].append(np.mean(measure[0,x,d1:d1+10,:]))
						var_T[j].append(np.mean(measure[1,x,d1:d1+10,:]))
						var_JFM[j].append(np.mean(measure[1,x,d1:d1+10,0:3]))
						var_JAS[j].append(np.mean(measure[1,x,d1:d1+10,6:9]))
			else:
				for x in range(no_regions):
					for j in range(len(decades)):
						mean_T[j].append(np.nan)
						var_T[j].append(np.nan)
						var_JFM[j].append(np.nan)
						var_JAS[j].append(np.nan)

		for j in range(len(decades)):	
			mask['mean_T_' + decaden[j]]=mean_T[j]
			mask['var_T_' + decaden[j]]=var_T[j]
			mask['var_JFM_' + decaden[j]]=var_JFM[j]
			mask['var_JAS_' + decaden[j]]=var_JAS[j]
			if j>0:	
				mask['mean_T_delta' + str(j-1)]=mask['mean_T_' + decaden[j]]-mask['mean_T_' + decaden[j-1]]
				mask['var_T_delta' + str(j-1)]=mask['var_T_' + decaden[j]]-mask['var_T_' + decaden[j-1]]
				mask['var_JFM_delta' + str(j-1)]=mask['var_JFM_' + decaden[j]]-mask['var_JFM_' + decaden[j-1]]
				mask['var_JAS_delta' + str(j-1)]=mask['var_JAS_' + decaden[j]]-mask['var_JAS_' + decaden[j-1]]

		#plot mean T changes	
		vs=[1,2,4]
		for i in range(0):#3):
			vmin=-vs[i]
			vmax=vs[i]
			column='mean_T_delta' + str(i)
			title='Mean T difference between ' + decaden[i+1] + ' and ' + decaden[0]
			save_name=folder_save + 'mean_T_changes/delta' + str(i) + '_' + model + '_' + ssp + '_' + experiment + '.png'
			make_map(maskc,mask,column,my_cmap,vmin,vmax,title,save_name)

		#plot var_T changes
		vmax=0.4
		vmin=-0.4

		for i in range(3):
			column='var_T_delta' + str(i)
			title='Regional d2d T variability difference between ' + decaden[i+1] + ' and ' + decaden[0]
			save_name=folder_save + 'T_var_changes/delta' + str(i) + '_var_T_' + model + '_' + ssp + '_' + experiment + '.png'
			make_map(maskc,mask,column,my_cmap,vmin,vmax,title,save_name)

		#plot seasonal var_T changes
		vmax=0.4
		vmin=-0.4
		
		for i in range(3):
			column='var_JFM_delta' + str(i)
			title='Regional JFM d2d T variability difference between ' + decaden[i+1] + ' and ' + decaden[0]
			save_name=folder_save + 'JFM_var_changes/delta' + str(i) + '_JFM_var_' + model + '_' + ssp + '_' + experiment + '.png'
			make_map(maskc,mask,column,my_cmap,vmin,vmax,title,save_name)
		
			column='var_JAS_delta' + str(i)  
			title='Regional JAS d2d T variability difference between ' + decaden[i+1] + ' and ' + decaden[0]
			save_name=folder_save + 'JAS_var_changes/delta' + str(i) + '_JAS_var_' + model + '_' + ssp + '_' + experiment + '.png'
			make_map(maskc,mask,column,my_cmap,vmin,vmax,title,save_name)
	
	

