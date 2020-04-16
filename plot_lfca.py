import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors 
from mpl_toolkits.basemap import Basemap  

models=['GFDL-ESM4','IPSL-CM6A-LR','MPI-ESM1-2-HR','MRI-ESM2-0','UKESM1-0-LL','CanESM5','CNRM-CM6-1','CNRM-ESM2-1','EC-Earth3','MIROC6']
ssps=['ssp126','ssp585']

model_is=[1]
ssp=ssps[1]

model=models[model_is[0]]
folder='LFCA/' + ssp + '/' + models[model_is[0]] + '/' 


colors1 = plt.cm.Blues_r(np.linspace(0., 1, 100))
colors3 = plt.cm.Reds(np.linspace(0, 1, 100))
colors2 = np.array([1,1,1,1])
colors = np.vstack((colors1, colors2, colors3))
my_cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)

LFP=[]
LFC=[]

for i in range(4):
	LFP.append(np.load(folder + 'var_T_LFP_' + str(i) + '.npy').transpose())
	LFC.append(np.load(folder + 'var_T_LFC_' + str(i) + '.npy'))

LFC[0]=-LFC[0]
LFP[0]=-LFP[0]

lon_axis=np.linspace(-179,179,180)                                                                                                                                        
lat_axis=np.linspace(-89,89,90) 
lat_axis=np.flip(lat_axis)
xx, yy = np.meshgrid(lon_axis,lat_axis)

map0=Basemap(projection='cyl') 

vmin=-1
vmax=1

divnorm = mcolors.DivergingNorm(vmin=vmin, vcenter=0, vmax=vmax)

fig, axes = plt.subplots(2,2,figsize=(20,10))
for i in range(2):
	for j in range(2):
		map0.pcolormesh(xx,yy,LFP[i*2+j],cmap=my_cmap,norm=divnorm,ax=axes[i,j])
		map0.drawcoastlines(ax=axes[i,j])
		axes[i,j].set_title('Low Frequency Pattern ' + str(i*2+j))
		fig.suptitle('LFCA of D2D_T_var 2015-2100: ' + ssp + ', ' + model)

plt.savefig('LFCA_plots/' + ssp + '/' + model + '/' + 'LFP_' + ssp + '_' + model + '.png')
plt.close()

time=np.linspace(2015,2100,len(LFC[0]))
fig, axes = plt.subplots(2,2,figsize=(20,10))

for i in range(2):
	for j in range(2):
		axes[i,j].plot(time,LFC[i*2+j])
		axes[i,j].set_xlabel('Time')
		axes[i,j].set_title('Low Frequency Component ' + str(i*2+j))
		fig.suptitle('LFCA of D2D_T_var 2015-2100: ' + ssp + ', ' + model)

plt.savefig('LFCA_plots/' + ssp + '/' + model + '/' + 'LFC_' + ssp + '_' + model + '.png')
plt.close()
		


