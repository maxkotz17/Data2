import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors 
from mpl_toolkits.basemap import Basemap  
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from scipy import signal

def tukey(field,alpha,points):
	if points%2==0:
		points+=1
	tukey=signal.tukey(points,alpha,sym=True)
	length=len(field)
	half=int(points/2)
	field2=np.zeros(np.shape(field))
	for i in range(length):
		mint=i-half
		maxt=i+half
		if mint<0 and maxt<(length):
			excess=abs(mint)
			tukey_m=tukey[excess:]
			shortened=field[:maxt+1]
		elif maxt>=(length) and mint>=0:
			excess=maxt-length 
			tukey_m=tukey[:-excess-1]
			shortened=field[mint:]
		elif mint<0 and maxt>=(length-1):
			excessl=abs(mint)
			excessu=maxt-length
			tukey_m=tukey[excessl:-1-excessu]
			shortened=field
		else:
			shortened=field[mint:maxt+1]
			tukey_m=tukey
		for j in range(len(tukey_m)):	
			field2[i]=field2[i]+shortened[j]*tukey_m[j]
		field2[i]=field2[i]/(sum(tukey_m))
	return(field2)

colors1 = plt.cm.Blues_r(np.linspace(0., 1, 100))
colors3 = plt.cm.Reds(np.linspace(0, 1, 100))
colors2 = np.array([1,1,1,1])
colors = np.vstack((colors1, colors2, colors3))
my_cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)

degree_sign= u'\N{DEGREE SIGN}'

models=['GFDL-ESM4','IPSL-CM6A-LR','MPI-ESM1-2-HR','MRI-ESM2-0','UKESM1-0-LL','CanESM5','CNRM-CM6-1','CNRM-ESM2-1','EC-Earth3','MIROC6']
ssps=['ssp126','ssp585']

model_is=[0,1,2,3,4,5,6,7,8,9]

#decades to sample and plot
decades=[2015,2035,2055,2075,2090]
decaden=[(str(x) + '-' + str(x+10)) for x in decades]

for y in range(2):
	ssp=ssps[y]
	print(ssp)
	for z in range(len(model_is)):
		model=models[model_is[z]]
		print(model)
		#load data
		folder='LFCA/' + ssp + '/' + model + '/' 

		LFP=[]
		LFC=[]
		for i in range(4):
			LFP.append(np.load(folder + 'var_T_LFP_' + str(i) + '.npy').transpose())
			LFC.append(np.load(folder + 'var_T_LFC_' + str(i) + '.npy'))

		#swith the sign of first component so that the trend is growing
		LFC[0]=-LFC[0]
		LFP[0]=-LFP[0]

		LFP_decades=np.zeros((LFP[0].shape[0],LFP[0].shape[1],5))

		for i in range(len(decades)):
			LFP_decades[:,:,i]=LFP[0][:,:]*np.mean(LFC[0][(decades[i]-2015)*12:(decades[i]-2005)*12])

		#create map
		lon_axis=np.linspace(-179,179,180)                                                                                                                        
		lat_axis=np.linspace(-89,89,90) 
		lat_axis=np.flip(lat_axis)
		xx, yy = np.meshgrid(lon_axis,lat_axis)
		map0=Basemap(projection='cyl') 

		#set colour scale
		vmin=-0.5
		vmax=0.5
		divnorm = mcolors.DivergingNorm(vmin=vmin, vcenter=0, vmax=vmax)

		fig=plt.figure(constrained_layout=True,figsize=(20,8))
		widths=[10,10]
		heights=[5,5]
		spec=fig.add_gridspec(ncols=2,nrows=2,width_ratios=widths,height_ratios=heights)
		for i in range(len(decades)-1):
			ax=fig.add_subplot(spec[int(i/2),i%2])
			#map0.contourf(xx,yy,LFP_decades[:,:,i+1]-LFP_decades[:,:,0],15,cmap=my_cmap,norm=divnorm,ax=ax)
			map0.pcolormesh(xx,yy,LFP_decades[:,:,i+1]-LFP_decades[:,:,0],cmap=my_cmap,norm=divnorm,ax=ax)
			map0.drawcoastlines(ax=ax)
			title=decaden[i+1] + ' from ' + decaden[0]
			ax.set_title(title)
			
		axins=inset_axes(ax,width='100%',height='5%',loc='lower center',bbox_to_anchor=(-0.52,-0.1,1,1),bbox_transform=ax.transAxes,borderpad=0,)
		sm = plt.cm.ScalarMappable(cmap=my_cmap, norm=divnorm)
		cbar=fig.colorbar(sm,cax=axins,orientation='horizontal')
		cbar.set_ticks([vmin,vmin/2,0,vmax/2,vmax])
		labels=[str(vmin) + degree_sign + 'C',str(vmin/2) + degree_sign + 'C','0' + degree_sign + 'C',str(vmax/2) + degree_sign + 'C',str(vmax) + degree_sign + 'C']
		cbar.set_ticklabels(labels)

		plt.savefig('LFCA_plots/' + ssp + '/' + model + '/LFC0_decades_' + ssp + '_' + model + '.png',bbox_inches='tight')
		plt.close()



