import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors 
from mpl_toolkits.basemap import Basemap  
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from scipy import signal
from mpl_toolkits.axes_grid1.inset_locator import inset_axes  

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

model_is=[0,1,2,3,4,5,6,7,8,9]
#plot panel of components and time series

for z in range(len(model_is)):
	model=models[model_is[z]]
	print(model)
	#load data
	folder='LFCA/historical/' + model + '/' 

	LFP=[]
	LFC=[]
	for i in range(4):
		LFP.append(np.load(folder + 'his_var_T_LFP_' + str(i) + '.npy').transpose())
		LFC.append(np.load(folder + 'his_var_T_LFC_' + str(i) + '.npy'))

	#swith the sign of first component so that the trend is growing
	LFC[0]=-LFC[0]
	LFP[0]=-LFP[0]

	#create map
	lon_axis=np.linspace(-179,179,180)                                                                                                                        
	lat_axis=np.linspace(-89,89,90) 
	lat_axis=np.flip(lat_axis)
	xx, yy = np.meshgrid(lon_axis,lat_axis)
	map0=Basemap(projection='cyl') 

	#set colour scale
	vmin=-1
	vmax=1
	divnorm = mcolors.DivergingNorm(vmin=vmin, vcenter=0, vmax=vmax)

	time=np.linspace(1950,2015,len(LFC[0]))		

	fig=plt.figure(constrained_layout=True,figsize=(20,12))
	widths=[10,10]
	heights=[5,1,5,1]
	spec=fig.add_gridspec(ncols=2,nrows=4,width_ratios=widths,height_ratios=heights)
	for i in range(2):
		for j in range(2):
			ax=fig.add_subplot(spec[i*2,j])
			#map0.contourf(xx,yy,LFP[i*2+j],15,cmap=my_cmap,norm=divnorm,ax=ax)
			map0.pcolormesh(xx,yy,LFP[i*2+j],cmap=my_cmap,norm=divnorm,ax=ax)
			map0.drawcoastlines(ax=ax)
	#		ax.set_title('Low Frequency Pattern ' + str(i*2+j))

			smoothed=tukey(LFC[i*2+j],0.5,120)
			ax2=fig.add_subplot(spec[i*2+1,j])
			ax2.plot(time,LFC[i*2+j],alpha=0.4,c='gray')	
			ax2.plot(time,smoothed,alpha=1,c='k')
			ax2.set_ylim(-2,2)
			ax2.set_xlim(time[0],time[-1])
			ax2.set_xlabel('Time')
			ax2.set_ylabel('Standard Deviations')
			ax2.set_title('Low Frequency Component ' + str(i*2+j))

	axins=inset_axes(ax,width='2%',height='100%',loc='center right',bbox_to_anchor=(0.1,0.65,1,1),bbox_transform=ax.transAxes,borderpad=0,)
	sm = plt.cm.ScalarMappable(cmap=my_cmap, norm=divnorm)
	cbar=fig.colorbar(sm,cax=axins,orientation='vertical')
	cbar.set_ticks([vmin,0,vmax])
	labels=[str(vmin) + degree_sign + 'C','0' + degree_sign + 'C',str(vmax) + degree_sign + 'C']
	cbar.set_ticklabels(labels)

	plt.savefig('LFCA_plots/historical/' + model + '/LFCA_panel_his_' + model + '.png',bbox_inches='tight')
	plt.close()
	
	fig=plt.figure(constrained_layout=True,figsize=(20,12))
	widths=[10,10]
	heights=[5,1,5,1]
	spec=fig.add_gridspec(ncols=2,nrows=4,width_ratios=widths,height_ratios=heights)
	for i in range(2):
		for j in range(2):
			ax=fig.add_subplot(spec[i*2,j])
			map0.contourf(xx,yy,LFP[i*2+j],15,cmap=my_cmap,norm=divnorm,ax=ax)
			#map0.pcolormesh(xx,yy,LFP[i*2+j],cmap=my_cmap,norm=divnorm,ax=ax)
			map0.drawcoastlines(ax=ax)
			#ax.set_title('Low Frequency Pattern ' + str(i*2+j))

			smoothed=tukey(LFC[i*2+j],0.5,120)
			ax2=fig.add_subplot(spec[i*2+1,j])
			ax2.plot(time,LFC[i*2+j],alpha=0.4,c='gray')
			ax2.plot(time,smoothed,alpha=1,c='k')
			ax2.set_ylim(-2,2)
			ax2.set_xlim(time[0],time[-1])
			ax2.set_xlabel('Time')
			ax2.set_ylabel('Standard Deviations')
			ax2.set_title('Low Frequency Component ' + str(i*2+j))

	axins=inset_axes(ax,width='2%',height='100%',loc='center right',bbox_to_anchor=(0.1,0.65,1,1),bbox_transform=ax.transAxes,borderpad=0,)
	sm = plt.cm.ScalarMappable(cmap=my_cmap, norm=divnorm)
	cbar=fig.colorbar(sm,cax=axins,orientation='vertical')
	cbar.set_ticks([vmin,0,vmax])
	labels=[str(vmin) + degree_sign + 'C','0' + degree_sign + 'C',str(vmax) + degree_sign + 'C']
	cbar.set_ticklabels(labels)
		
	plt.savefig('LFCA_plots/historical/' + model + '/LFCA_panel_hi_' + model + '_cont.png',bbox_inches='tight')
	plt.close()

#plot all time series and all spatial patterns together
fig=plt.figure(constrained_layout=True,figsize=(20,20))
widths=[7,7]
heights=[2,2,2,2,2]
spec=fig.add_gridspec(ncols=2,nrows=5,width_ratios=widths,height_ratios=heights)
for z in range(len(model_is)):
	model=models[model_is[z]]                
	folder='LFCA/historical/' + model + '/'

	LFC=[]
	for i in range(4):
		LFC.append(np.load(folder + 'his_var_T_LFC_' + str(i) + '.npy'))

	#swith the sign of first component so that the trend is growing
	LFC[0]=-LFC[0]
	smoothed=tukey(LFC[0],0.5,120)

	time=np.linspace(1950,2015,len(LFC[0]))

	ax=fig.add_subplot(spec[int(z/2),z%2])
	ax.plot(time,LFC[0],alpha=0.2,c='gray')
	ax.plot(time,smoothed,alpha=1,c='k')
	ax.set_ylim(-2,2)
	ax.set_xlim(time[0],time[-1])
	ax.set_xlabel('Time')
	ax.set_ylabel('Standard Deviations')
	ax.set_title(model)

plt.savefig('LFCA_plots/historical/LFC-0_his_time_series.png')
plt.close()

#plot all spatial patterns together
fig=plt.figure(constrained_layout=True,figsize=(20,20))
widths=[7,7]
heights=[2,2,2,2,2]
spec=fig.add_gridspec(ncols=2,nrows=5,width_ratios=widths,height_ratios=heights)
vmin=-1
vmax=1
divnorm = mcolors.DivergingNorm(vmin=vmin, vcenter=0, vmax=vmax)
for z in range(len(model_is)):
	model=models[model_is[z]]
	folder='LFCA/historical/' + model + '/'
	
	LFP=[]
	LFC=[]
	for i in range(4):
		LFP.append(np.load(folder + 'his_var_T_LFP_' + str(i) + '.npy').transpose())

	LFP[0]=-LFP[0]
	
	lon_axis=np.linspace(-179,179,180)
	lat_axis=np.linspace(-89,89,90)
	lat_axis=np.flip(lat_axis)
	xx, yy = np.meshgrid(lon_axis,lat_axis)
	map0=Basemap(projection='cyl')	

	ax=fig.add_subplot(spec[int(z/2),z%2])

	map0.contourf(xx,yy,LFP[0],15,cmap=my_cmap,norm=divnorm,ax=ax)
#	map0.pcolormesh(xx,yy,LFP[0],cmap=my_cmap,norm=divnorm,ax=ax)
	map0.drawcoastlines(ax=ax)
	ax.set_title(model)

axins=inset_axes(ax,width='100%',height='5%',loc='lower center',bbox_to_anchor=(-0.52,-0.1,1,1),bbox_transform=ax.transAxes,borderpad=0,)
sm = plt.cm.ScalarMappable(cmap=my_cmap, norm=divnorm)
cbar=fig.colorbar(sm,cax=axins,orientation='horizontal')
cbar.set_ticks([vmin,0,vmax])
labels=[str(vmin) + degree_sign + 'C','0' + degree_sign + 'C',str(vmax) + degree_sign + 'C']
cbar.set_ticklabels(labels)

plt.savefig('LFCA_plots/historical/LFC-0_his_spatial_patterns_cont.png',bbox_inches='tight')
plt.close()

#plot all end of century differences together on one plot
fig=plt.figure(constrained_layout=False,figsize=(15,12))
widths=[7,7]
heights=[2,2,2,2,2]
spec=fig.add_gridspec(ncols=2,nrows=5,width_ratios=widths,height_ratios=heights)
vmin=-0.5
vmax=0.5
divnorm = mcolors.DivergingNorm(vmin=vmin, vcenter=0, vmax=vmax)
for z in range(len(model_is)):
	model=models[model_is[z]]
	folder='LFCA/historical/' + model + '/'

	LFP=[]
	LFC=[]
	
	for i in range(4):
		LFP.append(np.load(folder + 'his_var_T_LFP_' + str(i) + '.npy').transpose())
		LFC.append(np.load(folder + 'his_var_T_LFC_' + str(i) + '.npy'))
	LFP[0]=-LFP[0]
	LFC[0]=-LFC[0]
	
	decade_diff=LFP[0]*(np.mean(np.real(LFC[0][-120:]))-np.mean(np.real(LFC[0][:120])))
	
	lon_axis=np.linspace(-179,179,180)
	lat_axis=np.linspace(-89,89,90)
	lat_axis=np.flip(lat_axis)
	xx, yy = np.meshgrid(lon_axis,lat_axis)
	map0=Basemap(projection='cyl')

	vmin=-0.5
	vmax=0.5
	divnorm = mcolors.DivergingNorm(vmin=vmin, vcenter=0, vmax=vmax)
	
	ax=fig.add_subplot(spec[int(z/2),z%2])
#	map0.contourf(xx,yy,LFP[0],15,cmap=my_cmap,norm=divnorm,ax=ax)
	map0.pcolormesh(xx,yy,LFP[0],cmap=my_cmap,norm=divnorm,ax=ax)
	map0.drawcoastlines(ax=ax)
	ax.set_title(model)
	ax.set_ylim([-60,80])

axins=inset_axes(ax,width='100%',height='5%',loc='lower center',bbox_to_anchor=(-0.52,-0.1,1,1),bbox_transform=ax.transAxes,borderpad=0,)
sm = plt.cm.ScalarMappable(cmap=my_cmap, norm=divnorm)
cbar=fig.colorbar(sm,cax=axins,orientation='horizontal')
cbar.set_ticks([vmin,0,vmax])
labels=[str(vmin) + degree_sign + 'C','0' + degree_sign + 'C',str(vmax) + degree_sign + 'C']
cbar.set_ticklabels(labels)
plt.savefig('LFCA_plots/historical/LFC-0_his_dec_diff.png',bbox_inches='tight')
plt.close()

#plot panel of LFC-0 pattern, time series and final decade difference
ssp='ssp585'
for z in range(len(model_is)):
	model=models[model_is[z]]
	folder='LFCA/' + ssp + '/' + model + '/'
	
	LFP=[]
	LFC=[]
	
	for i in range(4):
		LFP.append(np.load(folder + 'var_T_LFP_' + str(i) + '.npy').transpose())
		LFC.append(np.load(folder + 'var_T_LFC_' + str(i) + '.npy'))
	LFP[0]=-LFP[0]
	LFC[0]=-LFC[0]
	
	decade_diff=LFP[0]*(np.mean(np.real(LFC[0][-120:]))-np.mean(np.real(LFC[0][:120])))

	fig=plt.figure(constrained_layout=False,figsize=(17.05,4.4))
	widths=[7,10]
	heights=[3,1]
	spec=fig.add_gridspec(ncols=2,nrows=2,width_ratios=widths,height_ratios=heights,hspace=0.4,wspace=0.05)
	vmin=-0.5
	vmax=0.5
	divnorm = mcolors.DivergingNorm(vmin=vmin, vcenter=0, vmax=vmax)
		        
	lon_axis=np.linspace(-179,179,180)
	lat_axis=np.linspace(-89,89,90)
	lat_axis=np.flip(lat_axis)
	xx, yy = np.meshgrid(lon_axis,lat_axis)
	map0=Basemap(projection='cyl')

	ax=fig.add_subplot(spec[0,0])
	map0.pcolormesh(xx,yy,LFP[0],cmap=my_cmap,norm=divnorm,ax=ax)
	map0.drawcoastlines(ax=ax)
	ax.set_ylim([-60,80])
	ax.set_title('Low Frequency Pattern 0 (LFP-0)')

	axins=inset_axes(ax,width='66%',height='5%',loc='lower center',bbox_to_anchor=(0,-0.1,1,1),bbox_transform=ax.transAxes,borderpad=0,)
	sm = plt.cm.ScalarMappable(cmap=my_cmap, norm=divnorm)
	cbar=fig.colorbar(sm,cax=axins,orientation='horizontal')
	cbar.set_ticks([vmin,0,vmax])
	labels=[str(vmin) + degree_sign + 'C','0' + degree_sign + 'C',str(vmax) + degree_sign + 'C']
	cbar.set_ticklabels(labels)

	time=np.linspace(2015,2100,len(LFC[0]))
	smoothed=tukey(LFC[0],0.5,120)
	axt=fig.add_subplot(spec[1,0])
	axt.plot(time,LFC[0],alpha=0.4,c='gray')
	axt.plot(time,smoothed,alpha=1,c='k')
	axt.set_xlabel('Time')
	axt.set_ylabel('Standard Deviation')
	axt.set_ylim([-2,2])
	axt.set_xlim([time[0],time[-1]])
	axt.set_title('Low Frequency Component 0 (LFC-0)')
	axt.set_aspect("auto")

	axd=fig.add_subplot(spec[:,1])
	map0.pcolormesh(xx,yy,decade_diff,cmap=my_cmap,norm=divnorm,ax=axd)
	map0.drawcoastlines(ax=axd)
	axd.set_title('2090-2100 from 2015-2025, estimated from LFC-0')
	axd.set_ylim([-60,80])

	axins=inset_axes(axd,width='66%',height='5%',loc='lower center',bbox_to_anchor=(0,-0.1,1,1),bbox_transform=axd.transAxes,borderpad=0,)
	sm = plt.cm.ScalarMappable(cmap=my_cmap, norm=divnorm)
	cbar=fig.colorbar(sm,cax=axins,orientation='horizontal')
	cbar.set_ticks([vmin,0,vmax])
	labels=[str(vmin) + degree_sign + 'C','0' + degree_sign + 'C',str(vmax) + degree_sign + 'C']
	cbar.set_ticklabels(labels)
	
	plt.savefig('LFCA_plots/' + ssp + '/' + model + '/' + 'LFCA_process.png',bbox_inches='tight')
	plt.close()	

