import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors 
from mpl_toolkits.basemap import Basemap  
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from scipy import signal
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

colors1 = plt.cm.Blues_r(np.linspace(0., 1, 100))
colors3 = plt.cm.Reds(np.linspace(0, 1, 100))
colors2 = np.array([1,1,1,1])
colors = np.vstack((colors1, colors2, colors3))
my_cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)

degree_sign= u'\N{DEGREE SIGN}'

#decades to sample and plot
decades=[1979,1989,1999,2009]
decaden=[(str(x) + '-' + str(x+10)) for x in decades]

folder='decades/'

T_var_decades=np.load(folder + 'ERA5_decades.npy')		

T_var_decades=np.swapaxes(T_var_decades,0,1)

#create map
lon_axis=np.linspace(-179.5,179.5,360)                                                                                                                        
lat_axis=np.linspace(-89.5,89.5,180)
lat_axis=np.flip(lat_axis)
xx, yy = np.meshgrid(lon_axis,lat_axis)
map0=Basemap(projection='cyl') 

#set colour scale
vmin=-0.5
vmax=0.5
divnorm = mcolors.DivergingNorm(vmin=vmin, vcenter=0, vmax=vmax)

fig=plt.figure(constrained_layout=True,figsize=(20,9))
widths=[10,10]
heights=[5,5]
spec=fig.add_gridspec(ncols=2,nrows=2,width_ratios=widths,height_ratios=heights)
for i in range(len(decades)-1):
	ax=fig.add_subplot(spec[int(i/2),i%2])
	#map0.contourf(xx,yy,T_var_decades[:,:,i+1]-T_var_decades[:,:,0],15,cmap=my_cmap,norm=divnorm,ax=ax)
	map0.pcolormesh(xx,yy,T_var_decades[:,:,i+1]-T_var_decades[:,:,0],cmap=my_cmap,norm=divnorm,ax=ax)
	map0.drawcoastlines(ax=ax)
	title=decaden[i+1] + ' from ' + decaden[0]
	ax.set_title(title)
	
axins=inset_axes(ax,width='100%',height='5%',loc='lower center',bbox_to_anchor=(0.52,-0.1,1,1),bbox_transform=ax.transAxes,borderpad=0,)
sm = plt.cm.ScalarMappable(cmap=my_cmap, norm=divnorm)
cbar=fig.colorbar(sm,cax=axins,orientation='horizontal')
cbar.set_ticks([vmin,vmin/2,0,vmax/2,vmax])
labels=[str(vmin) + degree_sign + 'C',str(vmin/2) + degree_sign + 'C','0' + degree_sign + 'C',str(vmax/2) + degree_sign + 'C',str(vmax) + degree_sign + 'C']
cbar.set_ticklabels(labels)
plt.savefig('decade_plots/' + 'ERA5_decades.png',bbox_inches='tight') 
plt.close()

# plot multi-decadal averages of day-to-day temperature variability to compare to averages from CMIP6 ensemble
#for all seasons

colors1 = plt.cm.Blues_r(np.linspace(0., 1, 100))
colors3 = plt.cm.Reds(np.linspace(0, 1, 100))
colors2 = np.array([1,1,1,1])
colors = np.vstack((colors1, colors2, colors3))
my_cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)

vmin=-0.000000001
vmax=6
divnorm = mcolors.DivergingNorm(vmin=vmin, vcenter=0, vmax=vmax)

fig=plt.figure(constrained_layout=False,figsize=(21,13))
widths=[10,10]
heights=[4,4,4]
spec=fig.add_gridspec(ncols=2,nrows=3,width_ratios=widths,height_ratios=heights,wspace=0.1,hspace=0.1)

lon_axis=np.linspace(-179.5,179.5,360)
lat_axis=np.linspace(-89.5,89.5,180)
lat_axis=np.flip(lat_axis)
xx, yy = np.meshgrid(lon_axis,lat_axis)
map0=Basemap(projection='cyl')

ERA5_T_var=np.load('decades/ERA5_decades_call.npy')
ERA5_T_var=np.swapaxes(ERA5_T_var,0,1)

ax=fig.add_subplot(spec[0,0])
map0.pcolormesh(xx,yy,np.mean(ERA5_T_var,axis=-1),cmap=my_cmap,norm=divnorm,ax=ax)
map0.drawcoastlines(ax=ax)
title='ERA5 across all seasons'
ax.set_title(title,fontsize=17)

ERA5_DJF=np.load('decades/ERA5_decades_cDJF.npy')
ERA5_DJF=np.swapaxes(ERA5_DJF,0,1)

ax1=fig.add_subplot(spec[1,0])
map0.pcolormesh(xx,yy,np.mean(ERA5_DJF,axis=-1),cmap=my_cmap,norm=divnorm,ax=ax1)
map0.drawcoastlines(ax=ax1)
title='ERA5 DJF'
ax1.set_title(title,fontsize=17)

ERA5_JJA=np.load('decades/ERA5_decades_cJJA.npy')
ERA5_JJA=np.swapaxes(ERA5_JJA,0,1)

ax2=fig.add_subplot(spec[2,0])
map0.pcolormesh(xx,yy,np.mean(ERA5_JJA,axis=-1),cmap=my_cmap,norm=divnorm,ax=ax2)
map0.drawcoastlines(ax=ax2)
title='ERA5 JJA'
ax2.set_title(title,fontsize=17)

model_avs=np.zeros((180,360,3))
models=['GFDL-ESM4','IPSL-CM6A-LR','MPI-ESM1-2-HR','MRI-ESM2-0','UKESM1-0-LL','CanESM5','CNRM-CM6-1','CNRM-ESM2-1','EC-Earth3','MIROC6']
for i in range(len(models)):
	x=np.load('decades/historical/T_std_full_hi_' + models[i] + '_decades.npy')
	x=np.swapaxes(x,0,1)
	model_avs[:,:,0]+=np.mean(x[:,:,-4:],axis=-1)
	y=np.load('decades/historical/T_std_full_hi_DJF_' + models[i] + '_decades.npy')
	y=np.swapaxes(y,0,1)
	model_avs[:,:,1]+=np.mean(y[:,:,-4:],axis=-1)
	z=np.load('decades/historical/T_std_full_hi_JJA_' + models[i] + '_decades.npy')
	z=np.swapaxes(z,0,1)
	model_avs[:,:,2]+=np.mean(z[:,:,-4:],axis=-1)

model_avs=model_avs/len(models)

ax3=fig.add_subplot(spec[0,1])
map0.pcolormesh(xx,yy,model_avs[:,:,0],cmap=my_cmap,norm=divnorm,ax=ax3)
map0.drawcoastlines(ax=ax3)
title='CMIP6 ensemble all seasons'
ax3.set_title(title,fontsize=17)

ax4=fig.add_subplot(spec[1,1])
map0.pcolormesh(xx,yy,model_avs[:,:,1],cmap=my_cmap,norm=divnorm,ax=ax4)
map0.drawcoastlines(ax=ax4)
title='CMIP6 ensemble DJF'
ax4.set_title(title,fontsize=17)

ax5=fig.add_subplot(spec[2,1])
map0.pcolormesh(xx,yy,model_avs[:,:,2],cmap=my_cmap,norm=divnorm,ax=ax5)
map0.drawcoastlines(ax=ax5)
title='CMIP6 ensemble JJA'
ax5.set_title(title,fontsize=17)

axins=inset_axes(ax5,width='100%',height='5%',loc='lower center',bbox_to_anchor=(-0.55,-0.1,1,1),bbox_transform=ax5.transAxes,borderpad=0,)
sm = plt.cm.ScalarMappable(cmap=my_cmap, norm=divnorm)
cbar=fig.colorbar(sm,cax=axins,orientation='horizontal')
cbar.set_ticks([0,vmax/2,vmax])
labels=['0' + degree_sign + 'C',str(vmax/2) + degree_sign + 'C',str(vmax) + degree_sign + 'C']
cbar.set_ticklabels(labels)
cbar.ax.tick_params(labelsize=17)
plt.savefig('decade_plots/comparison.png')
plt.close()

#plot ERA5 trends
lon_axis=np.linspace(-179.5,179.5,360)
lat_axis=np.linspace(-89.5,89.5,180)
lat_axis=np.flip(lat_axis)
xx, yy = np.meshgrid(lon_axis,lat_axis)
map0=Basemap(projection='cyl')

fig=plt.figure(constrained_layout=False,figsize=(31,10))
widths=[10,10,10]
heights=[3,3,3]
spec=fig.add_gridspec(ncols=3,nrows=3,width_ratios=widths,height_ratios=heights,wspace=0.1,hspace=0.1)

ERA5trends=np.load('trends/ERA5_trend_stats.npy')
ERA5trends=np.swapaxes(ERA5trends,0,1)

vmin=-0.01
vmax=0.01
divnorm = mcolors.DivergingNorm(vmin=vmin, vcenter=0, vmax=vmax)

ERA5trends[:,:,0][np.where(ERA5trends[:,:,2]>0.1)]=np.nan

ax=fig.add_subplot(spec[0,0])
map0.pcolormesh(xx,yy,ERA5trends[:,:,0],cmap=my_cmap,norm=divnorm,ax=ax)
map0.drawcoastlines(ax=ax)
ax.set_title('ERA5 in day-to-day T var')

DJFtrends=np.load('trends/ERA5_DJF_trend_stats.npy')
DJFtrends=np.swapaxes(DJFtrends,0,1)
DJFtrends[:,:,0][np.where(DJFtrends[:,:,2]>0.1)]=np.nan

ax2=fig.add_subplot(spec[1,0])
map0.pcolormesh(xx,yy,DJFtrends[:,:,0],cmap=my_cmap,norm=divnorm,ax=ax2)
map0.drawcoastlines(ax=ax2)
ax2.set_title('ERA5 DJF in day-to-day T var')

JJAtrends=np.load('trends/ERA5_JJA_trend_stats.npy')
JJAtrends=np.swapaxes(JJAtrends,0,1)
JJAtrends[:,:,0][np.where(JJAtrends[:,:,2]>0.1)]=np.nan

ax4=fig.add_subplot(spec[2,0])
map0.pcolormesh(xx,yy,JJAtrends[:,:,0],cmap=my_cmap,norm=divnorm,ax=ax4)
map0.drawcoastlines(ax=ax4)
ax4.set_title('ERA5 JJA in day-to-day T var')

axins=inset_axes(ax4,width='70%',height='5%',loc='lower center',bbox_to_anchor=(0,-0.1,1,1),bbox_transform=ax4.transAxes,borderpad=0,)
sm = plt.cm.ScalarMappable(cmap=my_cmap, norm=divnorm)
cbar=fig.colorbar(sm,cax=axins,orientation='horizontal')
cbar.set_ticks([vmin,0,vmax])
labels=[str(vmin) + degree_sign + 'C/year','0',str(vmax) + degree_sign + 'C/year']
cbar.set_ticklabels(labels)
#

vmin=-0.5
vmax=0.5
divnorm = mcolors.DivergingNorm(vmin=vmin, vcenter=0, vmax=vmax)

ax1=fig.add_subplot(spec[0,1])
map0.pcolormesh(xx,yy,30*ERA5trends[:,:,0],cmap=my_cmap,norm=divnorm,ax=ax1)
map0.drawcoastlines(ax=ax1)
ax1.set_title('ERA5 in day-to-day T var')

ax3=fig.add_subplot(spec[1,1])
map0.pcolormesh(xx,yy,30*DJFtrends[:,:,0],cmap=my_cmap,norm=divnorm,ax=ax3)
map0.drawcoastlines(ax=ax3)
ax3.set_title('ERA5 DJF in day-to-day T var')

ax5=fig.add_subplot(spec[2,1])
map0.pcolormesh(xx,yy,30*JJAtrends[:,:,0],cmap=my_cmap,norm=divnorm,ax=ax5)
map0.drawcoastlines(ax=ax5)
ax5.set_title('ERA5 JJA in day-to-day T var')

axins=inset_axes(ax5,width='70%',height='5%',loc='lower center',bbox_to_anchor=(0,-0.1,1,1),bbox_transform=ax5.transAxes,borderpad=0,)
sm = plt.cm.ScalarMappable(cmap=my_cmap, norm=divnorm)
cbar=fig.colorbar(sm,cax=axins,orientation='horizontal')
cbar.set_ticks([vmin,0,vmax])
labels=[str(vmin) + degree_sign + 'C','0',str(vmax) + degree_sign + 'C']
cbar.set_ticklabels(labels)

vmin=-20
vmax=20
divnorm = mcolors.DivergingNorm(vmin=vmin, vcenter=0, vmax=vmax)

ax6=fig.add_subplot(spec[0,2])
map0.pcolormesh(xx,yy,3000*ERA5trends[:,:,0]/T_var_decades[:,:,0],cmap=my_cmap,norm=divnorm,ax=ax6)
map0.drawcoastlines(ax=ax6)
ax6.set_title('ERA5 in day-to-day T var')

ax7=fig.add_subplot(spec[1,2])
map0.pcolormesh(xx,yy,3000*DJFtrends[:,:,0]/ERA5_DJF[:,:,0],cmap=my_cmap,norm=divnorm,ax=ax7)
map0.drawcoastlines(ax=ax7)
ax7.set_title('ERA5 DJF in day-to-day T var')

ax8=fig.add_subplot(spec[2,2])
map0.pcolormesh(xx,yy,3000*JJAtrends[:,:,0]/ERA5_DJF[:,:,0],cmap=my_cmap,norm=divnorm,ax=ax8)
map0.drawcoastlines(ax=ax8)
ax8.set_title('ERA5 JJA in day-to-day T var')

axins=inset_axes(ax8,width='70%',height='5%',loc='lower center',bbox_to_anchor=(0.05,-0.1,1,1),bbox_transform=ax8.transAxes,borderpad=0,)
sm = plt.cm.ScalarMappable(cmap=my_cmap, norm=divnorm)
cbar=fig.colorbar(sm,cax=axins,orientation='horizontal')
cbar.set_ticks([vmin,0,vmax])
labels=[str(vmin) + '%','0',str(vmax) + '%']
cbar.set_ticklabels(labels)

plt.savefig('trend_plots/ERA5_trend_seas.png')
plt.close()

#same as above but without the seasons
lon_axis=np.linspace(-179.5,179.5,360)
lat_axis=np.linspace(-89.5,89.5,180)
lat_axis=np.flip(lat_axis)
xx, yy = np.meshgrid(lon_axis,lat_axis)
map0=Basemap(projection='cyl')

fig=plt.figure(constrained_layout=False,figsize=(12,13))
widths=[10]
heights=[4,4,4]
spec=fig.add_gridspec(ncols=1,nrows=3,width_ratios=widths,height_ratios=heights,wspace=0.1,hspace=0.1)

ERA5trends=np.load('trends/ERA5_trend_stats.npy')
ERA5trends=np.swapaxes(ERA5trends,0,1)

vmin=-0.02
vmax=0.02
divnorm = mcolors.DivergingNorm(vmin=vmin, vcenter=0, vmax=vmax)

ERA5trends[:,:,0][np.where(ERA5trends[:,:,2]>0.1)]=np.nan

ax=fig.add_subplot(spec[0])
map0.pcolormesh(xx,yy,ERA5trends[:,:,0],cmap=my_cmap,norm=divnorm,ax=ax)
map0.drawcoastlines(ax=ax)
#ax.set_title('Trend',fontsize=17)

axins=inset_axes(ax,width='2%',height='80%',loc='center right',bbox_to_anchor=(0.05,0,1,1),bbox_transform=ax.transAxes,borderpad=0,)
sm = plt.cm.ScalarMappable(cmap=my_cmap, norm=divnorm)
cbar=fig.colorbar(sm,cax=axins,orientation='vertical')
cbar.set_ticks([vmin,0,vmax])
labels=[str(vmin) + degree_sign + 'C/year','0',str(vmax) + degree_sign + 'C/year']
cbar.set_ticklabels(labels)
cbar.ax.tick_params(labelsize=15)
vmin=-1
vmax=0.5
divnorm = mcolors.DivergingNorm(vmin=vmin, vcenter=0, vmax=vmax)

ax1=fig.add_subplot(spec[1])
map0.pcolormesh(xx,yy,30*ERA5trends[:,:,0],cmap=my_cmap,norm=divnorm,ax=ax1)
map0.drawcoastlines(ax=ax1)
#ax1.set_title('30 year change',fontsize=17)

axins=inset_axes(ax1,width='2%',height='80%',loc='center right',bbox_to_anchor=(0.05,0,1,1),bbox_transform=ax1.transAxes,borderpad=0,)
sm = plt.cm.ScalarMappable(cmap=my_cmap, norm=divnorm)
cbar=fig.colorbar(sm,cax=axins,orientation='vertical')
cbar.set_ticks([vmin,0,vmax])
labels=[str(vmin) + degree_sign + 'C','0',str(vmax) + degree_sign + 'C']
cbar.set_ticklabels(labels)
cbar.ax.tick_params(labelsize=15)

vmin=-20
vmax=20
divnorm = mcolors.DivergingNorm(vmin=vmin, vcenter=0, vmax=vmax)

ax2=fig.add_subplot(spec[2])
map0.pcolormesh(xx,yy,3000*ERA5trends[:,:,0]/T_var_decades[:,:,0],cmap=my_cmap,norm=divnorm,ax=ax2)
map0.drawcoastlines(ax=ax2)
#ax2.set_title('30 year percentage change',fontsize=17)

axins=inset_axes(ax2,width='2%',height='80%',loc='center right',bbox_to_anchor=(0.05,0,1,1),bbox_transform=ax2.transAxes,borderpad=0,)
sm = plt.cm.ScalarMappable(cmap=my_cmap, norm=divnorm)
cbar=fig.colorbar(sm,cax=axins,orientation='vertical')
cbar.set_ticks([vmin,0,vmax])
labels=[str(vmin) + degree_sign + '%','0',str(vmax) + degree_sign + '%']
cbar.set_ticklabels(labels)
cbar.ax.tick_params(labelsize=15)
plt.savefig('trend_plots/ERA5_trends.png')#,bbox_inches='tight')
plt.close()


