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

def sub_sample(field,longz,latz,sub):
	#number of data entries along x and y axis of data
	x_no=field.shape[0]
	y_no=field.shape[1]
	new_x_no=int(np.floor(x_no/sub))
	new_y_no=int(np.floor(y_no/sub))
	#numpy arrays to hold new field and new axes
	fieldn=np.zeros((new_x_no,new_y_no,field.shape[2]))
	fieldn_count=np.zeros((new_x_no,new_y_no))
	longn=np.zeros((new_x_no))
	latn=np.zeros((new_y_no))
	#begin subsampling
	for i in range (x_no):
		for j in range(y_no):
			index=int((i-i%sub)/sub)
			indey=int((j-j%sub)/sub)
			fieldn[index,indey,:]=fieldn[index,indey,:]+field[i,j,:]
			longn[index]=longn[index]+longz[i]
			latn[indey]=latn[indey]+latz[j]
			fieldn_count[index,indey]=fieldn_count[index,indey]+1

#	fieldf[np.where(fieldn_count==0),:,:]=np.nan
	fieldf=np.divide(fieldn[:,:,:],fieldn_count[:,:,np.newaxis])
	longf=np.divide(longn,np.sum(fieldn_count[:,:],1))
	latf=np.divide(latn,np.sum(fieldn_count[:,:],0))
	return [fieldf,longf,latf]

colors1 = plt.cm.Blues_r(np.linspace(0., 1, 100))
colors3 = plt.cm.Reds(np.linspace(0, 1, 100))
colors2 = np.array([1,1,1,1])
colors = np.vstack((colors1, colors2, colors3))
my_cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)

degree_sign= u'\N{DEGREE SIGN}'


#load data
folder='LFCA/GSWP3/'

LFP=[]
LFC=[]
for i in range(4):
	LFP.append(np.load(folder + 'gswp3_var_T_cont_LFP_' + str(i) + '.npy').transpose())
	LFC.append(np.load(folder + 'gswp3_var_T_cont_LFC_' + str(i) + '.npy'))

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

time=np.linspace(1901,2000,len(LFC[0]))		

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

		smoothed=tukey(LFC[i*2+j],0.5,120)
		ax2=fig.add_subplot(spec[i*2+1,j])
		ax2.plot(time,LFC[i*2+j],alpha=0.4,c='gray')	
		ax2.plot(time,smoothed,alpha=1,c='k')
		ax2.set_ylim(-2,2)
		ax2.set_xlabel('Time')
		ax2.set_ylabel('Standard Deviations')
		ax2.set_title('Low Frequency Component ' + str(i*2+j))

axins=inset_axes(ax,width='2%',height='100%',loc='center right',bbox_to_anchor=(0.1,0.65,1,1),bbox_transform=ax.transAxes,borderpad=0,)
sm = plt.cm.ScalarMappable(cmap=my_cmap, norm=divnorm)
cbar=fig.colorbar(sm,cax=axins,orientation='vertical')
cbar.set_ticks([vmin,0,vmax])
labels=[str(vmin) + degree_sign + 'C','0' + degree_sign + 'C',str(vmax) + degree_sign + 'C']
cbar.set_ticklabels(labels)

plt.tight_layout()
plt.savefig('LFCA_plots/GSWP3/GSWP3_panel.png',bbox_inches='tight')
plt.close()

#plot decade differences based on LFCA of ERA5	
decades=[1901,1951,1991,2001]
decaden=[(str(x) + '-' + str(x+10)) for x in decades]

LFP_decades=np.zeros((LFP[0].shape[0],LFP[0].shape[1],4))

for i in range(len(decades)):        
	LFP_decades[:,:,i]=LFP[0][:,:]*np.mean(np.real(LFC[0])[(decades[i]-1901)*12:(decades[i]-1891)*12])
	print(np.mean(np.real(LFC[0])[(decades[i]-1901)*12:(decades[i]-1901)*12]))

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

axins=inset_axes(ax,width='100%',height='5%',loc='lower center',bbox_to_anchor=(0.52,-0.1,1,1),bbox_transform=ax.transAxes,borderpad=0,)
sm = plt.cm.ScalarMappable(cmap=my_cmap, norm=divnorm)
cbar=fig.colorbar(sm,cax=axins,orientation='horizontal')
cbar.set_ticks([vmin,vmin/2,0,vmax/2,vmax])
labels=[str(vmin) + degree_sign + 'C',str(vmin/2) + degree_sign + 'C','0' + degree_sign + 'C',str(vmax/2) + degree_sign + 'C',str(vmax) + degree_sign + 'C']
cbar.set_ticklabels(labels)

plt.savefig('LFCA_plots/GSWP3/GSWP3_lfca_decades.png',bbox_inches='tight') 
plt.close()

#plot percentage differences based on LFCA of ERA5
folder='decades/'

T_var_decades=np.load(folder + 'ERA5_decades.npy')

T_var_decades=np.swapaxes(T_var_decades,0,1)

[T_var_decs,lon,lat]=sub_sample(T_var_decades,np.linspace(1,180,180),np.linspace(1,360,360),2)

vmin=-10
vmax=10
divnorm = mcolors.DivergingNorm(vmin=vmin, vcenter=0, vmax=vmax)

fig=plt.figure(constrained_layout=True,figsize=(20,8))
widths=[10,10]
heights=[5,5]
spec=fig.add_gridspec(ncols=2,nrows=2,width_ratios=widths,height_ratios=heights)
for i in range(len(decades)-1):
        ax=fig.add_subplot(spec[int(i/2),i%2])
        #map0.contourf(xx,yy,LFP_decades[:,:,i+1]-LFP_decades[:,:,0],15,cmap=my_cmap,norm=divnorm,ax=ax)
        map0.pcolormesh(xx,yy,100*np.divide((LFP_decades[:,:,i+1]-LFP_decades[:,:,0]),T_var_decs[:,:,0]),cmap=my_cmap,norm=divnorm,ax=ax)
        map0.drawcoastlines(ax=ax)
        title=decaden[i+1] + ' from ' + decaden[0]
        ax.set_title(title)

axins=inset_axes(ax,width='100%',height='5%',loc='lower center',bbox_to_anchor=(0.52,-0.1,1,1),bbox_transform=ax.transAxes,borderpad=0,)
sm = plt.cm.ScalarMappable(cmap=my_cmap, norm=divnorm)
cbar=fig.colorbar(sm,cax=axins,orientation='horizontal')
cbar.set_ticks([vmin,vmin/2,0,vmax/2,vmax])
labels=[str(vmin) + '%',str(vmin/2) + '%','0' + '%',str(vmax/2) + '%',str(vmax) + '%']
cbar.set_ticklabels(labels)

plt.savefig('LFCA_plots/ERA5/ERA5_lfca_decades_perc_diff.png',bbox_inches='tight')
plt.close()

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

time=np.linspace(1979,2019,len(LFC[0]))
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
axd.set_title('2009-2019 from 1979-1989, estimated from LFC-0')
axd.set_ylim([-60,80])

axins=inset_axes(axd,width='66%',height='5%',loc='lower center',bbox_to_anchor=(0,-0.1,1,1),bbox_transform=axd.transAxes,borderpad=0,)
sm = plt.cm.ScalarMappable(cmap=my_cmap, norm=divnorm)
cbar=fig.colorbar(sm,cax=axins,orientation='horizontal')
cbar.set_ticks([vmin,0,vmax])
labels=[str(vmin) + degree_sign + 'C','0' + degree_sign + 'C',str(vmax) + degree_sign + 'C']
cbar.set_ticklabels(labels)

plt.savefig('LFCA_plots/ERA5/ERA5_process.png',bbox_layout='tight')
plt.close()
