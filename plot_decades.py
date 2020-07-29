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
C=degree_sign + 'C'
models=['GFDL-ESM4','IPSL-CM6A-LR','MPI-ESM1-2-HR','MRI-ESM2-0','UKESM1-0-LL','CanESM5','CNRM-CM6-1','CNRM-ESM2-1','EC-Earth3','MIROC6']
ssps=['ssp126','ssp585']

model_is=[0,1,2,3,4,5,6,7,8,9]

#decades to sample and plot
decades=[2015,2035,2055,2075,2090]
decaden=[(str(x) + '-' + str(x+10)) for x in decades]
#specific model plots
for y in range(2):
	ssp=ssps[y]
	print(ssp)
	for z in range(len(model_is)):
		model=models[model_is[z]]
		print(model)
		#load data
		folder='decades/' + ssp + '/'

		T_var_decades=np.load(folder + 'T_std_' + ssp + '_' + model + '_decades.npy')		

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

		fig=plt.figure(constrained_layout=True,figsize=(20,8))
		widths=[10,10]
		heights=[5,5]
		spec=fig.add_gridspec(ncols=2,nrows=2,width_ratios=widths,height_ratios=heights)
		for i in range(len(decades)-1):
			ax=fig.add_subplot(spec[int(i/2),i%2])
			#map0.contourf(xx,yy,LFP_decades[:,:,i+1]-LFP_decades[:,:,0],15,cmap=my_cmap,norm=divnorm,ax=ax)
			map0.pcolormesh(xx,yy,T_var_decades[:,:,i+1]-T_var_decades[:,:,0],cmap=my_cmap,norm=divnorm,ax=ax)
			map0.drawcoastlines(ax=ax)
			title=decaden[i+1] + ' from ' + decaden[0]
			ax.set_title(title)

		axins=inset_axes(ax,width='100%',height='5%',loc='lower center',bbox_to_anchor=(-0.52,-0.1,1,1),bbox_transform=ax.transAxes,borderpad=0,)
		sm = plt.cm.ScalarMappable(cmap=my_cmap, norm=divnorm)
		cbar=fig.colorbar(sm,cax=axins,orientation='horizontal')
		cbar.set_ticks([vmin,vmin/2,0,vmax/2,vmax])
		labels=[str(vmin) + degree_sign + 'C',str(vmin/2) + degree_sign + 'C','0' + degree_sign + 'C',str(vmax/2) + degree_sign + 'C',str(vmax) + degree_sign + 'C']
		cbar.set_ticklabels(labels)
		plt.savefig('decade_plots/' + ssp + '/' + 'T_var_decades_' + ssp + '_' + model + '.png',bbox_inches='tight')
		plt.close()


#specific model plots, all in one plot with just final decade difference 
fig=plt.figure(constrained_layout=True,figsize=(20,20))
widths=[7,7]
heights=[2,2,2,2,2]
spec=fig.add_gridspec(ncols=2,nrows=5,width_ratios=widths,height_ratios=heights)
vmin=-0.5
vmax=0.5
divnorm = mcolors.DivergingNorm(vmin=vmin, vcenter=0, vmax=vmax)
for y in range(2):
        ssp=ssps[y]
        print(ssp)
        fig=plt.figure(constrained_layout=True,figsize=(20,20))
        widths=[7,7]
        heights=[2,2,2,2,2]
        spec=fig.add_gridspec(ncols=2,nrows=5,width_ratios=widths,height_ratios=heights)
        vmin=-0.5
        vmax=0.5
        divnorm = mcolors.DivergingNorm(vmin=vmin, vcenter=0, vmax=vmax)
        for z in range(len(model_is)):
                model=models[model_is[z]]
                print(model)
                #load data
                folder='decades/' + ssp + '/'

                T_var_decades=np.load(folder + 'T_std_' + ssp + '_' + model + '_decades.npy')

                T_var_decades=np.swapaxes(T_var_decades,0,1)

                #create map
                lon_axis=np.linspace(-179.5,179.5,360)
                lat_axis=np.linspace(-89.5,89.5,180)
                lat_axis=np.flip(lat_axis)
                xx, yy = np.meshgrid(lon_axis,lat_axis)
                map0=Basemap(projection='cyl')


                ax=fig.add_subplot(spec[int(z/2),z%2])
                #map0.contourf(xx,yy,LFP_decades[:,:,i+1]-LFP_decades[:,:,0],15,cmap=my_cmap,norm=divnorm,ax=ax)
                map0.pcolormesh(xx,yy,T_var_decades[:,:,-1]-T_var_decades[:,:,0],cmap=my_cmap,norm=divnorm,ax=ax)
                map0.drawcoastlines(ax=ax)
                title=model + ': ' + decaden[-1] + ' from ' + decaden[0]
                ax.set_title(title)

        axins=inset_axes(ax,width='100%',height='5%',loc='lower center',bbox_to_anchor=(-0.52,-0.1,1,1),bbox_transform=ax.transAxes,borderpad=0,)
        sm = plt.cm.ScalarMappable(cmap=my_cmap, norm=divnorm)
        cbar=fig.colorbar(sm,cax=axins,orientation='horizontal')
        cbar.set_ticks([vmin,vmin/2,0,vmax/2,vmax])
        labels=[str(vmin) + degree_sign + 'C',str(vmin/2) + degree_sign + 'C','0' + degree_sign + 'C',str(vmax/2) + degree_sign + 'C',str(vmax) + degree_sign + 'C']
        cbar.set_ticklabels(labels)
        plt.savefig('decade_plots/' + ssp + '/T_var_decades_' + ssp + '_all.png',bbox_inches='tight')
        plt.close()

#model aggregate plots
for y in range(2):
	ssp=ssps[y]
	for z in range(len(model_is)):
		model=models[model_is[z]]
		folder='decades/' + ssp + '/'

		if z==0:
			x=np.load(folder + 'T_std_' + ssp + '_' + model + '_decades.npy')
			T_var_decades=np.swapaxes(x,0,1)
		else:
			x=np.load(folder + 'T_std_' + ssp + '_' + model + '_decades.npy')
			T_var_decades+=np.swapaxes(x,0,1)

	T_var_decades=T_var_decades/len(model_is)

	lon_axis=np.linspace(-179.5,179.5,360)    
	lat_axis=np.linspace(-89.5,89.5,180)
	lat_axis=np.flip(lat_axis)
	xx, yy = np.meshgrid(lon_axis,lat_axis)
	map0=Basemap(projection='cyl')

	#set colour scale
	vmin=-1
	vmax=0.5
	divnorm = mcolors.DivergingNorm(vmin=vmin, vcenter=0, vmax=vmax)

	fig=plt.figure(constrained_layout=False,figsize=(21,10))
	widths=[10,10]
	heights=[4,4]
	spec=fig.add_gridspec(ncols=2,nrows=2,width_ratios=widths,height_ratios=heights,hspace=0.002,wspace=0.05)

	for i in range(2):
		ax=fig.add_subplot(spec[int(i/2),i%2])
		#map0.contourf(xx,yy,T_var_decades[:,:,i+1]-T_var_decades[:,:,0],15,cmap=my_cmap,norm=divnorm,ax=ax)
		map0.pcolormesh(xx,yy,T_var_decades[:,:,2*i+2]-T_var_decades[:,:,0],cmap=my_cmap,norm=divnorm,ax=ax)
		map0.drawcoastlines(ax=ax)
		title=decaden[2*i+2] + ' from ' + decaden[0]
		ax.set_title(title,fontsize=17)

	axins=inset_axes(ax,width='100%',height='5%',loc='lower center',bbox_to_anchor=(-0.53,-0.1,1,1),bbox_transform=ax.transAxes,borderpad=0,)
	sm = plt.cm.ScalarMappable(cmap=my_cmap, norm=divnorm)
	cbar=fig.colorbar(sm,cax=axins,orientation='horizontal')
	cbar.set_ticks([vmin,vmin/2,0,vmax/2,vmax])
	labels=[str(vmin) + degree_sign + 'C',str(vmin/2) + degree_sign + 'C','0' + degree_sign + 'C',str(vmax/2) + degree_sign + 'C',str(vmax) + degree_sign + 'C']
	cbar.set_ticklabels(labels)
	cbar.ax.tick_params(labelsize=17)

	vmin=-50
	vmax=50
	divnorm = mcolors.DivergingNorm(vmin=vmin, vcenter=0, vmax=vmax)

	for i in range(2):
		ax=fig.add_subplot(spec[1+int(i/2),i%2])
		#map0.contourf(xx,yy,100*np.divide((T_var_decades[:,:,i+1]-T_var_decades[:,:,0]),T_var_decades[:,:,0]),15,cmap=my_cmap,norm=divnorm,ax=ax)
		map0.pcolormesh(xx,yy,100*np.divide((T_var_decades[:,:,2*i+2]-T_var_decades[:,:,0]),T_var_decades[:,:,0]),cmap=my_cmap,norm=divnorm,ax=ax)
		map0.drawcoastlines(ax=ax)
		title=decaden[2*i+2] + ' from ' + decaden[0]
		ax.set_title(title,fontsize=17)

	axins=inset_axes(ax,width='100%',height='5%',loc='lower center',bbox_to_anchor=(-0.53,-0.1,1,1),bbox_transform=ax.transAxes,borderpad=0,)
	sm = plt.cm.ScalarMappable(cmap=my_cmap, norm=divnorm)
	cbar=fig.colorbar(sm,cax=axins,orientation='horizontal')
	cbar.set_ticks([vmin,vmin/2,0,vmax/2,vmax])
	labels=[str(vmin) + '%',str(vmin/2) + '%','0' + '%',str(vmax/2) + '%',str(vmax) + '%']
	cbar.set_ticklabels(labels)
	cbar.ax.tick_params(labelsize=17)

	plt.savefig('decade_plots/' + ssp + '/' + 'T_var_decades_' + ssp + '_agg.png')#,bbox_inches='tight')
	plt.close()

#model aggregate percentage differences
for y in range(2):
	ssp=ssps[y]
	for z in range(len(model_is)):
		model=models[model_is[z]]
		folder='decades/' + ssp + '/'

		if z==0:
			x=np.load(folder + 'T_std_' + ssp + '_' + model + '_decades.npy')
			T_var_decades=np.swapaxes(x,0,1)
		else:
			x=np.load(folder + 'T_std_' + ssp + '_' + model + '_decades.npy')
			T_var_decades+=np.swapaxes(x,0,1)

	T_var_decades=T_var_decades/len(model_is)

	lon_axis=np.linspace(-179.5,179.5,360)
	lat_axis=np.linspace(-89.5,89.5,180)
	lat_axis=np.flip(lat_axis)
	xx, yy = np.meshgrid(lon_axis,lat_axis)
	map0=Basemap(projection='cyl')

	#set colour scale
	vmin=-50
	vmax=50
	divnorm = mcolors.DivergingNorm(vmin=vmin, vcenter=0, vmax=vmax)

	fig=plt.figure(constrained_layout=False,figsize=(21,9))
	widths=[10,10]
	heights=[4,4]
	spec=fig.add_gridspec(ncols=2,nrows=2,width_ratios=widths,height_ratios=heights,hspace=0.002,wspace=0.05)

	for i in range(len(decades)-1):
		ax=fig.add_subplot(spec[int(i/2),i%2])
		#map0.contourf(xx,yy,100*np.divide((T_var_decades[:,:,i+1]-T_var_decades[:,:,0]),T_var_decades[:,:,0]),15,cmap=my_cmap,norm=divnorm,ax=ax)
		map0.pcolormesh(xx,yy,100*np.divide((T_var_decades[:,:,i+1]-T_var_decades[:,:,0]),T_var_decades[:,:,0]),cmap=my_cmap,norm=divnorm,ax=ax)
		map0.drawcoastlines(ax=ax)
		title=decaden[i+1] + ' from ' + decaden[0]
		ax.set_title(title,fontsize=17)

	axins=inset_axes(ax,width='100%',height='5%',loc='lower center',bbox_to_anchor=(-0.53,-0.1,1,1),bbox_transform=ax.transAxes,borderpad=0,)
	sm = plt.cm.ScalarMappable(cmap=my_cmap, norm=divnorm)
	cbar=fig.colorbar(sm,cax=axins,orientation='horizontal')
	cbar.set_ticks([vmin,vmin/2,0,vmax/2,vmax])
	labels=[str(vmin) + '%',str(vmin/2) + '%','0' + '%',str(vmax/2) + '%',str(vmax) + '%']
	cbar.set_ticklabels(labels)
	cbar.ax.tick_params(labelsize=17)
	plt.savefig('decade_plots/' + ssp + '/' + 'T_var_decades_' + ssp + '_agg_perc.png')#,bbox_inches='tight')
	plt.close()


#model aggregate pre-industrial differences
decades=[2015,2035,2055,2075,2090]
decaden=[(str(x) + '-' + str(x+10)) for x in decades]
for y in range(2):
	ssp=ssps[y]
	for z in range(len(model_is)):
		model=models[model_is[z]]
		folder='decades/' + ssp + '/' 
		folderpi='decades/pi_control/'
		if z==0:
			x=np.load(folder + 'T_std_' + ssp + '_' + model + '_decades.npy')
			T_var_decades_ssp=np.swapaxes(x,0,1)
			w=np.load(folderpi + 'T_std_pi_control_' + model + '_decades.npy')
			T_var_decades_pi=np.swapaxes(w,0,1)
		else:
			x=np.load(folder + 'T_std_' + ssp + '_' + model + '_decades.npy')
			T_var_decades_ssp+=np.swapaxes(x,0,1)
			w=np.load(folderpi + 'T_std_pi_control_' + model + '_decades.npy')
			T_var_decades_pi+=np.swapaxes(w,0,1)

	T_var_decades_ssp=T_var_decades_ssp/len(model_is)
	T_var_decades_pi=T_var_decades_pi/len(model_is)

	lon_axis=np.linspace(-179.5,179.5,360)
	lat_axis=np.linspace(-89.5,89.5,180)
	lat_axis=np.flip(lat_axis)
	xx, yy = np.meshgrid(lon_axis,lat_axis)
	map0=Basemap(projection='cyl')

	vmin=-0.5
	vmax=0.5
	divnorm=mcolors.DivergingNorm(vmin=vmin, vcenter=0, vmax=vmax)
	
	for i in range(len(decades)):
		fig,ax=plt.subplots(constrained_layout=True,figsize=(10,8))
		map0.pcolormesh(xx,yy,T_var_decades_ssp[:,:,i]-T_var_decades_pi[:,:,2+i],cmap=my_cmap,norm=divnorm,ax=ax)
		map0.drawcoastlines(ax=ax)
		title='Difference between ' + ssp + ' and pi_control ' + decaden[i]
		ax.set_title(title)
		axins=inset_axes(ax,width='50%',height='5%',loc='lower center',bbox_to_anchor=(0,-0.1,1,1),bbox_transform=ax.transAxes,borderpad=0,)
		sm = plt.cm.ScalarMappable(cmap=my_cmap, norm=divnorm)
		cbar=fig.colorbar(sm,cax=axins,orientation='horizontal')
		cbar.set_ticks([vmin,vmin/2,0,vmax/2,vmax])
		labels=[str(vmin) + degree_sign + 'C',str(vmin/2) + degree_sign + 'C','0' + degree_sign + 'C',str(vmax/2) + degree_sign + 'C',str(vmax) + degree_sign + 'C']
		cbar.set_ticklabels(labels)
		plt.savefig('decade_plots/' + ssp + '_pi/T_var_' + ssp + '_pi_' + decaden[i] + '.png')
		plt.close()

#model aggregate historical vs pre-industrial differences
decades=[2001,2011]
decaden=[(str(x) + '-' + str(x+10)) for x in decades]
	
for z in range(len(model_is)):
	model=models[model_is[z]]
	folder='decades/historical/'
	folderpi='decades/pi_control/'
	if z==0:
		x=np.load(folder + 'T_std_historical_' + model + '_decades.npy')
		T_var_decades_hi=np.swapaxes(x,0,1)
		w=np.load(folderpi + 'T_std_pi_control_' + model + '_decades.npy')
		T_var_decades_pi=np.swapaxes(w,0,1)
	else:
		x=np.load(folder + 'T_std_historical_' + model + '_decades.npy')
		T_var_decades_hi+=np.swapaxes(x,0,1)
		w=np.load(folderpi + 'T_std_pi_control_' + model + '_decades.npy')
		T_var_decades_pi+=np.swapaxes(w,0,1)
	
T_var_decades_hi=T_var_decades_hi/len(model_is)
T_var_decades_pi=T_var_decades_pi/len(model_is)

lon_axis=np.linspace(-179.5,179.5,360)
lat_axis=np.linspace(-89.5,89.5,180)
lat_axis=np.flip(lat_axis)
xx, yy = np.meshgrid(lon_axis,lat_axis)
map0=Basemap(projection='cyl')

vmin=-0.5
vmax=0.5
divnorm=mcolors.DivergingNorm(vmin=vmin, vcenter=0, vmax=vmax)

for i in range(len(decades)):
	fig,ax=plt.subplots(constrained_layout=True,figsize=(10,8))
	map0.pcolormesh(xx,yy,T_var_decades_hi[:,:,i]-T_var_decades_pi[:,:,i],cmap=my_cmap,norm=divnorm,ax=ax)
	map0.drawcoastlines(ax=ax)
	title='Difference between historical and pi_control ' + decaden[i]
	ax.set_title(title)
	axins=inset_axes(ax,width='50%',height='5%',loc='lower center',bbox_to_anchor=(0,-0.1,1,1),bbox_transform=ax.transAxes,borderpad=0,)
	sm = plt.cm.ScalarMappable(cmap=my_cmap, norm=divnorm)
	cbar=fig.colorbar(sm,cax=axins,orientation='horizontal')
	cbar.set_ticks([vmin,vmin/2,0,vmax/2,vmax])
	labels=[str(vmin) + degree_sign + 'C',str(vmin/2) + degree_sign + 'C','0' + degree_sign + 'C',str(vmax/2) + degree_sign + 'C',str(vmax) + degree_sign + 'C']
	cbar.set_ticklabels(labels)
	plt.savefig('decade_plots/hi_pi/T_var_hi_pi_' + decaden[i] + '.png')
	plt.close()

#specific grid of 3 plots hi, ssp vs pre industrial for SI Fig 1.
fig=plt.figure(constrained_layout=True,figsize=(13,15.6))
widths=[10]
heights=[5,5,5]
spec=fig.add_gridspec(ncols=1,nrows=3,width_ratios=widths,height_ratios=heights,hspace=0.2,wspace=0)
vmin=-1
vmax=0.5
divnorm=mcolors.DivergingNorm(vmin=vmin, vcenter=0, vmax=vmax)

decades=[2001,2011]
decaden=[(str(x) + '-' + str(x+10)) for x in decades]

for z in range(len(model_is)):
        model=models[model_is[z]]
        folder='decades/historical/'
        folderpi='decades/pi_control/'
        if z==0:
                x=np.load(folder + 'T_std_historical_' + model + '_decades.npy')
                T_var_decades_hi=np.swapaxes(x,0,1)
                w=np.load(folderpi + 'T_std_pi_control_' + model + '_decades.npy')
                T_var_decades_pi=np.swapaxes(w,0,1)
        else:
                x=np.load(folder + 'T_std_historical_' + model + '_decades.npy')
                T_var_decades_hi+=np.swapaxes(x,0,1)
                w=np.load(folderpi + 'T_std_pi_control_' + model + '_decades.npy')
                T_var_decades_pi+=np.swapaxes(w,0,1)

T_var_decades_hi=T_var_decades_hi/len(model_is)
T_var_decades_pi=T_var_decades_pi/len(model_is)

lon_axis=np.linspace(-179.5,179.5,360)
lat_axis=np.linspace(-89.5,89.5,180)
lat_axis=np.flip(lat_axis)
xx, yy = np.meshgrid(lon_axis,lat_axis)
ax=fig.add_subplot(spec[0,0])
map0=Basemap(projection='cyl')
map0.pcolormesh(xx,yy,T_var_decades_hi[:,:,0]-T_var_decades_pi[:,:,0],cmap=my_cmap,norm=divnorm,ax=ax)
map0.drawcoastlines(ax=ax)

ax2=[]
for y in range(2):
	ssp=ssps[y]
	for z in range(len(model_is)):
		model=models[model_is[z]]
		folder='decades/' + ssp + '/'
		folderpi='decades/pi_control/'
		if z==0:
			x=np.load(folder + 'T_std_' + ssp + '_' + model + '_decades.npy')
			T_var_decades_ssp=np.swapaxes(x,0,1)
			w=np.load(folderpi + 'T_std_pi_control_' + model + '_decades.npy')
			T_var_decades_pi=np.swapaxes(w,0,1)
		else:
			x=np.load(folder + 'T_std_' + ssp + '_' + model + '_decades.npy')
			T_var_decades_ssp+=np.swapaxes(x,0,1)
			w=np.load(folderpi + 'T_std_pi_control_' + model + '_decades.npy')
			T_var_decades_pi+=np.swapaxes(w,0,1)

	T_var_decades_ssp=T_var_decades_ssp/len(model_is)
	T_var_decades_pi=T_var_decades_pi/len(model_is)

	lon_axis=np.linspace(-179.5,179.5,360)
	lat_axis=np.linspace(-89.5,89.5,180)
	lat_axis=np.flip(lat_axis)
	xx, yy = np.meshgrid(lon_axis,lat_axis)
	map2=Basemap(projection='cyl')
	ax2.append(fig.add_subplot(spec[1+y,0]))
	map2.pcolormesh(xx,yy,(T_var_decades_ssp[:,:,-1]-T_var_decades_pi[:,:,-1]),cmap=my_cmap,norm=divnorm,ax=ax2[y])
	map2.drawcoastlines(ax=ax2[y])
	
axins=inset_axes(ax2[0],width='2%',height='100%',loc='center right',bbox_to_anchor=(0.05,0,1,1),bbox_transform=ax2[0].transAxes,borderpad=0,)
sm = plt.cm.ScalarMappable(cmap=my_cmap, norm=divnorm)
cbar=fig.colorbar(sm,cax=axins,orientation='vertical')
cbar.set_ticks([vmin,vmin/2,0,vmax/2,vmax])
labels=[str(vmin) + C,str(vmin/2) + C,'0' + C,str(vmax/2) + C,str(vmax) + C]
cbar.set_ticklabels(labels)
cbar.ax.tick_params(labelsize=17)
plt.savefig('decade_plots/SI_fig1.png')
plt.close()

#Fig 1 but as percentages 
fig=plt.figure(constrained_layout=True,figsize=(13,15.6))
widths=[10]
heights=[5,5,5]
spec=fig.add_gridspec(ncols=1,nrows=3,width_ratios=widths,height_ratios=heights,hspace=0.2,wspace=0)
vmin=-50
vmax=50
divnorm=mcolors.DivergingNorm(vmin=vmin, vcenter=0, vmax=vmax)

decades=[2001,2011]
decaden=[(str(x) + '-' + str(x+10)) for x in decades]

for z in range(len(model_is)):
	model=models[model_is[z]]
	folder='decades/historical/'
	folderpi='decades/pi_control/'
	if z==0:
		x=np.load(folder + 'T_std_historical_' + model + '_decades.npy')
		T_var_decades_hi=np.swapaxes(x,0,1)
		w=np.load(folderpi + 'T_std_pi_control_' + model + '_decades.npy')
		T_var_decades_pi=np.swapaxes(w,0,1)
	else:
		x=np.load(folder + 'T_std_historical_' + model + '_decades.npy')
		T_var_decades_hi+=np.swapaxes(x,0,1)
		w=np.load(folderpi + 'T_std_pi_control_' + model + '_decades.npy')
		T_var_decades_pi+=np.swapaxes(w,0,1)

T_var_decades_hi=T_var_decades_hi/len(model_is)
T_var_decades_pi=T_var_decades_pi/len(model_is)

lon_axis=np.linspace(-179.5,179.5,360)
lat_axis=np.linspace(-89.5,89.5,180)
lat_axis=np.flip(lat_axis)
xx, yy = np.meshgrid(lon_axis,lat_axis)
ax=fig.add_subplot(spec[0,0])
map0=Basemap(projection='cyl')
map0.pcolormesh(xx,yy,100*np.divide(T_var_decades_hi[:,:,0]-T_var_decades_pi[:,:,0],T_var_decades_pi[:,:,0]),cmap=my_cmap,norm=divnorm,ax=ax)
map0.drawcoastlines(ax=ax)

ax2=[]
for y in range(2):
	ssp=ssps[y]
	for z in range(len(model_is)):
		model=models[model_is[z]]
		folder='decades/' + ssp + '/'
		folderpi='decades/pi_control/'
		if z==0:
			x=np.load(folder + 'T_std_' + ssp + '_' + model + '_decades.npy')
			T_var_decades_ssp=np.swapaxes(x,0,1)
			w=np.load(folderpi + 'T_std_pi_control_' + model + '_decades.npy')
			T_var_decades_pi=np.swapaxes(w,0,1)
		else:
			x=np.load(folder + 'T_std_' + ssp + '_' + model + '_decades.npy')
			T_var_decades_ssp+=np.swapaxes(x,0,1)
			w=np.load(folderpi + 'T_std_pi_control_' + model + '_decades.npy')
			T_var_decades_pi+=np.swapaxes(w,0,1)

	T_var_decades_ssp=T_var_decades_ssp/len(model_is)
	T_var_decades_pi=T_var_decades_pi/len(model_is)

	lon_axis=np.linspace(-179.5,179.5,360)
	lat_axis=np.linspace(-89.5,89.5,180)
	lat_axis=np.flip(lat_axis)
	xx, yy = np.meshgrid(lon_axis,lat_axis)
	map2=Basemap(projection='cyl')
	ax2.append(fig.add_subplot(spec[1+y,0]))
	map2.pcolormesh(xx,yy,100*np.divide(T_var_decades_ssp[:,:,-1]-T_var_decades_pi[:,:,-1],T_var_decades_pi[:,:,-1]),cmap=my_cmap,norm=divnorm,ax=ax2[y])
	map2.drawcoastlines(ax=ax2[y])

axins=inset_axes(ax2[0],width='2%',height='100%',loc='center right',bbox_to_anchor=(0.05,0,1,1),bbox_transform=ax2[0].transAxes,borderpad=0,)
sm = plt.cm.ScalarMappable(cmap=my_cmap, norm=divnorm)
cbar=fig.colorbar(sm,cax=axins,orientation='vertical')
cbar.set_ticks([vmin,vmin/2,0,vmax/2,vmax])
labels=[str(vmin) + '%',str(vmin/2) + '%','0' + '%',str(vmax/2) + '%',str(vmax) + '%']
cbar.set_ticklabels(labels)
cbar.ax.tick_params(labelsize=17)
plt.savefig('decade_plots/Fig1_percent.png')
plt.close()

#model aggregate seasonal plots
seass=['DJF','JJA']
for y in range(2):
        ssp=ssps[y]
	for w in range(2):
		seas=seass[w]

		for z in range(len(model_is)):
			model=models[model_is[z]]
			folder='decades/' + ssp + '/'

			if z==0:
				x=np.load(folder + 'T_' + seas + 'std_' + ssp + '_' + model + '_decades.npy')
				T_var_decades=np.swapaxes(x,0,1)
			else:
				x=np.load(folder + 'T_' + seas + 'std_' + ssp + '_' + model + '_decades.npy')
				T_var_decades+=np.swapaxes(x,0,1)

		T_var_decades=T_var_decades/len(model_is)

		lon_axis=np.linspace(-179.5,179.5,360)
		lat_axis=np.linspace(-89.5,89.5,180)
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
		spec=fig.add_gridspec(ncols=2,nrows=2,width_ratios=widths,height_ratios=heights,hspace=0,wspace=0)

		for i in range(len(decades)-1):
			ax=fig.add_subplot(spec[int(i/2),i%2])
			#map0.contourf(xx,yy,T_var_decades[:,:,i+1]-T_var_decades[:,:,0],15,cmap=my_cmap,norm=divnorm,ax=ax)
			map0.pcolormesh(xx,yy,T_var_decades[:,:,i+1]-T_var_decades[:,:,0],cmap=my_cmap,norm=divnorm,ax=ax)
			map0.drawcoastlines(ax=ax)
			title=decaden[i+1] + ' from ' + decaden[0]
			ax.set_title(title,fontsize=20)

		axins=inset_axes(ax,width='100%',height='5%',loc='lower center',bbox_to_anchor=(-0.52,-0.1,1,1),bbox_transform=ax.transAxes,borderpad=0,)
		sm = plt.cm.ScalarMappable(cmap=my_cmap, norm=divnorm)
		cbar=fig.colorbar(sm,cax=axins,orientation='horizontal')
		cbar.set_ticks([vmin,vmin/2,0,vmax/2,vmax])
		labels=[str(vmin) + degree_sign + 'C',str(vmin/2) + degree_sign + 'C','0' + degree_sign + 'C',str(vmax/2) + degree_sign + 'C',str(vmax) + degree_sign + 'C']
		cbar.set_ticklabels(labels)
		cbar.ax.tick_params(labelsize=20)
		plt.savefig('decade_plots/' + ssp + '/' + seas + '_T_var_decades_' + ssp + '_agg.png',bbox_inches='tight')
		plt.close()

#specific seasonal plot for Fig S3. 
seass=['DJF','JJA']
for y in range(2):
	ssp=ssps[y]

	fig=plt.figure(constrained_layout=False,figsize=(13,14))
	widths=[10]
	heights=[5,5]
	spec=fig.add_gridspec(ncols=1,nrows=2,width_ratios=widths,height_ratios=heights,hspace=0.2,wspace=0)

	for w in range(2):
		seas=seass[w]
		
		for z in range(len(model_is)):
			model=models[model_is[z]]
			folder='decades/' + ssp + '/'
			if z==0:
				x=np.load(folder + 'T_' + seas + 'std_' + ssp + '_' + model + '_decades.npy')
				T_var_decades=np.swapaxes(x,0,1)
			else:
				x=np.load(folder + 'T_' + seas + 'std_' + ssp + '_' + model + '_decades.npy')
				T_var_decades+=np.swapaxes(x,0,1)
			
		T_var_decades=T_var_decades/len(model_is)
		
                lon_axis=np.linspace(-179.5,179.5,360)
		lat_axis=np.linspace(-89.5,89.5,180)
		lat_axis=np.flip(lat_axis)
		xx, yy = np.meshgrid(lon_axis,lat_axis)
		map0=Basemap(projection='cyl')

		#set colour scale
		vmin=-1
		vmax=0.5
		divnorm = mcolors.DivergingNorm(vmin=vmin, vcenter=0, vmax=vmax)
	
		ax=fig.add_subplot(spec[w,0])
		map0.pcolormesh(xx,yy,T_var_decades[:,:,-1]-T_var_decades[:,:,0],cmap=my_cmap,norm=divnorm,ax=ax)
		map0.drawcoastlines(ax=ax)
		title=seas + ': 2090-2100 from 2015-2025'
		ax.set_title(title,fontsize=17)
						

	axins=inset_axes(ax,width='70%',height='4%',loc='lower center',bbox_to_anchor=(0,-0.1,1,1),bbox_transform=ax.transAxes,borderpad=0,)
	sm = plt.cm.ScalarMappable(cmap=my_cmap, norm=divnorm)
	cbar=fig.colorbar(sm,cax=axins,orientation='horizontal')
	cbar.set_ticks([vmin,vmin/2,0,vmax/2,vmax])
	labels=[str(vmin) + degree_sign + 'C',str(vmin/2) + degree_sign + 'C','0' + degree_sign + 'C',str(vmax/2) + degree_sign + 'C',str(vmax) + degree_sign + 'C']
	cbar.set_ticklabels(labels)
	cbar.ax.tick_params(labelsize=17)
	plt.savefig('decade_plots/' + ssp + '/seas_T_var_decades_' + ssp + '_agg.png')#,bbox_inches='tight')
	plt.close()


#model specific seasonal plots
#specific model plots, all in one plot with just final decade difference 
fig=plt.figure(constrained_layout=True,figsize=(20,20))
widths=[7,7]
heights=[2,2,2,2,2]
spec=fig.add_gridspec(ncols=2,nrows=5,width_ratios=widths,height_ratios=heights)
vmin=-0.5
vmax=0.5
divnorm = mcolors.DivergingNorm(vmin=vmin, vcenter=0, vmax=vmax)
for y in range(2):
	ssp=ssps[y]
	print(ssp)
	fig=plt.figure(constrained_layout=True,figsize=(20,20))
	widths=[7,7]
	heights=[2,2,2,2,2]
	spec=fig.add_gridspec(ncols=2,nrows=5,width_ratios=widths,height_ratios=heights)
	vmin=-0.5
	vmax=0.5
	divnorm = mcolors.DivergingNorm(vmin=vmin, vcenter=0, vmax=vmax)
	for z in range(len(model_is)):
		model=models[model_is[z]]
		print(model)
		#load data
		folder='decades/' + ssp + '/'

		T_var_decades=np.load(folder + 'T_' + seas + 'std_' + ssp + '_' + model + '_decades.npy')

		T_var_decades=np.swapaxes(T_var_decades,0,1)

		#create map
		lon_axis=np.linspace(-179.5,179.5,360)
		lat_axis=np.linspace(-89.5,89.5,180)
		lat_axis=np.flip(lat_axis)
		xx, yy = np.meshgrid(lon_axis,lat_axis)
		map0=Basemap(projection='cyl')

		
		ax=fig.add_subplot(spec[int(z/2),z%2])
		#map0.contourf(xx,yy,LFP_decades[:,:,i+1]-LFP_decades[:,:,0],15,cmap=my_cmap,norm=divnorm,ax=ax)
		map0.pcolormesh(xx,yy,T_var_decades[:,:,-1]-T_var_decades[:,:,0],cmap=my_cmap,norm=divnorm,ax=ax)
		map0.drawcoastlines(ax=ax)
		title=model + ': ' + decaden[-1] + ' from ' + decaden[0]
		ax.set_title(title)

	axins=inset_axes(ax,width='100%',height='5%',loc='lower center',bbox_to_anchor=(-0.52,-0.1,1,1),bbox_transform=ax.transAxes,borderpad=0,)
	sm = plt.cm.ScalarMappable(cmap=my_cmap, norm=divnorm)
	cbar=fig.colorbar(sm,cax=axins,orientation='horizontal')
	cbar.set_ticks([vmin,vmin/2,0,vmax/2,vmax])
	labels=[str(vmin) + degree_sign + 'C',str(vmin/2) + degree_sign + 'C','0' + degree_sign + 'C',str(vmax/2) + degree_sign + 'C',str(vmax) + degree_sign + 'C']
	cbar.set_ticklabels(labels)
	plt.savefig('decade_plots/' + ssp + '/' + seas + 'T_var_decades_' + ssp + '_all.png',bbox_inches='tight')
	plt.close()

#model meridional temperature gradient plots

for y in range(2):
	ssp=ssps[y]
	print(ssp)
	for z in range(len(model_is)):
		model=models[model_is[z]]
		print(model)
		#load data
		folder='decades/' + ssp + '/'

		T_grad_decades=np.load(folder + 'T_grad_mean_' + ssp + '_' + model + '_decades.npy')

		T_grad_decades=np.swapaxes(T_grad_decades,0,1)

		#create map
		lon_axis=np.linspace(-179.5,179.5,360)
		lat_axis=np.linspace(-89,89,179)
		lat_axis=np.flip(lat_axis)
		xx, yy = np.meshgrid(lon_axis,lat_axis)
		map0=Basemap(projection='cyl')

		#such that the gradient is expressed as that going away from the equator 
		T_grad_decades[0:90,:,:]*=-1

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
			map0.pcolormesh(xx,yy,T_grad_decades[:,:,i+1]-T_grad_decades[:,:,0],cmap=my_cmap,norm=divnorm,ax=ax)
			map0.drawcoastlines(ax=ax)
			title=decaden[i+1] + ' from ' + decaden[0]
			ax.set_title(title)

		axins=inset_axes(ax,width='100%',height='5%',loc='lower center',bbox_to_anchor=(-0.52,-0.1,1,1),bbox_transform=ax.transAxes,borderpad=0,)
		sm = plt.cm.ScalarMappable(cmap=my_cmap, norm=divnorm)
		cbar=fig.colorbar(sm,cax=axins,orientation='horizontal')
		cbar.set_ticks([vmin,vmin/2,0,vmax/2,vmax])
		labels=[str(vmin) + degree_sign + 'C',str(vmin/2) + degree_sign + 'C','0' + degree_sign + 'C',str(vmax/2) + degree_sign + 'C',str(vmax) + degree_sign + 'C']
		cbar.set_ticklabels(labels)
		plt.savefig('decade_plots/' + ssp + '/merid_grad/' + 'T_var_decades_' + ssp + '_' + model + '.png',bbox_inches='tight')
		plt.close()



