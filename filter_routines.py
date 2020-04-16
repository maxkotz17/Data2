import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.io import netcdf
import xarray as xr
import matplotlib.cm as cm


with xr.open_dataset("SSHA_v1812_2010010612.nc") as data:
	SLA = data.SLA.values
	longitude=data.Longitude.values
	latitude=data.Latitude.values
	time=data.Time.values

def isfloat(value):
	try:
		float(value)
		return True
	except ValueError:
		return False

def is_data(value):
	try:
		float(value)
		return [float(value),True]
	except ValueError:
		return [666,False]

def crop(field,longitude,latitude,lonmin,lonmax,latmin,latmax):
	lon_list=np.where(np.logical_and(longitude>lonmin,longitude<lonmax))
	lat_list=np.where(np.logical_and(latitude>latmin,latitude<latmax))
	lonmin=np.min(lon_list)
	lonmax=np.max(lon_list)
	latmin=np.min(lat_list)
	latmax=np.max(lat_list)
	return [field[latmin:latmax,lonmin:lonmax],longitude[lonmin:lonmax],latitude[latmin:latmax]]

def sub_sample(field,longz,latz,sub):
	x_no=len(field[0,:])	#longitude
	y_no=len(field[:,0])	#latitude
	new_x_no=int(np.floor(x_no/sub)+1)
	new_y_no=int(np.floor(y_no/sub)+1)
	fieldn=np.zeros((new_y_no,new_x_no))
	fieldn_count=np.zeros((new_y_no,new_x_no))
	longn=np.zeros((new_x_no))
	latn=np.zeros((new_y_no))
	for i in range (x_no):	
		for j in range(y_no):
			index=(i-i%sub)/sub
			indey=(j-j%sub)/sub
			fieldn[indey,index]=fieldn[indey,index]+field[j,i]
			longn[index]=longn[index]+longz[i]
			latn[indey]=latn[indey]+latz[j]
			fieldn_count[indey,index]=fieldn_count[indey,index]+1

	fieldf=np.divide(fieldn[:,0:-1],fieldn_count[:,0:-1])
	longf=np.divide(longn[0:-1],np.sum(elevn_count[:,0:-1],0))
	latf=np.divide(latn,np.sum(fieldn_count[:,:],1))
	return [fieldf,longf,latf]


#crop bathymetry for plot
[SLA_c,long_crop,lat_crop]=crop(SLA,longitude,latitude,10,40,-75,-25)

#sub_sample the bathymetry for plot
[SLA_s,longs,lats]=sub_sample(SLA_c,long_crop,lat_crop,7)
	





