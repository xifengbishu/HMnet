#coding:utf-8
# Force matplotlib to not use any Xwindows backend.
import matplotlib
matplotlib.use('Agg')
""" This script demonstrates the use of a convolutional LSTM network.
This network is used to predict the next frame of an artificially
generated movie which contains moving squares.
"""
import numpy as np
import numpy.ma as ma
import pylab as plt
import matplotlib.pyplot as plt  
import matplotlib.cm as cm
import collections
from collections import Counter
from matplotlib.ticker import MultipleLocator

import netCDF4 as nc
from netCDF4 import num2date, date2num, date2index, Dataset  # http://code.google.com/p/netcdf4-python/
#from PIL import Image
import csv
from PIL import Image



from six.moves import range
import random
import seaborn as sns
import pandas as pd
from pandas import datetime
import math, time
import itertools
import time
import datetime
from datetime import datetime, timedelta
from math import sqrt
import os

from sklearn import preprocessing
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import explained_variance_score,mean_absolute_error,mean_squared_error,r2_score
from sklearn.metrics import mean_squared_error

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.models import load_model
from keras.optimizers import SGD, Adadelta, Adagrad, RMSprop
from keras.layers.advanced_activations import LeakyReLU,PReLU,ELU,ThresholdedReLU
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras import optimizers
from keras.utils import plot_model
from keras.utils import multi_gpu_model   #导入keras多GPU函数
import keras
import h5py
from keras import backend as K
from keras import regularizers
from keras.callbacks import EarlyStopping

from keras import Model
from keras.layers import Input, Conv2D
from keras.layers import add, Lambda

from keras.models import *
from keras.layers import *
from keras.optimizers import *

#from data_test import load_data_test
#from wgs_load import wgs_load_data,plot_xy_result,load_data,build_model,model_score,plot_result,percentage_difference,denormalize,quick_measure
#from wgs_load import wgs_load_data,plot_xy_result,load_data,build_model,model_score,plot_result,percentage_difference,denormalize,quick_measure,wgs_load_ydata

import sys
from WGS import get_WGS
from ATT_Unet3D_ConvLSTM import get_ATT_unet3D 
from ATT_3D_RAM_ConvLSTM import get_3D_ATT_ConvLSTMWGSN 
from ops import RAM, upsampler

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
in_sys  = sys.argv[1]
out_sys = sys.argv[2]
var_num = sys.argv[3]
ker_num = sys.argv[4]
wind_num= sys.argv[5]


in_window  = int(in_sys)
out_window = int(out_sys)
channels   = int(var_num)
ker        = int(ker_num)
add_wind   = int(wind_num)

nc_f = '../../ALL_DT_China_allsat_phy_l4_1993_2019.nc'  # Your filename
nc_fid = Dataset(nc_f, 'r')  # Dataset is the class behavior to open the file
                             # and create an instance of the ncCDF4 class


Anum = 100
#batch_size=in_window
batch_size=30
epochs = 2000


'''
Blon=0
Elon=200
Blat=0
Elat=200

Blon=60
Elon=90
Blat=70
Elat=100

'''
Blon=20
Blat=20
Elon=100
Elat=100

WIDTH = Elat-Blat
HEIGHT = Elon-Blon
def_val=0.0
ReduceLR_patience=30

ori_data  = nc.Dataset('../../ALL_DT_China_allsat_phy_l4_1993_2019.nc')     # 读取nc文件
sst_data  = nc.Dataset('../../SCS_OISST_daymean_anomaly_1993-2019.nc')     # 读取nc文件
curl_data = nc.Dataset('../../SCS_CCMP_CURL_daymean_climate_1993-2019.nc')     # 读取nc文件
wind_data = nc.Dataset('../../SCS_CCMP_daymean_anomaly_1993-2019.nc')     # 读取nc文件

def ncdump(nc_fid, verb=True):
    '''
    ncdump outputs dimensions, variables and their attribute information.
    The information is similar to that of NCAR's ncdump utility.
    ncdump requires a valid instance of Dataset.

    Parameters
    ----------
    nc_fid : netCDF4.Dataset
        A netCDF4 dateset object
    verb : Boolean
        whether or not nc_attrs, nc_dims, and nc_vars are printed

    Returns
    -------
    nc_attrs : list
        A Python list of the NetCDF file global attributes
    nc_dims : list
        A Python list of the NetCDF file dimensions
    nc_vars : list
        A Python list of the NetCDF file variables
    '''
    def print_ncattr(key):
        """
        Prints the NetCDF file attributes for a given key

        Parameters
        ----------
        key : unicode
            a valid netCDF4.Dataset.variables key
        """
        try:
            print "\t\ttype:", repr(nc_fid.variables[key].dtype)
            for ncattr in nc_fid.variables[key].ncattrs():
                print '\t\t%s:' % ncattr,\
                      repr(nc_fid.variables[key].getncattr(ncattr))
        except KeyError:
            print "\t\tWARNING: %s does not contain variable attributes" % key

    # NetCDF global attributes
    nc_attrs = nc_fid.ncattrs()
    if verb:
        print "NetCDF Global Attributes:"
        for nc_attr in nc_attrs:
            print '\t%s:' % nc_attr, repr(nc_fid.getncattr(nc_attr))
    nc_dims = [dim for dim in nc_fid.dimensions]  # list of nc dimensions
    # Dimension shape information.
    if verb:
        print "NetCDF dimension information:"
        for dim in nc_dims:
            print "\tName:", dim 
            print "\t\tsize:", len(nc_fid.dimensions[dim])
            print_ncattr(dim)
    # Variable information.
    nc_vars = [var for var in nc_fid.variables]  # list of nc variables
    if verb:
        print "NetCDF variable information:"
        for var in nc_vars:
            if var not in nc_dims:
                print '\tName:', var
                print "\t\tdimensions:", nc_fid.variables[var].dimensions
                print "\t\tsize:", nc_fid.variables[var].size
                print_ncattr(var)
    return nc_attrs, nc_dims, nc_vars

# transform series into train and test sets for supervised learning
# scale train and test data to [-1, 1]
def scale(train, test):
	# fit scaler
	scaler = MinMaxScaler(feature_range=(-1, 1))
	scaler = scaler.fit(train)
	print "=== scale shape ==="
	print (train.shape)
	# transform train
	#train = train.reshape(train.shape[0], train.shape[1])
	train_scaled = scaler.transform(train)
	# transform test
	#test = test.reshape(test.shape[0], test.shape[1])
	test_scaled = scaler.transform(test)
	return scaler, train_scaled, test_scaled

def prepare_data(series, n_test, n_lag, n_seq):
	# extract raw values
	#series = series.values
	# transform data to be stationary
	diff_series = difference(series, 1)
	diff_values = diff_series.values
	diff_values = diff_values.reshape(len(diff_values), 1)
	# rescale values to -1, 1
	scaler = MinMaxScaler(feature_range=(-1, 1))
	scaled_values = scaler.fit_transform(diff_values)
	scaled_values = scaled_values.reshape(len(scaled_values), 1)
	# transform into supervised learning problem X, y
	#supervised = series_to_supervised(series, n_lag, n_seq)
	supervised = series_to_supervised(scaled_values, n_lag, n_seq)
	supervised_values = supervised.values
	# split into train and test sets
	train, test = supervised_values[0:-n_test], supervised_values[-n_test:]
	return scaler, train, test

def wgs_load_ydata(x_data, seq_len):
    amount_of_features = x_data.shape[1]
    #print ("Amount of features = {}".format(amount_of_features))
    data = x_data
    #era  = x_data[:,21]
    #data = stock.as_matrix()
    sequence_length = seq_len # index starting from 0
    x_result = []

    for index in range(len(data) - sequence_length): # maxmimum date = lastest date - sequence length
        x_result.append(data[index: index + sequence_length]) # index : index + 22days

    X_result = np.array(x_result)
    #row = round(x_result.shape[0]-seq_len*3) # 80% split

    #X_result = x_result[:int(row), :] # 90% date
    #X_result = np.reshape(X_result, (X_result.shape[0], amount_of_features,X_result.shape[1],X_result.shape[2]))
    X_result = np.reshape(X_result, (X_result.shape[0], X_result.shape[1], amount_of_features,X_result.shape[2],1))

    return X_result

def wgs_load_data(x_data, seq_len):
    amount_of_features = x_data.shape[1]
    #print ("Amount of features = {}".format(amount_of_features))
    data = x_data
    #era  = x_data[:,21]
    #data = stock.as_matrix()
    sequence_length = seq_len # index starting from 0
    x_result = []

    for index in range(len(data) - sequence_length): # maxmimum date = lastest date - sequence length
        x_result.append(data[index: index + sequence_length]) # index : index + 22days

    X_result = np.array(x_result)
    #row = round(x_result.shape[0]-seq_len*3) # 80% split

    #X_result = x_result[:int(row), :] # 90% date
    #X_result = np.reshape(X_result, (X_result.shape[0], amount_of_features,X_result.shape[1],X_result.shape[2]))

    # --------------------------------------------------------------------------------------------------------- #
    # ----- if data_format='channels_last' 5D tensor with shape:  (samples, time, rows, cols, channels) ------- #
    # --------------------------------------------------------------------------------------------------------- #
    #X_result = np.reshape(X_result, (X_result.shape[0], X_result.shape[1], amount_of_features,X_result.shape[2],channels))
    #X_result = np.reshape(X_result, (X_result.shape[0],amount_of_features,X_result.shape[1],X_result.shape[2],channels))
    #X_result = np.reshape(X_result, (X_result.shape[0], X_result.shape[1], amount_of_features,X_result.shape[3],channels))

    return X_result

'''
输入的data的shape=(627,652)
'''
def write_to_nc_4dvar(data,file_name_path,lon,lat,time,lev,nc_fid):
    import netCDF4 as nc
    import time
    import numpy as np

    nc_attrs, nc_dims, nc_vars = ncdump(nc_fid)

    nlon=len(lon)
    nlat=len(lat)
    da=nc.Dataset(file_name_path,'w',format='NETCDF4_CLASSIC')


    da.createDimension('level', lev)
    da.createDimension('time', None)
    da.createDimension('lon',nlon)  #创建坐标点
    da.createDimension('lat',nlat)  #创建坐标点


    #da.createVariable("lon",'f',("lons"))  #添加coordinates  'f'为数据类型，不可或缺
    #da.createVariable("lat",'f',("lats"))  #添加coordinates  'f'为数据类型，不可或缺

    # Create coordinate variables for 4-dimensions
    #times = da.createVariable('time', np.int32, ('time',)) 
    levels =da.createVariable('level', np.int32, ('level',)) 
    latitudes  = da.createVariable('latitude', np.float32,('lat',))
    longitudes = da.createVariable('longitude', np.float32,('lon',))
    times      = da.createVariable('time', nc_fid.variables['time'].dtype,('time',))
    # You can do this step yourself but someone else did the work for us.
    for ncattr in nc_fid.variables['time'].ncattrs():
         times.setncattr(ncattr, nc_fid.variables['time'].getncattr(ncattr))



    # Global Attributes
    da.description = 'bogus example script' 
    da.history = 'Created ' + time.ctime(time.time()) 
    da.source = 'netCDF4 python module tutorial'
    # Variable Attributes 
    latitudes.units = 'degree_north'
    longitudes.units = 'degree_east'
    levels.units = 'step'
    times.units = 'hours since 0001-01-01 00:00:00'
    times.calendar = 'gregorian'
    # Create the actual 4-d variable
    da.createVariable('u', np.float32,('time','level','lat','lon'))
    #da.createVariable('u', np.float32,('time','lat','lon'))


    lev_list = np.linspace(1, lev, lev)
    da.variables['latitude'][:]=lat     #填充数据
    da.variables['longitude'][:]=lon     #填充数据
    #da.variables['time'][:]=time     #填充数据
    # Assign the dimension data to the new NetCDF file.
    da.variables['time'][:] = 9999
    da.variables['level'][:]=lev_list     #填充数据
    
    #da.createVariable('u','f8',('lats','lons')) #创建变量，shape=(627,652)  'f'为数据类型，不可或缺
    da.variables['u'][:]=data       #填充数据
    da.close()


#nc_attrs, nc_dims, nc_vars = ncdump(nc_fid)

ori_dimensions, ori_variables   = ori_data.dimensions, ori_data.variables    # 获取文件中的维度和变量
sst_dimensions, sst_variables   = sst_data.dimensions, sst_data.variables    # 获取文件中的维度和变量
wind_dimensions, wind_variables= wind_data.dimensions, wind_data.variables    # 获取文件中的维度和变量
curl_dimensions, curl_variables= curl_data.dimensions, curl_data.variables    # 获取文件中的维度和变量
time_ori = ori_variables['time']
#time_ori = wwtime_ori[:9600]
sla_ori = ori_variables['sla']
lat_ori=ori_variables['latitude']
lon_ori=ori_variables['longitude']

tu_ori   = curl_variables['tu']
tv_ori   = curl_variables['tv']
curl_ori  = curl_variables['t_curl']

uu_ori   = wind_variables['uua']
vv_ori   = wind_variables['vva']
spd_ori  = wind_variables['spda']
#lat_wind=wind_variables['latitude']
#lon_wind=wind_variables['longitude']

asst_ori  = sst_variables['ssta']

dates = num2date(time_ori[:], time_ori.units)
# --- Get index associated with a specified date, extract forecast data for that date. 
date_beg = datetime(1993, 1, 1, 00, 00)
Tbeg = date2index(date_beg,time_ori,select='nearest')

date_test = datetime(2016, 1, 1, 00, 00)
Ttest = date2index(date_test,time_ori,select='nearest')
date = datetime(2015, 1, 1, 00, 00)
Tval = date2index(date,time_ori,select='nearest')
SLA_ORI  =  sla_ori[:9600,Blat:Elat,Blon:Elon]
aSST_ORI = asst_ori[:9600,Blat:Elat,Blon:Elon]
uu_ORI   =   uu_ori[:9600,Blat:Elat,Blon:Elon]
vv_ORI   =   vv_ori[:9600,Blat:Elat,Blon:Elon]
spd_ORI  =  spd_ori[:9600,Blat:Elat,Blon:Elon]

tu_ORI   =   tu_ori[:9600,Blat:Elat,Blon:Elon]
tv_ORI   =   tv_ori[:9600,Blat:Elat,Blon:Elon]
curl_ORI  =  curl_ori[:9600,Blat:Elat,Blon:Elon]

def_sla = SLA_ORI[0,:,:]
def_sla = def_sla.filled(-999)


SLA_max  = np.max(SLA_ORI[:Ttest])
SLA_min  = np.min(SLA_ORI[:Ttest])
uu_max   = np.max(uu_ORI[:Ttest])
uu_min   = np.min(uu_ORI[:Ttest])
vv_max   = np.max(vv_ORI[:Ttest])
vv_min   = np.min(vv_ORI[:Ttest])
spd_max  = np.max(spd_ORI[:Ttest])
spd_min  = np.min(spd_ORI[:Ttest])
aSST_max = np.max(aSST_ORI[:Ttest])
aSST_min = np.min(aSST_ORI[:Ttest])
tu_max   = np.max(tu_ORI[:Ttest])
tu_min   = np.min(tu_ORI[:Ttest])
tv_max   = np.max(tv_ORI[:Ttest])
tv_min   = np.min(tv_ORI[:Ttest])
curl_max  = np.max(curl_ORI[:Ttest])
curl_min  = np.min(curl_ORI[:Ttest])

SLA_max = -SLA_min

tu_max = -tu_min
curl_max = -curl_min

lon = lon_ori[Blon:Elon]
lat = lat_ori[Blat:Elat]

nlon=len(lon)
nlat=len(lat)


cmap = plt.cm.coolwarm
#CS=plt.contourf(SLA_ORI[0,:,:],levels=SLA_ORI.levels[::2],cmap=cmap)  
#CS1=plt.contourf(CS,levels=CS.levels[::2])  
#xlist = np.linspace(100, 130, 120)
#ylist = np.linspace(0, 30, 120)

xlist = np.linspace(Blon, Elon, Elon-Blon)
ylist = np.linspace(Blat, Elat, Elat-Blat)
X, Y = np.meshgrid(xlist, ylist)
plt.figure()
levels = [-0.2,0.0, 0.2]
#levels = [-0.5, -0.4,-0.3,-0.2,-0.1,0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
#contour = plt.contour(X, Y, SLA_ORI[0,:,:], levels, colors='k')
contour = plt.contour(X, Y, SLA_ORI[2200,:,:], colors='k')
plt.clabel(contour, colors = 'k', fmt = '%2.1f', fontsize=5)
contour_filled = plt.contourf(X, Y, SLA_ORI[2200,:,:],cmap=cmap)
plt.colorbar(contour_filled)
plt.title('AVISO SLA'+str(Ttest)+'')
plt.xlabel('lon')
plt.ylabel('lat')
plt.savefig('ori_SLA.jpg',dpi=300)
plt.close()
'''

fig, ax = plt.subplots()
q = ax.quiver(X, Y, uu_ORI[1,:,:],vv_ORI[1,:,:], units='width')
qk = ax.quiverkey(q, 0.9, 0.9, 2, r'$2 \frac{m}{s}$', labelpos='E',coordinates='figure')
plt.title('Arrows scale with plot width, not view')
plt.xlabel('lon')
plt.ylabel('lat')
plt.savefig('ori_WIND.jpg',dpi=300)
plt.close()

contour = plt.contour(X, Y, uu_ORI[0,:,:], colors='k')
plt.clabel(contour, colors = 'k', fmt = '%2.1f', fontsize=5)
contour_filled = plt.contourf(X, Y, uu_ORI[0,:,:],cmap=cmap)
plt.colorbar(contour_filled)
plt.savefig('0ori_UU.jpg',dpi=300)
plt.close()

contour = plt.contour(X, Y, SLA_ORI[0,:,:], colors='k')
plt.clabel(contour, colors = 'k', fmt = '%2.1f', fontsize=5)
contour_filled = plt.contourf(X, Y, SLA_ORI[0,:,:],cmap=cmap)
plt.colorbar(contour_filled)
plt.savefig('0ori_SLA.jpg',dpi=300)
plt.close()

contour = plt.contour(X, Y, aSST_ORI[0,:,:], colors='k')
plt.clabel(contour, colors = 'k', fmt = '%2.1f', fontsize=5)
contour_filled = plt.contourf(X, Y, aSST_ORI[0,:,:],cmap=cmap)
plt.colorbar(contour_filled)
plt.savefig('0ori_aSST.jpg',dpi=300)
plt.close()

for i in range(4):
	ww=i+1
	plt.subplot(2,2,ww)
	contour_filled = plt.contourf(X, Y, SLA_ORI[ww,:,:],cmap=cmap)
	#plt.colorbar(contour_filled,shrink=0.5)

plt.savefig('2ori_SLA.jpg',dpi=300)
plt.close()

for i in range(4):
	ww=i+1
	plt.subplot(2,2,ww)
	contour_filled = plt.contourf(X, Y, uu_ORI[ww,:,:],cmap=cmap)
	#plt.colorbar(contour_filled,shrink=0.5)

plt.savefig('2ori_UU.jpg',dpi=300)
plt.close()

for i in range(4):
	ww=i+1
	plt.subplot(2,2,ww)
        q = plt.quiver(X, Y, uu_ORI[ww,:,:],vv_ORI[ww,:,:])
	#plt.colorbar(contour_filled,shrink=0.5)

plt.savefig('2ori_WIND.jpg',dpi=300)
plt.close()

'''
# mark all missing values
#SLA.replace('--', nan, inplace=True)
SLA_ORI=SLA_ORI.filled(def_val)
aSST_ORI=aSST_ORI.filled(def_val)
uu_ORI  = uu_ORI.filled(def_val)
vv_ORI  = vv_ORI.filled(def_val)
spd_ORI  = spd_ORI.filled(def_val)

tu_ORI  = tu_ORI.filled(def_val)
tv_ORI  = tv_ORI.filled(def_val)
curl_ORI  = curl_ORI.filled(def_val)
#SLA_ORI=SLA_ORI.float()
 
# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------

Anum = Ttest
#Anum = 1825
'''
SLA_max=1
aSST_max=1
spd_max=1
uu_max=1
vv_max=1
curl_max=1
tu_max=1
tv_max=1
'''

sla =  SLA_ORI[Tbeg:Anum-in_window-out_window*2,:,:]/SLA_max
asst= aSST_ORI[Tbeg:Anum-in_window-out_window*2,:,:]/aSST_max
spd =  spd_ORI[Tbeg:Anum-in_window-out_window*2,:,:]/spd_max
uu  =   uu_ORI[Tbeg:Anum-in_window-out_window*2,:,:]/uu_max
vv  =   vv_ORI[Tbeg:Anum-in_window-out_window*2,:,:]/vv_max

curl =  curl_ORI[Tbeg:Anum-in_window-out_window*2,:,:]/curl_max
tu  =   tu_ORI[Tbeg:Anum-in_window-out_window*2,:,:]/tu_max
tv  =   tv_ORI[Tbeg:Anum-in_window-out_window*2,:,:]/tv_max

sla_TT  = SLA_ORI[Ttest-in_window-out_window:,:,:]/SLA_max
asst_TT =aSST_ORI[Ttest-in_window-out_window:,:,:]/aSST_max
spd_TT  = spd_ORI[Ttest-in_window-out_window:,:,:]/spd_max
uu_TT   =  uu_ORI[Ttest-in_window-out_window:,:,:]/uu_max
vv_TT   =  vv_ORI[Ttest-in_window-out_window:,:,:]/vv_max
time_TT = time_ori[Ttest-in_window-out_window:9600-1]

curl_TT  = curl_ORI[Ttest-in_window-out_window:,:,:]/curl_max
tu_TT   =  tu_ORI[Ttest-in_window-out_window:,:,:]/tu_max
tv_TT   =  tv_ORI[Ttest-in_window-out_window:,:,:]/tv_max

'''
# transform the scale of the data
sla_train = sla.reshape(sla.shape[0],sla.shape[1]*sla.shape[2])
uu_train  =  uu.reshape(uu.shape[0],uu.shape[1]*uu.shape[2])
vv_train  =  vv.reshape(vv.shape[0],vv.shape[1]*vv.shape[2])

sla_test  = sla_TT.reshape(sla_TT.shape[0],sla_TT.shape[1]*sla_TT.shape[2])
uu_test   =  uu_TT.reshape(uu_TT.shape[0],uu_TT.shape[1]*uu_TT.shape[2])
vv_test   =  vv_TT.reshape(vv_TT.shape[0],vv_TT.shape[1]*vv_TT.shape[2])

print (" === scaling ===")
scaler, train_scaled,test_scaled = scale(sla_train, sla_test)
uscaler, uu_train_scaled, uu_test_scaled = scale(uu_train, uu_test)
vscaler, vv_train_scaled, vv_test_scaled = scale(vv_train, vv_test)

# invert scaling
#ww=inverted[0, -1]
#yhat = invert_scale(scaler, xSLA_scaled, xSLA_scaled)
print ('inverted ww.shape')
#print (inverted.shape)

sla = train_scaled.reshape(sla.shape[0],sla.shape[1],sla.shape[2])
uu = uu_train_scaled.reshape(uu.shape[0],uu.shape[1],uu.shape[2])
vv = vv_train_scaled.reshape(vv.shape[0],vv.shape[1],vv.shape[2])

sla_test = test_scaled.reshape(sla_TT.shape[0],sla_TT.shape[1],sla_TT.shape[2])
uu_test = uu_test_scaled.reshape(uu_TT.shape[0],uu_TT.shape[1],uu_TT.shape[2])
vv_test = vv_test_scaled.reshape(vv_TT.shape[0],vv_TT.shape[1],vv_TT.shape[2])

contour = plt.contour(X, Y, uu[0,:,:], colors='k')
plt.clabel(contour, colors = 'k', fmt = '%2.1f', fontsize=5)
contour_filled = plt.contourf(X, Y, uu[0,:,:],cmap=cmap)
plt.colorbar(contour_filled)
plt.savefig('0scale_UU.jpg',dpi=300)
plt.close()

contour = plt.contour(X, Y, sla[0,:,:], colors='k')
plt.clabel(contour, colors = 'k', fmt = '%2.1f', fontsize=5)
contour_filled = plt.contourf(X, Y, sla[0,:,:],cmap=cmap)
plt.colorbar(contour_filled)
plt.savefig('0scale_SLA.jpg',dpi=300)
plt.close()

contour = plt.contour(X, Y, asst[0,:,:], colors='k')
plt.clabel(contour, colors = 'k', fmt = '%2.1f', fontsize=5)
contour_filled = plt.contourf(X, Y, asst[0,:,:],cmap=cmap)
plt.colorbar(contour_filled)
plt.savefig('0scale_aSST.jpg',dpi=300)
plt.close()

contour = plt.contour(X, Y, spd[0,:,:], colors='k')
plt.clabel(contour, colors = 'k', fmt = '%2.1f', fontsize=5)
contour_filled = plt.contourf(X, Y, spd[0,:,:],cmap=cmap)
plt.colorbar(contour_filled)
plt.savefig('0scale_SPD.jpg',dpi=300)
plt.close()
'''

xSLA  =  sla[:-out_window,:,:]
xaSST = asst[:-out_window,:,:]
xSPD  =  spd[:-out_window,:,:]
xUU   =   uu[:-out_window,:,:]
xVV   =   vv[:-out_window,:,:]
xCURL  =  curl[:-out_window,:,:]
xTU   =   tu[:-out_window,:,:]
xTV   =   tv[:-out_window,:,:]

ySLA = sla[in_window:,:,:]
#yUU  =  uu[in_window:Anum+in_window,:,:]
#yVV  =  vv[in_window:Anum+in_window,:,:]

xpre_aSST = asst_TT[:-out_window,:,:]
xpre_SLA  =  sla_TT[:-out_window,:,:]
xpre_SPD  =  spd_TT[:-out_window,:,:]
xpre_UU   =   uu_TT[:-out_window,:,:]
xpre_VV   =   vv_TT[:-out_window,:,:]

xpre_CURL  =  curl_TT[:-out_window,:,:]
xpre_TU   =   tu_TT[:-out_window,:,:]
xpre_TV   =   tv_TT[:-out_window,:,:]

ypre_SLA =  sla_TT[in_window:,:,:]
time     = time_TT[in_window:]
#ypre_UU  =  uu_TT[in_window:400-60+in_window,:,:]
#ypre_VV  =  vv_TT[in_window:400-60+in_window,:,:]

x_train = np.zeros((xSLA.shape[0] , xSLA.shape[1], xSLA.shape[2], 5))
x_test = np.zeros((xpre_SLA.shape[0] , xpre_SLA.shape[1], xpre_SLA.shape[2], 5))
#y_train = np.zeros((ySLA.shape[0] , ySLA.shape[1], ySLA.shape[2], channels))
#y_test = np.zeros((ypre_SLA.shape[0] , ypre_SLA.shape[1], ypre_SLA.shape[2], channels))

x_train[:,:,:,0]=xSLA
x_train[:,:,:,1]=xaSST

x_test[:,:,:,0]=xpre_SLA
x_test[:,:,:,1]=xpre_aSST

if add_wind == 1:
   x_train[:,:,:,2]=xSPD
   x_train[:,:,:,3]=xUU
   x_train[:,:,:,4]=xVV
   x_test[:,:,:,2]=xpre_SPD
   x_test[:,:,:,3]=xpre_UU
   x_test[:,:,:,4]=xpre_VV

if add_wind == 2:
   x_train[:,:,:,2]=xCURL
   x_train[:,:,:,3]=xTU
   x_train[:,:,:,4]=xTV
   x_test[:,:,:,2]=xpre_CURL
   x_test[:,:,:,3]=xpre_TU
   x_test[:,:,:,4]=xpre_TV

#y_train[:,:,:,0]=ySLA
#y_train[:,:,:,1]=yUU
#y_train[:,:,:,2]=yVV


#y_test[:,:,:,0]=ypre_SLA
#y_test[:,:,:,1]=ypre_UU
#y_test[:,:,:,2]=ypre_VV

#plt.imshow(BASIC_SLA[0][0].reshape(WIDTH, HEIGHT))
#plt.show()
xx_train = x_train[:,:,:,:channels]
xx_test  =  x_test[:,:,:,:channels]
'''
if channels == 1:
	xx_train = xx_train.reshape(xx_train.shape[0],xx_train.shape[1],xx_train.shape[2], channels)
	xx_test  =  xx_test.reshape( xx_test.shape[0], xx_test.shape[1], xx_test.shape[2], channels)
print ("xx_train shape")
print (xx_train.shape)
print (xx_test.shape)
quit()
exit()
'''

X_train = wgs_load_data(xx_train,in_window)
X_test  = wgs_load_data(xx_test,in_window)

Y_train = wgs_load_ydata(ySLA,out_window)
Y_test  = wgs_load_ydata(ypre_SLA,out_window)

#YY_test = wgs_load_ydata(ypre_SLA,out_window*8)


dates = num2date(time[:5], time_ori.units)
print([date.strftime('%Y-%m-%d %H:%M:%S') for date in dates]) # print only first ten...
dates = num2date(time[-5:], time_ori.units)
print([date.strftime('%Y-%m-%d %H:%M:%S') for date in dates]) # print only first ten...

'''
print ("X_train shape")
print (X_train[:5,:,0,10,:])
print ("Y_train shape")
print (Y_train[:5,:,0,10,:])
print ("X_test shape")
print (X_test[:5,:,0,10,:])
print ("Y_test shape")
print (Y_test[:5,:,0,10,:])

for i in range(in_window):
	print ("X_train",X_train[i,:,10,0,0])

for i in range(in_window):
	print ("Y_train",Y_train[i,:,10,0,0])



plot_range=2
for i in range(plot_range):
	ww=i+1
	plt.subplot(4,plot_range, ww)
	CS1=plt.contourf(X_train[0,-ww,:,:,0],cmap=cmap)  
	CS1=plt.contour(X_train[0,-ww,:,:,0],colors='k')  
	plt.clabel(CS1, fontsize=3, inline=1)
	if i == 0:
		plt.ylabel('X_train')

	plt.subplot(4,plot_range, ww+plot_range)
	CS1=plt.contourf(Y_train[0,ww,:,:,0],cmap=cmap)  
	CS1=plt.contour(Y_train[0,ww,:,:,0],colors='k')  
	plt.clabel(CS1, fontsize=3, inline=1)
	if i == 0:
		plt.ylabel('Y_train')

	plt.subplot(4,plot_range, ww+plot_range*2)
	CS1=plt.contourf(X_train[20,-ww,:,:,0],cmap=cmap)  
	CS1=plt.contour(X_train[20,-ww,:,:,0],colors='k')  
	plt.clabel(CS1, fontsize=3, inline=1)
	if i == 0:
		plt.ylabel('X_train')

	plt.subplot(4,plot_range, ww+plot_range*3)
	CS1=plt.contourf(Y_train[20,ww,:,:,0],cmap=cmap)  
	CS1=plt.contour(Y_train[20,ww,:,:,0],colors='k')  
	plt.clabel(CS1, fontsize=3, inline=1)
	if i == 0:
		plt.ylabel('Y_train')


plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
cax = plt.axes([0.85, 0.1, 0.025, 0.8])
plt.colorbar(cax=cax)

#plt.savefig('XY_train.jpg',dpi=600)
plt.savefig('XY_train_scaled.jpg',dpi=600)
plt.close()

for i in range(plot_range):
	ww=i+1
	plt.subplot(4,plot_range, ww)
	CS1=plt.contourf(X_test[0,-ww,:,:,0],cmap=cmap)  
	CS1=plt.contour(X_test[0,-ww,:,:,0],colors='k')  
	plt.subplot(4,plot_range, ww)
	CS1=plt.contourf(X_train[0,-ww,:,:,0],cmap=cmap)  
	CS1=plt.contour(X_train[0,-ww,:,:,0],colors='k')  
	plt.clabel(CS1, fontsize=3, inline=1)
	if i == 0:
		plt.ylabel('X_train')

	plt.subplot(4,plot_range, ww+plot_range)
	CS1=plt.contourf(Y_train[0,ww,:,:,0],cmap=cmap)  
	CS1=plt.contour(Y_train[0,ww,:,:,0],colors='k')  
	plt.clabel(CS1, fontsize=3, inline=1)
	if i == 0:
		plt.ylabel('Y_train')

	plt.subplot(4,plot_range, ww+plot_range*2)
	CS1=plt.contourf(X_train[20,-ww,:,:,0],cmap=cmap)  
	CS1=plt.contour(X_train[20,-ww,:,:,0],colors='k')  
	plt.clabel(CS1, fontsize=3, inline=1)
	if i == 0:
		plt.ylabel('X_train')

	plt.subplot(4,plot_range, ww+plot_range*3)
	CS1=plt.contourf(Y_train[20,ww,:,:,0],cmap=cmap)  
	CS1=plt.contour(Y_train[20,ww,:,:,0],colors='k')  
	plt.clabel(CS1, fontsize=3, inline=1)
	if i == 0:
		plt.ylabel('Y_train')


plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
cax = plt.axes([0.85, 0.1, 0.025, 0.8])
plt.colorbar(cax=cax)

#plt.savefig('XY_train.jpg',dpi=600)
plt.savefig('XY_train_scaled.jpg',dpi=600)
plt.close()

for i in range(plot_range):
	ww=i+1
	plt.subplot(4,plot_range, ww)
	CS1=plt.contourf(X_test[0,-ww,:,:,0],cmap=cmap)  
	CS1=plt.contour(X_test[0,-ww,:,:,0],colors='k')  
	plt.clabel(CS1, fontsize=3, inline=1)
	if i == 0:
		plt.ylabel('X_test')

	plt.subplot(4,plot_range, ww+plot_range)
	CS1=plt.contourf(Y_test[0,ww,:,:,0],cmap=cmap)  
	CS1=plt.contour(Y_test[0,ww,:,:,0],colors='k')  
	plt.clabel(CS1, fontsize=3, inline=1)
	if i == 0:
		plt.ylabel('Y_test')

	plt.subplot(4,plot_range, ww+plot_range*2)
	CS1=plt.contourf(X_test[20,-ww,:,:,0],cmap=cmap)  
	CS1=plt.contour(X_test[20,-ww,:,:,0],colors='k')  
	plt.clabel(CS1, fontsize=3, inline=1)
	if i == 0:
		plt.ylabel('X_test')

	plt.subplot(4,plot_range, ww+plot_range*3)
	CS1=plt.contourf(Y_test[20,ww,:,:,0],cmap=cmap)  
	CS1=plt.contour(Y_test[20,ww,:,:,0],colors='k')  
	plt.clabel(CS1, fontsize=3, inline=1)
	if i == 0:
		plt.ylabel('Y_test')


plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
cax = plt.axes([0.85, 0.1, 0.025, 0.8])
plt.colorbar(cax=cax)

#plt.savefig('XY_train.jpg',dpi=600)
plt.savefig('XY_test_scaled.jpg',dpi=600)
plt.close()

'''
# We create a layer which take as input movies of shape
# (n_frames, width, height, channels) and returns a movie
# of identical shape.
'''
Input shape

if data_format='channels_first' 5D tensor with shape:  (samples, time, channels, rows, cols)
if data_format='channels_last' 5D tensor with shape:  (samples, time, rows, cols, channels)
Output shape

if return_sequences
if data_format='channels_first' 5D tensor with shape:  (samples, time, filters, output_row, output_col)
if data_format='channels_last' 5D tensor with shape:  (samples, time, output_row, output_col, filters)
else

if data_format='channels_first' 4D tensor with shape:  (samples, filters, output_row, output_col)
if data_format='channels_last' 4D tensor with shape:  (samples, output_row, output_col, filters)
where o_row and o_col depend on the shape of the filter and the padding
'''

if channels == 1:
   WID=32

if channels == 2:
   WID=64

if channels == 3:
   WID=128

if channels == 1 and in_window == 13:
   batch_size=26

if channels == 1 and in_window >= 15:
   batch_size=20

if channels == 1 and in_window > 20:
   batch_size=15

if channels == 2 and in_window >= 9:
   batch_size=20

if channels == 3 and in_window >= 7:
   batch_size=15

if channels == 3 and in_window >= 13:
   batch_size=10

if channels == 5 and in_window >= 11:
   batch_size=10

if channels == 1 and in_window == 23:
   batch_size=16

if channels >= 2 and in_window == 13:
   batch_size=9

print ("channels "+str(channels)+"  filters"+str(WID))
#WID=WIDTH

# Model
#time_start = time.time()
input_image = Input(shape=(None,HEIGHT, WIDTH, channels))
#mean = mean(input_image)
#std = std(input_image)
mean = 0
std = 1
RAM_number = 8
kernel_size = 3
x = input_image
output_image = get_3D_ATT_ConvLSTMWGSN(input_image,HEIGHT,WIDTH,channels, RAM_number) 
#output_image = get_ATT_unet3D(input_image,RAM_number) 
#output_image = get_WGS(input_image,HEIGHT,WIDTH,channels, RAM_number) 

'''
R = RAM_number   # RAM_number
k = in_window  # historical days
l = out_window  # future days

x = Lambda(lambda x: (x - mean) / std)(input_image)
x = f0 = Conv2D(k, kernel_size=3, strides=1, padding='same')(x)
for i in range(R):
    x = RAM(x, channels=k, RAM_number=R)
x = Conv2D(k, kernel_size=3, strides=1, padding='same')(x)
x = add([x, f0])
x = upsampler(x, scale=1)
x = Conv2D(l, kernel_size=3, strides=1, padding='same')(x)
output_image = Lambda(lambda x: (x * std + mean))(x)
'''

model = Model(input_image, output_image)
plot_model(model, to_file='SCS_3D_CRAM_ConvLSTM.png',show_shapes=True,show_layer_names=True)
parallel_model = multi_gpu_model(model, gpus=2) # 设置使用2个gpu，该句放在模型compile之前
parallel_model.compile(optimizer=Adam(lr=1e-4, epsilon=1e-8), loss='logcosh', metrics=['mae'])
#model.compile(optimizer=Adam(lr=1e-4, epsilon=1e-8), loss='logcosh', metrics=['mae'])
model.summary()

callbacks_list = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=ReduceLR_patience*2, verbose=1, mode='auto'),
keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=ReduceLR_patience, verbose=1, mode='auto'),
keras.callbacks.ModelCheckpoint(filepath='nice_model_SCS_sla_var'+str(channels)+'_input'+str(in_window)+'output'+str(out_window)+'_ker'+str(ker)+'.h5',monitor='val_loss',save_best_only=True)]


history=parallel_model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, verbose=1,callbacks=callbacks_list, validation_split=0.1, shuffle=False)

# batch_size = 32
# print('steps_per_epoch:',str((train_length - k - l) // batch_size))
# History = model.fit_generator(train_generator(train_data, batch_size), steps_per_epoch=(train_length - k - l) // batch_size,epochs=1, verbose=1,max_queue_size=1)

#seq.save('Final_model_SCS_sla_var'+str(channels)+'_input'+str(in_window)+'output'+str(out_window)+'_ker'+str(ker)+'.h5')
#model.save_weights('model_weight_SLA.h5')
model.save_weights('Final_model_SCS_sla_var'+str(channels)+'_input'+str(in_window)+'output'+str(out_window)+'_ker'+str(ker)+'.h5')

#time_end = time.time()
#f = open('result.txt', 'a')
#f.write('N = ' + str(N) + '\nR = ' + str(R) + '\nk = ' + str(k) + '\nl = ' + str(l) + '\n')
#f.write('Train time: %.2fs'% (time_end - time_start))
#f.close()
#print('Model training has completed！Cost %.2fs.' % (time_end - time_start))


'''
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='test_loss')
plt.legend()
plt.savefig('history_sla_loss_var'+str(channels)+'_input'+str(in_window)+'output'+str(out_window)+'_ker'+str(ker)+'.jpg',dpi=600)
plt.close()

plt.plot(history.history['acc'], label='train_acc')
plt.plot(history.history['val_acc'], label='test_acc')
plt.legend()
plt.savefig('history_sla_acc_var'+str(channels)+'_input'+str(in_window)+'output'+str(out_window)+'_ker'+str(ker)+'.jpg',dpi=600)
plt.close()
'''
#new_pos = seq.predict(BASIC_SLA[:30,:,:,:,:])
print (" === predict ===")

#model.load_weights('model_weight_SLA.h5')
model.load_weights('Final_model_SCS_sla_var'+str(channels)+'_input'+str(in_window)+'output'+str(out_window)+'_ker'+str(ker)+'.h5')
#model = load_model('nice_model_SCS_sla_var'+str(channels)+'_input'+str(in_window)+'output'+str(out_window)+'_ker'+str(ker)+'.h5')
sla_predict = model.predict(X_test)

#seq = load_model('nice_model_SCS_sla_var'+str(channels)+'_input'+str(in_window)+'output'+str(out_window)+'_ker'+str(ker)+'.h5')
#seq = load_model('Final_model_sla_var'+str(channels)+'_input'+str(in_window)+'output'+str(out_window)+'_ker'+str(ker)+'.h5')

Tdef_val=np.min(Y_test)
Pdef_val=np.min(sla_predict)

sla_predict=sla_predict*SLA_max
Y_test=Y_test*SLA_max


'''
Tdef_val=Y_test[0,0,2,10,0]
Y_test[Y_test==Tdef_val]=np.nan

#Y_test = scaler.inverse_transform(Y_test)
for i in range(def_sla.shape[0]):
	for j in range(def_sla.shape[1]):
    		if def_sla[i,j] == -999:
			sla_predict[:,:,i,j,:]=np.nan
PS2 = plt.contour(X, Y, sla_predict[0,1,:,:,0], levels,colors='k')
plt.clabel(PS2, colors = 'k', fmt = '%2.1f', fontsize=5)
plt.colorbar(PS22)
plt.subplot(2,2,3)
PS2= plt.contourf(X, Y, sla_predict[2,0,:,:,0],cmap=cmap)
PS2 = plt.contour(X, Y, sla_predict[2,0,:,:,0], levels,colors='k')
plt.clabel(PS2, colors = 'k', fmt = '%2.1f', fontsize=5)
plt.colorbar(PS22)
plt.subplot(2,2,4)
PS2= plt.contourf(X, Y, sla_predict[2,1,:,:,0],cmap=cmap)
PS2 = plt.contour(X, Y, sla_predict[2,1,:,:,0], levels,colors='k')
plt.clabel(PS2, colors = 'k', fmt = '%2.1f', fontsize=5)
plt.colorbar(PS22)


plt.savefig('SLA_predict_var'+str(channels)+'_input'+str(in_window)+'output'+str(out_window)+'_ker'+str(ker)+'.jpg',dpi=300)
plt.close()
'''

nc_def = -2.147484e+09
Y_test[Y_test==np.nan]=nc_def
sla_predict[sla_predict==np.nan]=nc_def

write_to_nc_4dvar(Y_test,'./scs_SLA_Y_test_var'+str(channels)+'_input'+str(in_window)+'output'+str(out_window)+'_ker'+str(ker)+'.nc',lon,lat,time,out_window,nc_fid)
write_to_nc_4dvar(sla_predict,'./scs_SLA_predict_var'+str(channels)+'_input'+str(in_window)+'output'+str(out_window)+'_ker'+str(ker)+'_nice.nc',lon,lat,time,out_window,nc_fid)

exit()

'''
# And then compare the predictions
# to the ground truth
track2 = noisy_movies[which][::, ::, ::, ::]

for i in range(15):
    fig = plt.figure(figsize=(10, 5))

    ax = fig.add_subplot(121)

    if i >= 7:
        ax.text(1, 3, 'Predictions !', fontsize=20, color='w')
    else:
        ax.text(1, 3, 'Initial trajectory', fontsize=20)

    toplot = track[i, ::, ::, 0]
    plt.imshow(toplot)

    ax = fig.add_subplot(122)
    plt.text(1, 3, 'Ground truth', fontsize=20)
    toplot = track2[i, ::, ::, 0]
    if i >= 2:
        toplot = shifted_movies[which][i - 1, ::, ::, 0]

    plt.imshow(toplot)

    plt.savefig('%i_animate.png' % (i + 1))
'''
