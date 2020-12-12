# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 12:03:39 2020

@author: 00316584
"""
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import datetime
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import rasterio

import os
from osgeo import gdal, gdal_array


#https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
#https://jakevdp.github.io/PythonDataScienceHandbook/05.11-k-means.html

def clusterPrepare(file, band = 1, set_neg_nan = False):
    img_ds = gdal.Open(file, gdal.GA_ReadOnly)
    band_array = img_ds.GetRasterBand(band)
    img = band_array.ReadAsArray()
    if set_neg_nan:
        img[img<=0] = np.nan 
    X = img.reshape((-1,1))
    
    #No nan values position:
    xPos     = np.argwhere(~np.isnan(X))
    rows, cols = zip(*xPos)
    
    #No-nan values:
    X = X[rows,cols]
    
    return X

           
# k-MEANS cluster process:
   
# Images directory:
stdno = 'Corrigida\\STDNorm'

#Number of clusters:
n_cluster = 4
stdnorm = glob.glob(os.path.join(basePath,stdir,stdno,'*tif'))
st =  output_name
for st in stdnorm:
    
    aux_ds = gdal.Open(st, gdal.GA_ReadOnly)
    aux = aux_ds.GetRasterBand(1).ReadAsArray()
    n_images = 1

    Pos = np.argwhere(~np.isnan(aux))
    
    X = aux.reshape((-1,1))
    xPos     = np.argwhere(~np.isnan(X)) 
    rows, cols = zip(*xPos)
    X = X[rows,cols]
    X = X.reshape((-1,1))
    
    array = np.zeros((X.shape[0],1, n_images),
             gdal_array.GDALTypeCodeToNumericTypeCode(aux_ds.GetRasterBand(1).DataType))
    data = [st]
    for i in range(0,len(data)):
        array[:,0,i] = clusterPrepare(data[i], band = 1, set_neg_nan = False)
    array = array.reshape((len(X),n_images))
    
    print('Data prepared, initializing clustering process.\n')
    k_means = KMeans(n_clusters=n_cluster, init ='k-means++')
    k_means.fit(array)
    
    X_cluster = k_means.labels_
    
    rows, cols = zip(*Pos)
    ClustedArray = np.empty((aux.shape[0], aux.shape[1],))
    ClustedArray[:] = np.nan
    ClustedArray[rows,cols] = X_cluster
    
    #X_cluster = X_cluster.reshape(img.shape)
    fig, ax0 = plt.subplots(figsize=(8,10))
    
    plt1 = ax0.imshow(ClustedArray, cmap="tab20b")
    
    out_clu = 'Cluster_'+st.split('\\')[-1]
    out_clu = os.path.join(basePath,stdir,'Corrigida','Cluster',out_clu) 
    with rasterio.open(out_clu , "w", **out_meta) as dest:
        dest.write(ClustedArray,1)
        print('End Cluster {}\n'.format(out_clu.split('\\')[-1].replace('.tif', '')))

