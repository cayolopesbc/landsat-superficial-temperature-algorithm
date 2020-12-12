# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 11:52:52 2020

@author: 00316584
"""

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from yellowbrick.cluster import KElbowVisualizer
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

def silhoutte(X, range_n_clusters, label = None, var = None):
    n   = []
    avg = []
    for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
        fig, (ax1,ax2) = plt.subplots(1,1)
        # fig, ax1 = plt.subplots(1,1)
        
        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
    
        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters=n_clusters, init ='k-means++', random_state=10)
        cluster_labels = clusterer.fit_predict(X)
    
        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X, cluster_labels)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)
        n.append(n_clusters)
        avg.append(silhouette_avg)
    
        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)
    
        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]
    
            ith_cluster_silhouette_values.sort()
    
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
    
            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)
    
            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    
            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples
    
        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")
    
        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    
        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
        
        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')
    
        # Labeling the clusters
        centers = clusterer.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                    c="white", alpha=1, s=200, edgecolor='k')
    
        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                        s=50, edgecolor='k')
    
        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")
    
        plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                      "with n_clusters = %d" % n_clusters),
                     fontsize=14, fontweight='bold')        
        
    
    plt.show()
    
    clusterer = KMeans(n_clusters = 2, random_state=10)
    cluster_labels = clusterer.fit_predict(X)
    res = pd.DataFrame({'n_clusters':n,'avg_sil_score':avg, 'label':label, 'var':var})
    print(res)
    return res



def ElbowMethod(model, X, k=(2,10), metric = 'distortion'):
    '''
    yellowBricks API: https://www.scikit-yb.org/en/latest/index.html
    
    The elbow method is used to determine the optimal number of clusters in k-means clustering. 
    The elbow method plots the value of the cost function produced by different values of k. 
    As you know, if k increases, average distortion will decrease, each cluster will have fewer 
    constituent instances, and the instances will be closer to their respective centroids. 
    However, the improvements in average distortion will decline as k increases. T
    he value of k at which improvement in distortion declines the most is called the elbow, 
    at which we should stop dividing the data into further clusters.

    By default, the scoring parameter metric is set to distortion, which computes the sum of 
    squared distances from each point to its assigned center. However, two other metrics can 
    also be used with the KElbowVisualizer – silhouette and calinski_harabasz. , while the calinski_harabasz 
    score computes the ratio of dispersion between and within clusters.

    Metrics:
        - distortion (default): computes the sum of squared distances from each point 
                                to its assigned center. 
        - silhouette: calculates the mean Silhouette Coefficient of all samples.
        - calinski_harabasz: computes the ratio of dispersion between and within clusters.
        
    Parameters:
     - k :  The k values to compute scores for. If a single integer is specified, 
    then will compute the range (2,k). If a tuple of 2 integers is specified, then k will be in 
    np.arange(k[0], k[1]). 


    Returns
    -------
    None.

    '''    
    visualizer = KElbowVisualizer(
        model, k=k, metric = metric)
    
    visualizer.fit(X)        # Fit the data to the visualizer
    visualizer.show() 
    # visualizer.close()
    return visualizer


################ Silhouette


#https://jakevdp.github.io/PythonDataScienceHandbook/05.11-k-means.html
basePath = r"C:\Users\00316584\Desktop\Cayo\SRHNE-2020\Temperatura"
chd = os.chdir(r"C:\Users\00316584\Desktop\Cayo\SRHNE-2020\Temperatura")
class_path = os.path.join(basePath,'Classifier')
       
files    =  glob.glob(os.path.join(basePath,'*tif'))
n_images = len(files)

lakes = ['MANGUEIRA', 'MIRIM']
variables = ['ST', 'Anomaly', 'STDNormalized','Normalized'] 

results = pd.DataFrame([], columns = ['n_clusters','avg_sil_score', 'label', 'var'])
for lake in lakes:
    lake_data  = [x for x in files if (x.split('\\')[-1].split('_')[1].upper() == lake)]
    for var in variables:
        data = [x for x in lake_data if(x.split('\\')[-1].split('_')[0].upper() == var.upper())]
        n_images = len(data)
        
        #Base data:
        aux_ds = gdal.Open(data[0], gdal.GA_ReadOnly)
        aux = aux_ds.GetRasterBand(1).ReadAsArray()
        Pos = np.argwhere(~np.isnan(aux))
        
        X = aux.reshape((-1,1))
        xPos     = np.argwhere(~np.isnan(X)) 
        rows, cols = zip(*xPos)
        X = X[rows,cols]
        X = X.reshape((-1,1))
        
        array = np.zeros((X.shape[0],1, n_images),
                     gdal_array.GDALTypeCodeToNumericTypeCode(aux_ds.GetRasterBand(1).DataType))
        
        for i in range(0,len(data)):
            array[:,0,i] = clusterPrepare(data[i], band = 1)
        
        array = array.reshape((len(X),n_images))
        print('##########SILHOUTTE ANALISYS ALL DATA###############')
        print('LAKE {}'.format(lake.upper()))
        print('########VARIABLE {}'.format(var.upper()))
        results3 = pd.concat([results, silhoutte(array, range_n_clusters, label = lake, var = var)], axis = 1)
        print('####################################################')
        

##### ELbow Method#####
viz_dir = r'C:\Users\00316584\Desktop\Cayo\SRHNE-2020\Temperatura\ST'
files_viz =glob.glob(os.path.join(viz_dir,'*tif'))
viz = []
i=0
for v in files_viz:
    fs = glob.glob(os.path.join(viz_dir,v)+'*.tif')
    for f in fs:
        i+=1
        print('{} - {} '.format(i,f))
        n_images = 1
        #Base data:
        data = f
        aux_ds = gdal.Open(data, gdal.GA_ReadOnly)
        aux = aux_ds.GetRasterBand(1).ReadAsArray()
        aux[aux<=0] = np.nan
        Pos = np.argwhere(~np.isnan(aux))
        
        X = aux.reshape((-1,1))
        xPos     = np.argwhere(~np.isnan(X)) 
        rows, cols = zip(*xPos)
        X = X[rows,cols]
        X = X.reshape((-1,1))
        
        array = np.zeros((X.shape[0],1, n_images),
                     gdal_array.GDALTypeCodeToNumericTypeCode(aux_ds.GetRasterBand(1).DataType))
        data = [data]
        for i in range(0,len(data)):
            array[:,0,i] = clusterPrepare(data[i], band = 1, set_neg_nan = True)
        array = array.reshape((len(X),n_images))
        model = KMeans(init ='k-means++')
        viz.append(ElbowMethod(model, array, k=(2,10), metric = 'distortion'))
        print('End file {}\n'.format(f))