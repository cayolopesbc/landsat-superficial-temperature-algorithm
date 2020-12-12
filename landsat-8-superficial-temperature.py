# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 11:36:04 2020

@author: 00316584
"""

################################ Imports ##########################################################

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from yellowbrick.cluster import KElbowVisualizer

from osgeo import gdal
import os
import pandas as pd
import numpy as np
import rasterio
from shapely.geometry import Point, MultiPoint, mapping
from datetime import date,timedelta,time
import calendar
import datetime
import geopandas as gpd
import rioxarray as rxy
#import gdal
from gdalconst import GA_ReadOnly
import matplotlib.pyplot as plt
from scipy import ndimage, misc

###################################################################################################

############### PLOTPNG ######################################

def format_func_lon(value, tick_number):
    # find number of multiples of pi/2
    offset = -value*(bounds[2]-bounds[0])/200
    return r'{}'.format(round(bounds[2] + offset,2))

def format_func_lat(value, tick_number):
    # find number of multiples of pi/2
    offset = -value*(bounds[1]-bounds[3])/250
    return r'{}'.format(round(bounds[1] + offset,2))

def crop_center(img,cropx,cropy):
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]


def classDefinition(df, var, n_class = None, delta = None, maxminV = None):
    if not n_class:
        n_class = round(1 + np.log10(df.shape[0]))
        print(n_class)

    if not isinstance(maxminV, list):
        maxminV = [df[var].max(), df[var].min()]
        
    delta   = (maxminV[0] - maxminV[1])/n_class
    vrange = np.arange(maxminV[1],maxminV[0],delta)
    irange = [ n + delta/2 for n in vrange]
    
    df['C_'+var] = np.nan
    
    for i in range(0,len(vrange)-1):
            ind  = df.loc[(df[var]>= irange[i])&(df[var]<= irange[i+1])].index
            df['C_'+var][ind] = vrange[i+1]
       
    ind  = df['C_'+var].loc[df['C_' + var].isnull()].index
    df['C_' + var][ind] = vrange[0]
    return df,n_class

def minimaxCB(df):
    minimax = {}
    minimax['ST'] = [df[['min_Mangueira','min_Mirim']].min().min(), df[['max_Mangueira','max_Mirim']].max().max()]
    minimax['Anomaly'] = [df[['Anomaly_Mangueira_min','Anomaly_Mirim_min']].min().min(), df[['Anomaly_Mangueira_max','Anomaly_Mirim_max']].max().max()] 
    minimax['STDNormalized'] = [df[['stdnorm_Mangueira_min','stdnorm_Mirim_min']].min().min(), df[['stdnorm_Mangueira_max','stdnorm_Mirim_max']].max().max()] 
    minimax['Normalized'] = [0,1] 
    return minimax

########################## Auxiliary Functions ####################################################

def read_file(file):
    with rasterio.open(file) as src:
        return(src.read(1))

def cropRst(raster, mask, out_tif = None, remove = False, buffer_mask = 0):
    
    if out_tif == None:
        out_tif = raster.replace('.tif','_crop.tif')
    
    if isinstance(mask,str):
        try:    
            mask_shp = gpd.GeoDataFrame.from_file(mask)
            with rxy.open_rasterio(raster, masked = True, chunks = True) as ds:
                
                clipped = ds.rio.clip(mask_shp.geometry.apply(mapping), mask_shp.crs, drop=False, invert=False)
                clipped.rio.to_raster(out_tif)
            
            if remove:
                os.remove(raster)
            
            print('Crop raster by shapefile.')
            return out_tif
        
        except:
            maskDs = gdal.Open(mask, GA_ReadOnly)
            projection=maskDs.GetProjectionRef()
            geoTransform = maskDs.GetGeoTransform()
            minx = geoTransform[0]
            maxy = geoTransform[3]
            maxx = minx + geoTransform[1] * maskDs.RasterXSize
            miny = maxy + geoTransform[5] * maskDs.RasterYSize
            print('Crop raster by raster.')
            
    elif isinstance(mask,list):
        minx, maxy, maxx, miny = mask
        projection = gdal.Open(raster,GA_ReadOnly).GetProjectionRef()
        
    data = gdal.Open(raster, GA_ReadOnly)
    gdal.Translate(out_tif,data,format='GTiff',projWin=[minx,maxy,maxx,miny],outputSRS=projection)    
    del data    
    print("Crop raster by bounds.")
    return out_tif

def julian2gregorianDate(year,juliandays):
    # Based on : 
    # https://www.science-emergence.com/Articles/MODIS-Convert-Julian-Date-to-MMDDYYYY-Date-format-using-python/
    month = 1
    while juliandays - calendar.monthrange(year,month)[1] > 0 and month <= 12:
        juliandays = juliandays - calendar.monthrange(year,month)[1]
        month = month + 1
        
    return datetime.date(year, month, juliandays)

def ndwi(file):
    '''
    Landsat 8 bands:
    https://www.usgs.gov/faqs/what-are-band-designations-landsat-satellites?qt-news_science_products=0#qt-news_science_products
    NDwI
        Landsat-8: 
        Green Band = band 3
        MIR band = band 5
    
    '''
    band2 =  read_file(file + '_b3.tif')*2.0000E-05 - 0.100000
    band4 = read_file(file + '_b5.tif')*2.0000E-05 - 0.100000
    ndwi = (band2 - band4)/(band2 + band4)
    ndwi[ndwi<0] = 0
    ndwi[ndwi>0] = 1
    return ndwi    

def MNDWI(file = None, band3 = None, band6 = None):
    '''    
    Modified Normalized Difference Water Index:
            
        MNDWI eliminate the interference of the shadow well; MNDWI image has more information than
        NDWI image and other visible band; MNDWI has the highest accuracy in small water body extraction
        
    Landsat-8: 
            Green = pixel values from the green band
            SWIR = pixel values from the short-wave infrared band
    
            Green Band = band 3
            MIR band = band 6
    
    '''
    if file:
        band3 =  read_file(file + '_b3.tif')*2.0000E-05 - 0.100000
        band6 = read_file(file + '_b6.tif')*2.0000E-05 - 0.100000
        
    else:
        band3 =  read_file(band3)*2.0000E-05 - 0.100000
        band6 = read_file(band6)*2.0000E-05 - 0.100000
        
    mndwi = np.true_divide(band3-band6, band3+band6, where=(band3+band6!=0))        
    
    mndwi[mndwi<0] = np.nan
    mndwi[mndwi>0] = 1
    return mndwi

#https://gis.stackexchange.com/questions/297460/clip-raster-using-mask-other-raster-using-python-gdal
def rastersMatcher(baseRst, raster2match, output):
    '''
    Parameters
    ----------
    baseRst : string, raster path
        Base raster to get parameters to match.
    raster2match : string, raster path
        Raster to match with baseRst.
    output : raster .tif
        Raster with same geo transforms parameters.

    Returns
    -------
    output : string
        Raster matched path.

    '''
   
    maskDs = gdal.Open(baseRst, GA_ReadOnly)
    geoTransform = maskDs.GetGeoTransform()
    data = gdal.Open(raster2match, GA_ReadOnly)
    data.GetRasterBand(1).SetNoDataValue(-9999)
    
    gdal.Translate(output,
                   data,
                   format='GTiff',
                   projWin=[geoTransform[0],geoTransform[3],geoTransform[0] + geoTransform[1]*maskDs.RasterXSize, geoTransform[3] + geoTransform[5] * maskDs.RasterYSize],
                   width  = maskDs.RasterXSize,
                   height = maskDs.RasterYSize,
                   outputSRS = maskDs.GetProjectionRef(),
                   resampleAlg = 'bilinear'
                   ) 
    return output


def lowPassFilter(file, size = (18,18)):
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.uniform_filter.html
    # return ndimage.uniform_filter(rasterio.open(file).read(1), mode= 'nearest', size=size)
    return ndimage.uniform_filter(rasterio.open(file).read(1), mode= 'constant', cval = 0, size=size)

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


    
########################################################################################################


'''



Tb10:  Brigthness Temperature band 10
Tb11:  Brigthness Temperature band 11
emissB10 : Emissivity Band 10
emissB11 : Emissivity Band 11
delEmiss : Difference between emissivities in bands 10 and 11
meanEmiss: Mean emissivity from bandas 10  and 11
wp: Water vapor
ci, i = 0,..6: SW algorithm coefficients

ST: Surface Temperature based in Split-Window Algorithm from Jiménez-Muñoz et al.(2014):
    
        ST = Tb10 + c1*(Tb10 - Tb11) + c2*(Tb10 - Tb11)^2 + c0 + 
                
                        (c3+c4*wp)(1-meanEmiss) + (c5 + c6*wp)*delEmiss



'''

############################# CONSTANTS #######################################

# SW algorithm coefficients for Global Atmospheric Profiles from Reanalysis
# Information (GAPRI) Atmospheric Profile (Jiménez-Muñoz et al., 2014):
SWcoef = {
    'c0':-0.268 ,
    'c1': 1.378,
    'c2': 0.183,
    'c3': 54.30,
    'c4': -2.238,
    'c5': -129.20,
    'c6': 16.40, 
    }

# Atmospheric functions coefficients for GAPRI (Jiménez-Muñoz et al., 2014):
psiCoefs = {
    'psi1': np.array([0.04019, 0.02916, 1.01523]),
    'psi2': np.array([-0.38333, -1.50294, 0.20324]),
    'psi3': np.array([0.00918, 1.36072, -0.27514])
    }

# TIRS_THERMAL_CONSTANTS for Landsat-8:
K1_CONSTANT_BAND_10 = 774.8853
K2_CONSTANT_BAND_10 = 1321.0789
K1_CONSTANT_BAND_11 = 480.8883
K2_CONSTANT_BAND_11 = 1201.1442

# Emissivity based in MODIS Emissivity Library (Zhang, 1999):
# L-8 Bands: 10 and 11
emissB10 = 0.992
emissB11 = 0.988
delEmiss = emissB10 - emissB11
meanEmiss = 0.5*(emissB10 + emissB11)

# MOD07L2 coefficients:
# https://atmosphere-imager.gsfc.nasa.gov/sites/default/files/ModAtmo/MYD07_L2.C6.CDL.fs
WV_scale_factor = 0.001000000047497451


################################# DATA ANALYSIS ################################
# 0. Paths/Initial Process:
# 0.1. Directory with .tif files:
    #Landsat-8
L8RawData = r'C:\Users\00316584\Downloads\Landsat\L8'
    #MOD07 - Water Vapor
MOD7Dir = r'C:\Users\00316584\Downloads\Landsat\MOD07L2'

# 0.2. Directory with shapefile masks:
maskPath = r'C:\Users\00316584\Downloads\Landsat\MangueiraShp'
    #Two Lakes
shapes = {'Mangueira':'mangueira_withoutTAIM.shp', 'Mirim':'polimirim.shp' }

# 0.3. .txt file with L-8 files names without band (used to USGS bulkdownload):
files = pd.read_csv(r'C:\Users\00316584\Desktop\Cayo\SRHNE-2020\Temperatura\landsat8.txt', header = None)

# 0.4. Output directory:
basePath = r"C:\Users\00316584\Desktop\Cayo\SRHNE-2020\Temperatura04"
chd = os.chdir(basePath)

# 0.5. Datetime List from MOD07:
waterVapor = []
for file in os.listdir(MOD7Dir):
    if file.endswith('Vapor.tif'):
        date = file.split('.')[1][1:]
        year = int(date[:4])
        julidays = int(date[4:])
        date =  julian2gregorianDate(year,julidays)
        waterVapor.append((np.datetime64(date),os.path.join(MOD7Dir,file)))
waterVaporDF = pd.DataFrame(waterVapor, columns = ['Date','MOD07File'])
waterVaporDF.set_index('Date', inplace = True)

# 0.6. Create Direcories to output:
directories = ['ST', 'Anomaly', 'ClassifierPNG','Cluster','STDNormalized','Normalized','PNG','ST_MERGE'] 
for direc in directories:
    if os.path.exists(os.path.join(basePath,direc)):
        continue
    else:
        os.mkdir(os.path.join(basePath,direc))

statistics = []
for file in files[0]:
    
    # 1 - Data Prepare:      
    # 1.1. Get Date/time observation:
    dateStep = datetime.datetime.strptime(file.split('_')[3], '%Y%m%d')
    print(dateStep)
    base_name = file
    try:
        print("Estimating Surface Temperature from {} file.".format(file))
        waterVaporDF['MOD07File'][np.datetime64(date)] 

        # 1.2. Get Quality Band data:
        qualBand = read_file(os.path.join(L8RawData,file + '_bqa.tif'))
    
        # 1.3. Quality band mask by values pixel values =< 2720 by:
        # https://www.usgs.gov/media/images/landsat-8-quality-assessment-band-pixel-value-interpretations
        qualBand[qualBand>2720] = 0
        qualBand[qualBand<=2720] = 1
        
        # # 1.4. MNDWI Mask:
        # mndwimask = MNDWI(os.path.join(L8RawData,file))
        
        # 1.5. Read Raw Data and apply quality filter and MNDWI mask:  
        # bandB10 = read_file(os.path.join(L8RawData,file + '_b10.tif'))*qualBand
        # bandB11 = read_file(os.path.join(L8RawData,file + '_b11.tif'))*qualBand
        bandB10 = lowPassFilter(os.path.join(L8RawData,file + '_b10.tif'), size = (20,20))*qualBand
        bandB11 = lowPassFilter(os.path.join(L8RawData,file + '_b11.tif'), size = (20,20))*qualBand        
        
        # 1.6. Water Vapor:
        waterVapor = waterVaporDF['MOD07File'].loc[dateStep]
        waterVapor = rastersMatcher(os.path.join(L8RawData,file + '_b10.tif'), waterVapor, 'wp_temp.tif')
        waterVapor = read_file(waterVapor)*qualBand*WV_scale_factor
        waterVapor[waterVapor<0] = np.nan
        
        del qualBand
        
        # 2. Temperature Estimative
        # 2.1. Psi function:
        psi = {}
        atmCoef = {}
        for psiKey  in psiCoefs.keys():
            psi[psiKey] = psiCoefs[psiKey][0]*waterVapor**2 + psiCoefs[psiKey][1]*waterVapor + psiCoefs[psiKey][2]*np.ones(waterVapor.shape)
        
        # 2.2. Atmosferic correction parameters:
        atmCoef['tal'] = 1/psi['psi1']
        atmCoef['Ldown'] = psi['psi3']
        atmCoef['Lup'] = -atmCoef['tal']*(psi['psi2'] + atmCoef['Ldown'])
               
        # 2.3. TOA Radiance correction: 
        LsensorB10 = bandB10*3.3420E-04 + 0.1
        # LcorrB10 = (LsensorB10 - atmCoef['Lup'] - atmCoef['tal']*atmCoef['Ldown']*(1 - emissB10))/(emissB10*atmCoef['tal'])
        
        # del LsensorB10, bandB10
        del bandB10
        
        LsensorB11 = bandB11*3.3420E-04 + 0.1
        # LcorrB11 = (LsensorB11 - atmCoef['Lup'] - atmCoef['tal']*atmCoef['Ldown']*(1 - emissB11))/(emissB11*atmCoef['tal'])
    
        # del LsensorB11, bandB11
        del bandB11
        
        # 2.4. TOA Brigthness Temperature:
        Tb10 = K2_CONSTANT_BAND_10/np.log( K1_CONSTANT_BAND_10/LsensorB10 + 1 )
        Tb11 = K2_CONSTANT_BAND_11/np.log( K1_CONSTANT_BAND_11/LsensorB11 + 1 )
        # del LcorrB10, LcorrB11
        del LsensorB10, LsensorB11
        
        # 2.5. Surface Temperature by Split-Window Algorithm (Jiménez-Muñoz et al.(2014):    
        ST = Tb10 + SWcoef['c1']*(Tb10 - Tb11) + SWcoef['c2']*(Tb10 - Tb11)**2 + SWcoef['c0'] + (SWcoef['c3'] + SWcoef['c4']*waterVapor)*(1-meanEmiss) + (SWcoef['c5'] + SWcoef['c6']*waterVapor)*delEmiss
    
        del Tb10, Tb11, waterVapor
    
        # 1.4. MNDWI Mask:
        mndwimask = MNDWI(os.path.join(L8RawData,file)) 
        ST = ST*mndwimask-273 #Kelvin to Celsius
        ST[ST<=0] = np.nan
        
        del mndwimask
        
        # 2.6. Get Metadatas for output temporary ST file:
        with rasterio.open(os.path.join(L8RawData,file + '_b10.tif')) as src:
            rst = src.read(1)
            out_meta = src.meta
        
        # 2.7. Output ST temporary file
        out_meta['dtype'] = 'float64'
        tempFile = os.path.join(basePath,'Temp_ST_File.tif')   
        with rasterio.open(tempFile , "w", **out_meta) as dest:
            dest.write(ST,1)
        del ST    
        tempFile = cropRst(tempFile, os.path.join(basePath,'Mask','Lakes_Regions.shp'), remove = True, buffer_mask = 0)
        tempFile = cropRst(tempFile, mask = [-53.84,-32.2657579211360073,-52.4983657666122099,-33.7951682672134268 ], remove = False)

        with rasterio.open(tempFile) as src:
            rst = src.read(1)
            out_meta = src.meta

        outs = []
        rasters = []
        dic = {}
        for lake in shapes.keys():
            outname = lake + '.tif'
        
        # 3.1 Crop ST data for each lake:
            # 3.1.1. Read shapefile mask:
            # 0.001 ~ 11.1 m:
            # shapeMask = gpd.GeoDataFrame.from_file(os.path.join(maskPath, shapes[lake]))
            # shapeMask.geometry[0] = shapeMask.geometry[0].buffer(0.005)
        
            # 3.1.2. Crop ST raster by shape mask:
            base_name = file + lake.upper()            
            output_name = base_name + "_" + '_'.join([str(dateStep.year),str(dateStep.month), str(dateStep.day),'ST']) + '.tif'       
            output_name = os.path.join(basePath, 'ST', output_name)
                 
            rasters.append(output_name)
            ST = cropRst(tempFile, os.path.join(maskPath, shapes[lake]), out_tif = output_name, remove = False, buffer_mask = -0.008)
            
            aux_ds = gdal.Open(ST, gdal.GA_ReadOnly)
            ST = rasterio.open(ST)
            # ST[ST<=0] = np.nan
            
            # 3.1.3 Save ST PNG:
            r = ST.read(1)
            r[r<=0] = np.nan
            fig, ax0 = plt.subplots(figsize=(8,10))
            plt1 = ax0.imshow(r, interpolation ='nearest',  cmap = plt.cm.get_cmap('viridis', 20),  aspect = 1)
            #Save file:
            png_out = output_name.split('\\')[-1].replace('.tif','.png')
            png_out = os.path.join(basePath,'PNG',png_out)
            plt.savefig(png_out, dpi = 350,  bbox_inches = 'tight') 
            plt.close()     
            
            # # 3.1.4 K-Means Cluster:
            # # aux_ds = gdal.Open(data, gdal.GA_ReadOnly)
            # aux = aux_ds.GetRasterBand(1).ReadAsArray()
            # aux[aux<=0] = np.nan
            # n_images = 1
            # n_cluster = 5
            # # aux = ST.read(1)
            # Pos = np.argwhere(~np.isnan(aux))
            
            # X = aux.reshape((-1,1))
            # xPos     = np.argwhere(~np.isnan(X)) 
            # rows, cols = zip(*xPos)
            # X = X[rows,cols]
            # X = X.reshape((-1,1))
            
            # array = np.zeros((X.shape[0],1, n_images),
            #          gdal_array.GDALTypeCodeToNumericTypeCode(aux_ds.GetRasterBand(1).DataType))
            # data = [output_name]
            # for i in range(0,len(data)):
            #     array[:,0,i] = clusterPrepare(data[i], band = 1, set_neg_nan = True)
            # array = array.reshape((len(X),n_images))
            
            # print('Data prepared, initializing clustering process.\n')
            # k_means = KMeans(n_clusters=n_cluster, init ='k-means++')
            # k_means.fit(array)
            
            # X_cluster = k_means.labels_
            
            # rows, cols = zip(*Pos)
            # ClustedArray = np.empty((aux.shape[0], aux.shape[1],))
            # ClustedArray[:] = np.nan
            # ClustedArray[rows,cols] = X_cluster
            
            # #X_cluster = X_cluster.reshape(img.shape)
            # fig, ax0 = plt.subplots(figsize=(8,10))
            
            # plt1 = ax0.imshow(ClustedArray, cmap="tab20b")
            
            # # out_meta2 = out_meta
            # # out_meta2['dtype'] = ClustedArray.dtype
            # out_clu = 'Cluster'+output_name.split('\\')[-1]
            # out_clu = os.path.join(basePath,'Cluster',out_clu) 
            # with rasterio.open(out_clu , "w", **out_meta) as dest:
            #     dest.write(ClustedArray,1)
            
            
            # text1 = u'Data: {}-{}-{}'.format(str(dateStep.day).zfill(2),str(dateStep.month).zfill(2),str(dateStep.year)) 
            # ax0.text(.3, 0.95, text1, fontsize = 14,  transform = ax0.transAxes)
            
            # bounds = ST.bounds
            # ax0.xaxis.set_major_formatter(plt.FuncFormatter(format_func_lon))
            # ax0.yaxis.set_major_formatter(plt.FuncFormatter(format_func_lat))
            # ax0.grid(True)   
            
            # png_out = 'Cluster_'+output_name.split('\\')[-1].replace('.tif','.png')  
            # png_out = os.path.join(basePath,'ClassifierPNG',png_out)
            # plt.savefig(png_out, dpi = 350,transparent = True,  bbox_inches = 'tight') 
            
            # plt.close()
            # print("Cluster end\n")            
                     
            
        # 3.2. Compute Temperature Anomaly, Standarlization and Normalization:     
            dt = ST.read(1)
            dt      = dt[~np.isnan(dt)]
            anomaly = ST.read(1) - dt.mean()
            stdnorm = (ST.read(1) - dt.mean())/dt.std()
            norm    = (ST.read(1) - dt.min())/(dt.max() - dt.min())
    
            dicAux = {
                'min_'+lake: dt.min(),
                'max_'+lake: dt.max(),
                'std_'+lake: dt.std(),
                'mean_'+lake: dt.mean(),
                'DateStep': dateStep,
                'Anomaly_'+lake+'_min': anomaly[~np.isnan(anomaly)].min(),
                'Anomaly_'+lake+'_max': anomaly[~np.isnan(anomaly)].max(),
                'stdnorm_'+lake+'_min': stdnorm[~np.isnan(stdnorm)].min(),
                'stdnorm_'+lake+'_max': stdnorm[~np.isnan(stdnorm)].max()                   
            }
                
            dic.update(dicAux)
            
            variables = {
                # 'Anomaly': anomaly,
                'STDNormalized': stdnorm,
                # 'Normalized': norm
                }
            
        # 3.3. Save files to each lake:
            for var in variables.keys():
                output_name = base_name+"_" + '_'.join([str(dateStep.year),str(dateStep.month), str(dateStep.day), var]) + '.tif'
            
                output_name = os.path.join(basePath, var, output_name)
                rasters.append(output_name)
                
                with rasterio.open(output_name, "w", **out_meta) as dest:
                    dest.write(variables[var],1)

            dic['variables'] = rasters   
    
        # # 4. Merge Lakes Rasters:
        out_name_merge = []
        r = 0
            
        out_name = os.path.join(basePath, 'ST_Merge', base_name.split('_')[0]+'_'+ '_'.join([ s.zfill(2) for s in [str(dateStep.year), str(dateStep.month), str(dateStep.day), directories[r]]]) + '.tif')
        out_name_merge.append(out_name)
        
        mergeRst = [rasterio.open(rasters[r]).read(1), rasterio.open(rasters[r+3]).read(1)]
        mergeRst[0][mergeRst[0]<= 0] = np.nan
        mergeRst[1][mergeRst[1]<= 0] = np.nan
        
        rst1 = mergeRst[0]
        Pos1 = np.argwhere(~np.isnan(rst1))
        rows, cols = zip(*Pos1)
        
        outRst = mergeRst[1]
        outRst[rows, cols] = rst1[rows, cols]
        
        with rasterio.open(out_name , "w", **out_meta) as dest:
            dest.write(outRst,1)
                    
        dic['mergeFiles'] = out_name_merge
        statistics.append(dic)
        
    except KeyError:
        print("No have water vapor data to specified date.")
        continue
 

df2 = pd.DataFrame(statistics)
df2.to_csv(os.path.join(basePath,'temp-l8.csv'),sep = ';')


# Standard deviation filter:
basePath = r'C:\Users\00316584\Desktop\Cayo\SRHNE-2020\Temperatura04'
stdir = 'ST\Mangueira'

# 3 std == 99.7%
n_std = 3
files = glob.glob(os.path.join(basePath,stdir,'*tif'))

for file in files:
    with rasterio.open(file) as src:
        data = src.read(1)
        out_meta = src.meta
        out_meta['nodata'] = np.nan
        
    # data = read_file(os.path.join(basePath,'ST','Mangueira',f))
    
        data[data<=0] = np.nan
        ds = data[~np.isnan(data)]

        std = ds.std()
        mean = ds.mean()
        
        data[data<mean-n_std*std] = np.nan
        data[data>=mean + n_std*std] = np.nan
        
        output_name = os.path.join(basePath,stdir,'Corrigida', 'Corr_' + file.split('\\')[-1])
        with rasterio.open(output_name, "w", **out_meta) as dest:
            dest.write(data,1)

sts = glob.glob(os.path.join(basePath,stdir,'Corrigida','*tif'))
max_std = 0
min_std = 10

st = output_name
for st in sts:
    print(st)
    with rasterio.open(st) as src:
        data = src.read(1)
        out_meta = src.meta
        out_meta['nodata'] = np.nan
        
        dt = data
        dt      = dt[~np.isnan(dt)]
        stdnorm = (data - dt.mean())/dt.std()
        if (dt.std()>max_std):
            max_std = dt.std()
        elif (dt.std()<min_std):
            min_std = dt.std()
        
        output_name = os.path.join(basePath,stdir,'Corrigida', 'STDNorm', 'STDNorm_' + st.split('\\')[-1])
        with rasterio.open(output_name, "w", **out_meta) as dest:
            dest.write(stdnorm,1)
 
 #Saving in PNG format:   
path = r'C:\Users\00316584\Desktop\Cayo\SRHNE-2020\Temperatura03\ST'
files = os.listdir(path)
print('Saving PNG.')
for file in files:
    print(file)
    file = os.path.join(path,file)
    r = rasterio.open(file).read(1)
    r[r==0] = np.nan
    fig, ax0 = plt.subplots(figsize=(8,10))
    plt1 = ax0.imshow(r, interpolation ='nearest',  cmap = plt.cm.get_cmap('viridis', 20),  aspect = 1)
    #Save file:
    png_out = file.split('\\')[-1].replace('.tif','.png')
    png_out = os.path.join(basePath,'PNG',png_out)
#        plt.savefig(png_out, transparent = True, dpi = 350,  bbox_inches = 'tight')
    plt.savefig(png_out, dpi = 350,  bbox_inches = 'tight') 
    plt.close()
