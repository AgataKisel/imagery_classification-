from __future__ import print_function, division
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from osgeo import gdal, ogr
import os
# os.environ['PROJ_LIB'] = '***'
# os.environ['GDAL_DATA'] = '***'

def prepare_roi_from_vector(filepath: str, img_ds: gdal.Dataset, attribute_name: str) -> np.ndarray:
    """
    Parametrs
    ---------
    filepath: str
        path to source gdal-compatible traning vector

    img_ds: osgeo.gdal.Dataset
        initial image with goespatial reference

    attribute_name: str
        the name of the attribute with class data
        
    Returns
    --------
    dataset: numpy.ndarray
        raster of traning  

    """
    shape_dataset = ogr.Open(filepath)
    shape_layer = shape_dataset.GetLayer()
    attributes = []
    ldefn = shape_layer.GetLayerDefn()
    
    for n in range(ldefn.GetFieldCount()):
        fdefn = ldefn.GetFieldDefn(n)
        attributes.append(fdefn.name)  

    mem_drv = gdal.GetDriverByName('MEM')
    mem_raster = mem_drv.Create('',img_ds.RasterXSize,img_ds.RasterYSize,1,gdal.GDT_UInt16)
    mem_raster.SetProjection(img_ds.GetProjection())
    mem_raster.SetGeoTransform(img_ds.GetGeoTransform())
    mem_band = mem_raster.GetRasterBand(1)
    mem_band.Fill(0)
    mem_band.SetNoDataValue(0)

    err = gdal.RasterizeLayer(mem_raster, [1], shape_layer, None, None, [1],  ['ALL_TOUCHED=TRUE', f'ATTRIBUTE={attribute_name}'])
    assert err == gdal.CE_None
    dataset = mem_raster.ReadAsArray()
    return dataset

def random_forest_classification(
    input_path_initial_image: str, 
    input_path_traning_data: str, 
    output_path: str,
    report_path: str,
    attribute_name: str = "id",
    n_estimators: int = 100,
    max_depth: int = None,
) -> int:
    """
    Performs random forest classification of any gdal-compatible raster
    
    Parametrs
    ---------
    input_path: str
        path to source gdal-compatible initial raster

    input_path_traning_data: str
        path to source gdal-compatible traning vector

    output_path: str
        path to create new raster in
        
    report_path: str
        path to save metrics information
        
    attribute_name: str
        the name of the attribute with class data

    Returns
    --------
    int 
        0 if finished correct
        1 if invalid data source
        2 if error in classification
        3 if error in file creation
    Description 
    -----------
    based on https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    """
    
    gdal.UseExceptions()
    gdal.AllRegister()
    
    img_ds = gdal.Open(input_path_initial_image, gdal.GA_ReadOnly)

    if (img_ds==None):
        return 1 
    
    num_rasters = img_ds.RasterCount

    training = input_path_traning_data
    roi = prepare_roi_from_vector(training, img_ds, attribute_name=attribute_name)  
    
    img = img_ds.ReadAsArray()

    if num_rasters > 1:
        img = np.moveaxis(img, 0, -1)
    else:
        img = img[..., np.newaxis]
    
    X = img[roi > 0, :]
    y = roi[roi > 0]
    X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8, shuffle=True)
    
    try:
        rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, oob_score=True)
        rf = rf.fit(X_train, y_train)
    except Exception as e: 
        print(e)
        return 2

    # error in the training sample
    accuracy = rf.oob_score_ * 100
    print(accuracy)
    print(f"Точность на обучающей выборке {accuracy}", file=open(report_path, "w"))

    # metrics precision, recall, f1 for validation data
    y_val_pred = rf.predict(X_val)
    labels = np.unique(roi[roi > 0])
    target_names = list()
    for name in range(1, labels.size + 1):
        target_names.append(str(name))
    sum_mat = classification_report(y_val, y_val_pred, target_names=target_names)
    print(sum_mat)
    print("Metrics by class", file=open(report_path, "a"))
    print(sum_mat, file=open(report_path, "a"))
    # confusion matrix для валидационных данных
    conf_matrix = pd.crosstab(y_val, y_val_pred, margins=True, rownames=['True'], colnames=['Predicted'])
    print("Classification error matrix", file=open(report_path, "a"))
    print(conf_matrix, file=open(report_path, "a"))
    
    prediction = np.where(img[:,:,0] > 0, rf.predict(img.reshape(-1, num_rasters)).reshape(img.shape[:2]), img[:,:,0])

    try:
        format = "GTiff"
        driver = gdal.GetDriverByName(format)
        out_data_raster = driver.Create(output_path, img_ds.RasterXSize, img_ds.RasterYSize, 1, gdal.GDT_Byte)
        out_data_raster.SetGeoTransform(img_ds.GetGeoTransform())
        out_data_raster.SetProjection(img_ds.GetProjection())
        
        out_data_raster.GetRasterBand(1).WriteArray(prediction)
        out_data_raster.FlushCache() 
        del out_data_raster
    except:
        return 3
        
    return 0