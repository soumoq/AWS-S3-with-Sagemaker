#!/usr/bin/env python
# coding: utf-8

# # Read file

# In[1]:


import boto3
import pandas as pd
import re
import json
from sagemaker import get_execution_role


# In[2]:


#Using this function we can wright data to s3
def s3BucketWrite(bucketName, fileName, data):
    bucket = bucketName
    data_key = fileName
    data_location = 's3://{}/{}'.format(bucket, data_key)
    data.to_csv(data_location)
    


# In[3]:


def s3BucketWriteText(bucketName, fileName, data):
    bucket = bucketName
    data_key = fileName
    #data_location = 's3://{}/{}'.format(bucket, data_key)
    #data.to_csv(data_location)
    #client = boto3.client('s3')
    s3 = boto3.resource('s3')
    object = s3.Object(bucketName, fileName)
    object.put(Body=data)


# In[4]:


#Using this function we can read data to s3
def s3BucketRead(bucketName, fileName,file_type = "xlsx"):
    bucketName = bucketName
    fileName = fileName
    #role = get_execution_role()
    #bucket='hmidriverdata'
    #driver_key = 'driver.xlsx'
    data_location = 's3://{}/{}'.format(bucketName, fileName)    
    if file_type == "xlsx":
        data = pd.read_excel(data_location)
    if file_type == "csv":
        data = pd.read_csv(data_location)
    if file_type == "txt":
        data = pd.read_csv(data_location,sep="\t",header = None)
    return data


# In[5]:


#Using this function we can convart RAW data to CSV
def raw_to_csv_data(data, driver_id, raw_to_data):
    data = data
    driver_id = driver_id
    if raw_to_data == "gyroscope":
        gyr_data = data.values
        gyr_json_array = []
        for x in range(0,len(gyr_data),5):
            gyr_json_array.append(
            {
                "gyroscope_time" : gyr_data[x].tolist()[0],  
                "trip_duration" : gyr_data[x+1].tolist()[0],
                "gyroscope_x" : gyr_data[x+2].tolist()[0],
                "gyroscope_y" : gyr_data[x+3].tolist()[0],
                "gyroscope_z" : gyr_data[x+4].tolist()[0],
            })
    
        gyr_json_obj = {
            "gyr_data" : gyr_json_array
        }

        d_data=gyr_json_obj["gyr_data"]
        timestamp=[]
        trip_duration=[]
        gyr_x=[]
        gyr_y=[]
        gyr_z=[]
        list_driver_id=[]
        for i in range(len(d_data)):
            ds=d_data[i]
            list_driver_id.append(driver_id)
            timestamp.append(ds["gyroscope_time"])
            trip_duration.append(ds["trip_duration"])
            gyr_x.append(ds["gyroscope_x"])
            gyr_y.append(ds["gyroscope_y"])
            gyr_z.append(ds["gyroscope_z"])
    
        gyr_dataframe = pd.DataFrame({"driver_id":list_driver_id,"timestamp":timestamp,"trip_duration":trip_duration,"gyroscope_x":gyr_x,"gyroscope_y":gyr_y,"gyroscope_z":gyr_z})
        data_return = gyr_dataframe
        
        
    if raw_to_data == "accelerometer":
        acc_data = data.values
        acc_json_array = []
        for x in range(0,len(acc_data),5):
            acc_json_array.append(
            {
                "accelerometer_time" : acc_data[x].tolist()[0],  
                "trip_duration" : acc_data[x+1].tolist()[0],
                "accelerometer_x" : acc_data[x+2].tolist()[0],
                "accelerometer_y" : acc_data[x+3].tolist()[0],
                "accelerometer_z" : acc_data[x+4].tolist()[0],
            })
    
        acc_json_obj = {
            "acc_data" : acc_json_array
        }


        d_data=acc_json_obj["acc_data"]
        timestamp=[]
        trip_duration=[]
        acc_x=[]
        acc_y=[]
        acc_z=[]
        list_driver_id=[]
        for i in range(len(d_data)):
            ds=d_data[i]
            list_driver_id.append(driver_id)
            timestamp.append(ds["accelerometer_time"])
            trip_duration.append(ds["trip_duration"])
            acc_x.append(ds["accelerometer_x"])
            acc_y.append(ds["accelerometer_y"])
            acc_z.append(ds["accelerometer_z"])
    
        acc_dataframe = pd.DataFrame({"driver_id":list_driver_id,"timestamp":timestamp,"trip_duration":trip_duration,"accelerometer_x":acc_x,"accelerometer_y":acc_y,"accelerometer_z":acc_z})
        data_return = acc_dataframe
            
    
    if raw_to_data == "gps":
        gps_data = data.values
        gps_json_array = []
        for x in range(0,len(gps_data),5):
            gps_json_array.append(
            {
                "gps_time" : gps_data[x].tolist()[0],  #gps_data[x].tolist()
                "trip_duration" : gps_data[x+1].tolist()[0],
                "latitude" : gps_data[x+2].tolist()[0],
                "longitude" : gps_data[x+3].tolist()[0],
                "speed" : gps_data[x+4].tolist()[0],
            })

        gps_json_obj = {
            "gps_data" : gps_json_array
        }

        d_data=gps_json_obj["gps_data"]
        timestamp=[]
        trip_duration=[]
        lat=[]
        log=[]
        speed=[]
        list_driver_id=[]
        for i in range(len(d_data)):
            ds=d_data[i]
            list_driver_id.append(driver_id)
            timestamp.append(ds["gps_time"])
            trip_duration.append(ds["trip_duration"])
            lat.append(ds["latitude"])
            log.append(ds["longitude"])
            speed.append(ds["speed"])
    
    
        gps_dataframe = pd.DataFrame({"driver_id":list_driver_id,"timestamp":timestamp,"trip_duration":trip_duration,"latitude":lat,"longitude":log,"speed":speed})
        data_return = gps_dataframe

           
            
    return data_return

    



# In[6]:


# Using this function we can read all filename from s3
def list_of_file_name_from_s3_bucket(bucket_name):
    list_name = []
    bucket_name = bucket_name
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(bucket_name)
    for obj in bucket.objects.all():
        list_name.append(obj.key)
        #print(obj.key)
    return list_name


# In[7]:


#using this function to merge into single file
def marg_in_single_csv(gyroscope_data, accelerometer_data,gps_data):
    gyroscope_data = gyroscope_data
    accelerometer_data = accelerometer_data
    gps_data = gps_data
    
    data = accelerometer_data.copy()
    data["gyroscope_x"] = gyroscope_data["gyroscope_x"]
    data["gyroscope_y"] = gyroscope_data["gyroscope_y"]
    data["gyroscope_z"] = gyroscope_data["gyroscope_z"]
    data["latitude"] = gps_data["latitude"]
    data["longitude"] = gps_data["longitude"]
    data["speed"] = gps_data["speed"]
    return data
    


# In[8]:


# This function used for read the data from s3 and convert into CSV
def read_data_s3_convart_to_csv(bucket_name,key):
    bucket_name = bucket_name
    file_name = list_of_file_name_from_s3_bucket(bucket_name)
    data_=pd.DataFrame()
    appended_data = []
    data = {}
    for x in range(len(file_name)):
        f = file_name[x]
        temp = str(f)
        driver_id = re.findall('\D+[\d]+', temp)
        driver_id = driver_id[0]
        gy_data = s3BucketRead(bucket_name,f, "txt")
        data[x]  = raw_to_csv_data(gy_data,driver_id,key)
        #appended_data.append(data)
    #appended_data = pd.concat(appended_data,ignore_index=True)
    return data
        
    


# In[9]:


#This function create file name with driver id and time stamp
def generate_file_name_timestamp(bucket_name, master_data):
    bucket_name = bucket_name
    master_data = master_data
    p,n = master_data.shape
    timestamp_filename = master_data.iloc[p-1,1]
    id_filename = master_data.iloc[p-1,0]
    #print(id_filename+"_"+timestamp_filename)
    file_name = id_filename+"_"+timestamp_filename
    s3BucketWrite(bucket_name, file_name, master_data)    


# In[10]:


# This function takes list of CSV parameter and store into s3 one by one 
def insert_s3_from_calculate_list(gyr,acc,gps):
    gyr = gyr
    acc = acc
    gps = gps
    for x in range(len(gps)):
        master = {}
        master[x] = marg_in_single_csv(gyr[x], acc[x],gps[x])
        generate_file_name_timestamp("hmimlbucket",master[x])    
        #print(x)


# In[11]:


# Using this function we get all muster driver file and marge one single master file
def marge_all_driver_file(bucket_name):
    bucket_name = bucket_name
    file_name = list_of_file_name_from_s3_bucket(bucket_name)
    appended_data = []
    for x in range(len(file_name)):
        data = s3BucketRead(bucket_name,file_name[x], "csv")
        appended_data.append(data)
    appended_data = pd.concat(appended_data,ignore_index=True)
    return appended_data


# In[ ]:





# In[ ]:




