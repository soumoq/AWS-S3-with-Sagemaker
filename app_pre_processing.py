#!/usr/bin/env python
# coding: utf-8

# In[1]:


import boto3
import pandas as pd
import re
import json
import import_ipynb
import datetime
import s3_bucket_function
from sagemaker import get_execution_role


# In[2]:


def convart_time(u_time):
    data = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(int(u_time)))
    return str(data)


# In[3]:


def raw_to_csv(data,file_name):
    data = data
    all_data = data.values
    all_json_array = []
    for x in range(0,len(all_data),5):
        all_json_array.append(
            {
                "app_name" : all_data[x].tolist()[0],  
                "user_id" : all_data[x+1].tolist()[0],
                "total_use_time" : all_data[x+2].tolist()[0],
                "last_use_time" : all_data[x+3].tolist()[0],
                "fast_use_time" : all_data[x+4].tolist()[0],
            })
        all_json_obj = {
                "all_data" : all_json_array
            }
        
        d_data=all_json_obj["all_data"]

        app_name = []
        user_id = []
        total_use_time = []
        last_use_time = []
        fast_use_time = []
        upload_timestamp = []
        
        for i in range(len(d_data)):
            ds=d_data[i]
            app_name.append(ds["app_name"])
            user_id.append(ds["user_id"])
            total_use_time.append(ds["total_use_time"])
            last_use_time.append(ds["last_use_time"])
            fast_use_time.append(ds["fast_use_time"])
            
            temp = file_name
            time = re.findall('\D+[\d]+', temp)
            time = time[1]
            time = time[1:]
            upload_timestamp.append(time)
            
        dataframe = pd.DataFrame({"app_name":app_name,"user_id":user_id,"TimeStamp":upload_timestamp,"Duration":total_use_time,"EndTime":last_use_time,"StartTime":fast_use_time})
        
    
    
    return dataframe


# In[4]:


def calculate_all_raw_file():
    file_list = s3_bucket_function.list_of_file_name_from_s3_bucket("febappuser")    
    all_cav_data = {}
    for i in range(len(file_list)):
        data = s3_bucket_function.s3BucketRead("febappuser",file_list[i],"txt")
        all_cav_data[i] = raw_to_csv(data,file_list[i]) 
        
        temp = file_list[i]
        file_name = re.findall('\D+[\d]+', temp)
        file_name = file_name[0] + file_name[1]
        
        s3_bucket_function.s3BucketWrite("febappcsv", file_name, all_cav_data[i])
    
    return all_cav_data


# In[5]:


def marge_csv(all_data):
    appended_data = []
    for i in range(len(all_data)):
        appended_data.append(all_data[i])
        
    appended_data = pd.concat(appended_data,ignore_index=True)
    s3_bucket_function.s3BucketWrite("febappmastercsv", "master", appended_data)
    
    return appended_data


# In[6]:


datetime.datetime.fromtimestamp(1347517370).strftime('%c')


# In[7]:


all_data = calculate_all_raw_file()
marge_csv(all_data)

