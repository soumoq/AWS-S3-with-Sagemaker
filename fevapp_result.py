#!/usr/bin/env python
# coding: utf-8

# import boto3
# import pandas as pd
# import re
# import json
# import import_ipynb
# import s3_bucket_function
# from sagemaker import get_execution_role

# In[ ]:


userId = "debojyoti_8637"


# In[ ]:


def reading_app_data(bucketName="febappmastercsv",fileName="master",file_type="csv"):
    app_data=s3_bucket_function.s3BucketRead(bucketName,fileName,file_type)
    app_data.drop(["Unnamed: 0"],axis=1,inplace=True)
    return app_data
    


# In[ ]:


app_data=reading_app_data(bucketName="febappmastercsv",fileName="master",file_type="csv")
app_data


# In[ ]:


#app_data.head(1)


# In[ ]:


#app_data["app_name"].value_counts()


# In[ ]:


#Displaying the top n used data on the basis of time spent 


# In[ ]:


def fev_app(data,userId="NA",n=10):
    data=data
    userId=userId
    n=n
    if userId=="NA":
        return_data=data
    else:
        d=data[data["user_id"]==userId]
        data_=d
        app_useedcount=data_["app_name"].value_counts()
        n=len(app_useedcount)
        app_used_result=pd.DataFrame({"App":app_useedcount.keys().tolist(),"NumerofTimeUsed":app_useedcount.values.tolist()})
        app_used_result=app_used_result[:10]
        return_data=data_.sort_values("Duration", ascending = False)
        top_n_app_duraion=return_data[["app_name","Duration"]]
        top_n_app_duraion.reset_index(inplace=True)
        top_n_app_duraion.drop("index",axis=1,inplace=True)
        top_n_app_duraion["DurationInMinute"]=top_n_app_duraion["Duration"]/60000
        #top_n_app_duraion.drop_duplicates(keep='first', inplace=True, ignore_index=False)
        topapp_duration=top_n_app_duraion.groupby(['app_name'],as_index=False,sort=False)['DurationInMinute'].sum()
        topapp_duration=topapp_duration[:10]
    
    return app_used_result,topapp_duration


# In[ ]:


app_used_count,topapp_duration=fev_app(app_data,userId)


# In[ ]:


app_used_count


# In[ ]:


topapp_duration


# In[ ]:


def convart_json(app_name,data,key):
    json_app = {
    "feb_app":[]
    }
    res = {}
    for i in range(len(app_name)):
        res[i] = {
            "app_name":app_name[i],
            key:data[i]
        }
    for i in range(len(res)):
        json_app["feb_app"].append(res[i])
    return str(json_app)


# In[ ]:


app_name = topapp_duration["app_name"].to_list()
DurationInMinute = topapp_duration["DurationInMinute"].to_list()
top_app_json = convart_json(app_name,DurationInMinute,"duration")
top_app_json


# In[ ]:


app_name = app_used_count["App"].to_list()
numOfUse = app_used_count["NumerofTimeUsed"].to_list()
app_count_json = convart_json(app_name,numOfUse,"num_of_use")
app_count_json


# In[ ]:


s3_bucket_function.s3BucketWriteText("febappresultjson", userId+"_count", app_count_json)
s3_bucket_function.s3BucketWriteText("febappresultjson", userId+"_duraion", top_app_json)


# In[ ]:


import time


# In[ ]:


timestamp = 1619620756203
time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(timestamp))


# In[ ]:


#from datetime import datetime

#timestamp = 1545730073
#dt_object = datetime.fromtimestamp(timestamp)

#print("dt_object =", dt_object)
#print("type(dt_object) =", type(dt_object))


# In[ ]:


#from datetime import datetime

# current date and time
#now = datetime.now()

#timestamp = datetime.timestamp(now)
#print("timestamp =", timestamp)


# In[ ]:




