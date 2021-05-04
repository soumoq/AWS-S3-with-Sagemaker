#!/usr/bin/env python
# coding: utf-8

# # Reading traning data from s3 bucket

# In[1]:


import boto3
import pandas as pd
import import_ipynb
import s3_bucket_function
from sagemaker import get_execution_role
import numpy as np
import seaborn as sb
import re
import json
import time
from geopy.extra.rate_limiter import RateLimiter
import calendar
from IPython.display import HTML, display
import folium  
import seaborn as sb
from matplotlib import pyplot as plt
#spliting training data in to testing and training part
from sklearn.model_selection import train_test_split


# In[2]:


#1traing_data = s3_bucket_function.s3BucketRead("hmitraningdata","hmitraningdata.csv","csv")


# In[3]:


#2traing_data


# In[4]:


#traing_data.isna().sum()


# In[5]:


#traing_data.describe()


# In[6]:


#3traing_data["Target(Class)"].value_counts()


# In[7]:


#4traing_data["class_name"] =traing_data["Target(Class)"].map({
#1 : "Sudden Acceleration",  
#2 : "Sudden Right Turn",
#3 : "Sudden Left Turn",
#4 : "Sudden Break",
#})


# In[8]:


#5plt.figure(figsize=(20,10))
#sb.countplot(x = "class_name",data = traing_data)
#plt.show()


# Driver Behaviors: Sudden Acceleration (Class Label: 1), Sudden Right Turn (Class Label: 2), Sudden Left Turn (Class Label: 3), Sudden Break (Class Label: 4)

# ML driver Behaviour Classification model

# In[9]:


#6X_feature=traing_data.iloc[:,1:7]


# In[10]:


#7Y_class=traing_data["class_name"]


# In[11]:


#8X_feature.shape


# In[12]:


#9X_train,X_test,Y_train,Y_test=train_test_split(X_feature,Y_class,test_size=.30,random_state=101)


# In[13]:


#Total Sample in Training data and Testing data 
#10print("X_train :{},X_test: {} ,Y_train: {} and Y_test: {}".format(len(X_train),len(X_test),len(Y_train),len(Y_test)))


# In[14]:


#X_train


# In[15]:


#Creating Machine learning Classification Model :KNN


# In[ ]:





# In[16]:


#11max_k=int(np.sqrt(len(X_train) ))
#max_k
#k=list(range(max_k-1))
#len(k)


# In[17]:


#12from sklearn.neighbors import KNeighborsClassifier
#from sklearn.metrics import accuracy_score
#Ya=Y_test # X_test
#list_acc=[]
#for i in range(1,max_k):
#    #print(i)
#    knn_trainer=KNeighborsClassifier(n_neighbors=i)
#    knn_learner=knn_trainer.fit(X_train,Y_train)
#    #Prediction for X_test
#    Yp_knn=knn_learner.predict(X_test)
#    acc_knn=accuracy_score(Ya,Yp_knn)*100
#    #print("acc of knn {} for k {}".format(acc_knn,i))
#    list_acc.append(acc_knn)
    


# In[18]:


#13len(list_acc)


# In[19]:


#14list_arr=np.array(list_acc)
#max_acc=list_arr.max()
#opt_k=list_arr.argmax()
#print(max_acc, opt_k)


# In[20]:


#15from matplotlib import pyplot as plt


# In[21]:


#16plt.plot(k,list_acc)
#plt.plot(k,list_acc,'r*')
#plt.xlabel("k....>")
#plt.ylabel("acc of k")
#plt.show()


# In[22]:


#17#Selected model
#knn_selected=KNeighborsClassifier(n_neighbors=opt_k)
#knn_model=knn_selected.fit(X_train,Y_train)


# In[23]:


#Testing with Real data


# In[24]:


#X_train


# In[25]:


#18import import_ipynb
#import s3_bucket_function


# In[26]:


#19Master Data
#real_driver_master = s3_bucket_function.s3BucketRead("hmimaster","master","csv")


# In[27]:


#real_driver_master.head(1)


# In[28]:


#19def get_acc_gyro_from_file(real_driver_master):
#    real_driver_master=real_driver_master
#    Acc=real_driver_master.iloc[:,5:8]
#    Gyro=real_driver_master.iloc[:,8:11]
#    data_acc_gyro=pd.concat([Gyro,Acc],axis=1)
#    return data_acc_gyro


# In[29]:


#testing driver Behaviour


# In[30]:


#20data_acc_gyro = get_acc_gyro_from_file(real_driver_master)
#Driver_behaviour=knn_model.predict(data_acc_gyro)


# In[31]:


#Driver_behaviour


# In[32]:


#real_driver_master["driver_id"][0]


# In[33]:


#21Result_driver=pd.DataFrame({"Prediction":Driver_behaviour})


# In[ ]:





# In[ ]:





# In[34]:


#22result_hmi=Result_driver["Prediction"].value_counts()
#result_hmi


# In[35]:


#23plt.figure(figsize=(10,5))
#sb.countplot("Prediction",data=Result_driver)
#plt.show()


# In[68]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt

def model_data():
    traing_data = s3_bucket_function.s3BucketRead("hmitraningdata","hmitraningdata.csv","csv")    
    traing_data["Target(Class)"].value_counts()    
    traing_data["class_name"] =traing_data["Target(Class)"].map({
        1 : "Sudden Acceleration",  
        2 : "Sudden Right Turn",
        3 : "Sudden Left Turn",
        4 : "Sudden Break",
    })
    #plt.figure(figsize=(20,10))
    #sb.countplot(x = "class_name",data = traing_data)
    #plt.show()
    X_feature=traing_data.iloc[:,1:7]
    Y_class=traing_data["class_name"]
    X_feature.shape
    X_train,X_test,Y_train,Y_test=train_test_split(X_feature,Y_class,test_size=.30,random_state=101)
    #Total Sample in Training data and Testing data 
    #print("X_train :{},X_test: {} ,Y_train: {} and Y_test: {}".format(len(X_train),len(X_test),len(Y_train),len(Y_test)))
    max_k=int(np.sqrt(len(X_train) ))
    #max_k
    k=list(range(max_k-1))
    len(k)
    
    Ya=Y_test # X_test
    list_acc=[]
    for i in range(1,max_k):
        #print(i)
        knn_trainer=KNeighborsClassifier(n_neighbors=i)
        knn_learner=knn_trainer.fit(X_train,Y_train)
        #Prediction for X_test
        Yp_knn=knn_learner.predict(X_test)
        acc_knn=accuracy_score(Ya,Yp_knn)*100
        #print("acc of knn {} for k {}".format(acc_knn,i))
        list_acc.append(acc_knn)
    len(list_acc)
    list_arr=np.array(list_acc)
    max_acc=list_arr.max()
    opt_k=list_arr.argmax()
    #print(max_acc, opt_k)
    
    #plt.plot(k,list_acc)
    #plt.plot(k,list_acc,'r*')
    #plt.xlabel("k....>")
    #plt.ylabel("acc of k")
    #plt.show()
    
    #Selected model
    knn_selected=KNeighborsClassifier(n_neighbors=opt_k)
    knn_model=knn_selected.fit(X_train,Y_train)
    return knn_model
    
    


# In[69]:


import import_ipynb
import s3_bucket_function

def real_data():
    #Master Data
    real_driver_master = s3_bucket_function.s3BucketRead("hmimaster","master","csv")

    real_driver_master=real_driver_master
    Acc=real_driver_master.iloc[:,5:8]
    Gyro=real_driver_master.iloc[:,8:11]
    data_acc_gyro=pd.concat([Gyro,Acc],axis=1)
    
    data_acc_gyro = data_acc_gyro
    Driver_behaviour=knn_model.predict(data_acc_gyro)
    
    Result_driver=pd.DataFrame({"Prediction":Driver_behaviour})

    result_hmi=Result_driver["Prediction"].value_counts()
    
    #plt.figure(figsize=(10,5))
    #sb.countplot("Prediction",data=Result_driver)
    #plt.show()
    







# In[ ]:





# In[ ]:





# In[ ]:





# # Function for Reading Latest Timesatmp of Driver

# In[38]:


def read_lastest_timestamp(bucketName,driverid):
    file_list = s3_bucket_function.list_of_file_name_from_s3_bucket(bucketName)    
    current_driver_time_info = []
    driver_id_maxtimesatmp = ""
    for i in range(len(file_list)):
        f = file_list[i]
        temp = str(f)
        driver_info = re.findall('\D+[\d]+', temp)
        driver_info_id = driver_info[0]
        driver_info_time = driver_info[1]
        #print(driver_info_time)
        if (driverid==driver_info_id):
            current_driver_time_info.append(driver_info_time[1:])
            
            
    current_driver_time_info.sort()
    for i in range(len(file_list)):
        f = file_list[i]
        temp = str(f)
        driver_info = re.findall('\D+[\d]+', temp)
        driver_info_id = driver_info[0]
        driver_info_time = driver_info[1]
        if (current_driver_time_info[-1] == driver_info_time[1:]):
            driver_id_maxtimesatmp = file_list[i]
            
    return driver_id_maxtimesatmp

#driver_id_maxtimesatmp = read_lastest_timestamp("hmimlbucket","hari_313")


# In[39]:


#driver_id_maxtimesatmp = read_lastest_timestamp("hmimlbucket","hari_313")
#driver_id_maxtimesatmp


# # Driver Trip Analysis function 

# In[40]:


def driver_trip_analysis_function(bucketName,driver_id_maxtimesatmp,filetype):
    real_driver = s3_bucket_function.s3BucketRead("hmimlbucket",driver_id_maxtimesatmp,filetype)
    Acc=real_driver.iloc[:,4:7]
    Gyro=real_driver.iloc[:,7:10]
    data_acc_gyro=pd.concat([Gyro,Acc],axis=1)
    Driver_behaviour=knn_model.predict(data_acc_gyro)
    
    Result_driver=pd.DataFrame({"Prediction":Driver_behaviour})    
    

    return Result_driver,real_driver,Driver_behaviour
    


# In[41]:


#x,y=driver_trip_analysis_function(bucketName,driver_id_maxtimesatmp,filetype)


# In[ ]:





# In[ ]:





# In[42]:


def idel_count(complete_info):
    idel_count = 0
    for i in range(len(complete_info["speed"])):
        if(complete_info["speed"][i] == 0.0):
            idel_count+=1
    return idel_count


# In[43]:


def calculate_behaviour(Driver_behaviour):
    left_turn = 0
    right_turn = 0
    sudden_break = 0
    sudden_acc = 0

    for i in range(len(Driver_behaviour)):
        if(Driver_behaviour[i] == "Sudden Left Turn"):
            left_turn += 1
        if(Driver_behaviour[i] == "Sudden Acceleration"):
            sudden_acc += 1
        if(Driver_behaviour[i] == "Sudden Break"):
            sudden_break += 1
        if(Driver_behaviour[i] == "Sudden Right Turn"):
            right_turn += 1

    return left_turn,right_turn,sudden_break,sudden_acc


# In[44]:


def json_to_csv(final_json):
    i = list(final_json.keys())
    v = list(final_json.values())
    result_csv = pd.DataFrame([v],columns = i)
    return result_csv


# In[45]:


def merge_result_file(result_file):
    result_file = result_file
    colums_name = ['Unnamed: 0',
                     'sudden_left_turn',
                     'sudden_acceleration',
                     'sudden_break',
                     'sudden_right_turn',
                     'driver',
                     'trip_duration',
                     'speed_max',
                     'speed_min',
                     'speed_avg',
                     'start_lat',
                     'start_lon',
                     'end_lat',
                     'end_lon',
                     'start_address',
                     'end_address',
                     'idle_time',
                     'final_score']
    
    df = pd.DataFrame(columns = colums_name)
    
    for i in range(len(result_file)):
        temp = s3_bucket_function.s3BucketRead("hmiresultdriverbehaviourcsv",result_file[i],"csv")
        
        df = df.append(temp,ignore_index=False)
        df = df.drop(["Unnamed: 0"],axis = 1)
        #temp_value = temp.values.tolist()[0]
        #df.insert(i,colums_name,temp_value)
        
    return df


# In[46]:


#call model data and real data
knn_model = model_data()
real_data()


#call function to update
gps = s3_bucket_function.read_data_s3_convart_to_csv("hmiinfogps","gps")
acc = s3_bucket_function.read_data_s3_convart_to_csv("hmiinfoaccelerometer","accelerometer")
gyr = s3_bucket_function.read_data_s3_convart_to_csv("hmiinfogyroscope","gyroscope")

s3_bucket_function.insert_s3_from_calculate_list(gyr,acc,gps)

master_data = s3_bucket_function.marge_all_driver_file("hmimlbucket")
s3_bucket_function.s3BucketWrite("hmimaster","master",master_data)

#call function to analysis
bucketName = "hmimlbucket"
driver_id = "michael_5676"
filetype = "csv"

driver_id_maxtimesatmp = read_lastest_timestamp(bucketName,driver_id)
Result_driver,complete_info,Driver_behaviour = driver_trip_analysis_function(bucketName,driver_id_maxtimesatmp,filetype)


# In[70]:


idel_count = idel_count(complete_info)


# In[ ]:


left_turn,right_turn,sudden_break,sudden_acc = calculate_behaviour(Driver_behaviour)


# In[49]:


#plt.figure(figsize=(10,5))
#sb.countplot("Prediction",data=Result_driver)
#plt.show()


# In[66]:


driver_id=complete_info["driver_id"][0]
trip_duration=complete_info[complete_info["timestamp"]==complete_info["timestamp"].max()]["trip_duration"].tolist()[0]
speed_max=complete_info["speed"].max()
speed_min=complete_info["speed"].min()
speed_avg=complete_info["speed"].sum()/len(complete_info["speed"])


# In[51]:


#trip_gps_info=[driver_id,trip_duration,speed_max,speed_min,speed_avg]
#trip_gps_info


# In[67]:


result_hmi=Result_driver["Prediction"].value_counts()


# In[53]:


driver_beha_infor=result_hmi.values.tolist()


# In[54]:


#latitude, longitude
n = len(complete_info["latitude"])
start_lat = complete_info["latitude"][0]
end_lat = complete_info["latitude"][n-1]
start_lon = complete_info["longitude"][0]
end_lon = complete_info["longitude"][n-1]


# In[65]:


final_json = {
  "sudden_left_turn": left_turn,
  "sudden_acceleration": sudden_acc,
  "sudden_break": sudden_break,
  "sudden_right_turn": right_turn,
  "driver": driver_id,
  "trip_duration": trip_duration,
  "speed_max": speed_max,
  "speed_min":speed_min,
  "speed_avg": speed_avg,
  "start_lat": start_lat,
  "start_lon": start_lon,
  "end_lat": end_lat,
  "end_lon": end_lon,
  "start_address":"",
  "end_address":"",
  "idle_time": idel_count,
  "final_score":"30"
}
driver_id_maxtimesatmp


# In[56]:


#final_json


# In[57]:


result_csv = json_to_csv(final_json)


# In[58]:


json_string = json.dumps(final_json, indent = 4)  


# In[59]:


#json_object = json.loads(json_string)


# In[60]:


s3_bucket_function.s3BucketWriteText("hmiresultdriverbehaviour", driver_id_maxtimesatmp, json_string)


# In[61]:


s3_bucket_function.s3BucketWrite("hmiresultdriverbehaviourcsv", driver_id_maxtimesatmp, result_csv)


# In[ ]:





# In[62]:


driver_res_file = s3_bucket_function.list_of_file_name_from_s3_bucket("hmiresultdriverbehaviourcsv")


# In[63]:


#s3_bucket_function.s3BucketRead("hmiresultdriverbehaviourcsv",driver_res_file[1],"csv")


# In[64]:


master_b = merge_result_file(driver_res_file)
s3_bucket_function.s3BucketWrite("hmidriverdehavior","result_driver_master",master_b)
#master_b


# In[ ]:




