#encoding:utf-8
import tensorflow as tf
import numpy as np
import time
import cv2
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import psycopg2
import pandas as pd
import os

#Imagenet图片都保存在/data目录下，里面有1000个子目录，获取这些子目录的名字
classes = os.listdir('train/')

#构建一个字典，Key是目录名，value是类名0-999
labels_dict = {}
for i in range(len(classes)):
    labels_dict[classes[i]]=i
print(classes[999])

#构建一个列表，里面的每个元素是图片文件名+类名
train_data = list()
images_labels_list = []
error_list = []
for i in range(len(classes)):
    path = 'train/'+classes[i]+'/'
    images_files = os.listdir(path)
    label = str(labels_dict[classes[i]])
    for image_file in images_files:
        tmp = list()
        try:
            f = open('/home/gpadmin/data/imagenet/train/{}/{}'.format(classes[i], image_file), 'rb')
            img_bytes = f.read()
            f.close()
            img_array = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            x_i = cv2.resize(img, (224, 224)).astype(np.float32) / 255.0
        except:
            res = []
            res.append(classes[i])
            res.append(image_file)
            error_list.append(res)
            continue
        x_i = x_i.flatten()
        x_i_str = ",".join(str(x) for x in x_i)
        c = "{" + x_i_str + "}"
        if len(train_data) < 10:
            print(c)
            print(label)
        tmp.append(c)
        tmp.append(label)
        train_data.append(tmp)
cifa = pd.DataFrame(train_data,columns=['x', 'target'])
print(cifa.head(10))
cifa.to_csv('imagenet_1.csv',index = True)
error_file = pd.DataFrame(error_list,columns=['class','path'])
error_file.to_csv('error_file.csv', index = True)
