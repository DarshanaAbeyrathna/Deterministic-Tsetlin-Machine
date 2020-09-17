import numpy as np
from numpy import array
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import pyximport; pyximport.install(setup_args={
                              "include_dirs":np.get_include()},
                            reload_support=True)

T = 10
s = 3
number_of_clauses = 100
states = 100
epochs = 100

import DeterministicTsetlinMachine

f=open("Qualitative_Bankruptcy.data.txt","r")

#df = np.zeros((286, 4))
p0, p1, p2, p3, p4, p5, label = [], [], [], [], [], [], []
x = f.readlines()
for j in range(250):
        p0.append(x[j].split(',')[0])
        p1.append(x[j].split(',')[1])
        p2.append(x[j].split(',')[2])
        p3.append(x[j].split(',')[3])
        p4.append(x[j].split(',')[4])
        p5.append(x[j].split(',')[5])
        label.append(x[j].split(',')[6])


p0, p1, p2, p3, p4, p5, label = \
array(p0), array(p1), array(p2), array(p3), array(p4), array(p5), array(label)
label_encoder = LabelEncoder()
p0 = label_encoder.fit_transform(p0).reshape((250, 1))
p1 = label_encoder.fit_transform(p1).reshape((250, 1))
p2 = label_encoder.fit_transform(p2).reshape((250, 1))
p3 = label_encoder.fit_transform(p3).reshape((250, 1))
p4 = label_encoder.fit_transform(p4).reshape((250, 1))
p5 = label_encoder.fit_transform(p5).reshape((250, 1))
label = label_encoder.fit_transform(label).reshape((250, 1))

df = np.hstack((p0, p1, p2, p3, p4, p5))

dataframe = np.zeros([1,df.shape[1]])
P0 = [] # 0 - negative.... 1- positive
for kk in range(df.shape[0]):
    if label[kk] == 0 or label[kk] == 1:
        dataframe = np.append(dataframe, df[kk,:].reshape((1, dataframe.shape[1])), axis=0)
        if label[kk] == 0:
            P0 = np.append(P0, 0)
        elif label[kk] == 1:
            P0 = np.append(P0, 1)
        
dataframe = dataframe[1:len(dataframe),:]
arr_Selected = [[] for _ in range(len(df[0]))]

NOofThresholds = 0
for i in range(len(dataframe[0])):
    uniqueValues = list(set(dataframe[:,i]))
    NOofThresholds = NOofThresholds + len(uniqueValues)

NewData = np.zeros((len(dataframe), NOofThresholds+1))

m = -1
for i in range(len(dataframe[0])):
    uniqueValues = list(set(dataframe[:,i]))
    uniqueValues.sort() 
    arr_Selected[i].append(uniqueValues)
    NOofuniqueValues = len(uniqueValues)
    for j in range(NOofuniqueValues):
        m += 1
        for k in range(len(dataframe)):
            if dataframe[k,i] <= uniqueValues[j]:
                NewData[k,m] = 1
            else:
                NewData[k,m] = 0
                
NewData[:,NOofThresholds] = P0
np.random.shuffle(NewData)

out = np.zeros((6, 5), dtype=np.float32)

outval = 0
for d in [1, 10, 100, 500, 1000, 5000]:
    determinism = d
    np.random.shuffle(NewData)
    
    NOofTestingSamples = len(NewData)*20//100
    NOofTrainingSamples = len(NewData)-NOofTestingSamples
    
    
    X_train = NewData[0:NOofTrainingSamples,0:len(NewData[0])-1].astype(dtype=np.int32)
    y_train = NewData[0:NOofTrainingSamples,len(NewData[0])-1:len(NewData[0])].flatten().astype(dtype=np.int32)
    rows, number_of_features = X_train.shape
        
    X_test = NewData[NOofTrainingSamples:NOofTrainingSamples+NOofTestingSamples,0:len(NewData[0])-1].astype(dtype=np.int32)
    X_test.tolist()
    y_test = NewData[NOofTrainingSamples:NOofTrainingSamples+NOofTestingSamples,len(NewData[0])-1:len(NewData[0])].flatten().astype(dtype=np.int32)
    
    tsetlin_machine = DeterministicTsetlinMachine.TsetlinMachine(number_of_clauses, number_of_features, states, s, T, determinism, boost_true_positive_feedback = 1)
    tsetlin_machine.fit(X_train, y_train, y_train.shape[0], epochs=epochs)
   
    print("'d' value: %d Training Accuracy: %.3f Testing Accuracy: %.3f" % (d, tsetlin_machine.evaluate(X_test, y_test, y_test.shape[0]), tsetlin_machine.evaluate(X_train, y_train, y_train.shape[0])))