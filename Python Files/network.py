import torch
#import torchvision
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import numpy
import json
import random
from random import choice
#if list TypeError del list

#Defining our neural network
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(34, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 16)
        self.fc4 = nn.Linear(16, 5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)

#function to convert string to ascii character
def ASCII(s):
    x = 0
    for i in range(len(s)):
        x += ord(s[i])*2**(8 * (len(s) - i - 1))
    return x

#Reading in data from json file
with open(r'C:\Users\ahalw\Downloads\data(1).json') as f:
    Data = json.load(f)
data = Data['data']
str_array=[]
List=[]

#convert class labels in List from string to ascii
for key in data:
    temp = key['ys']
    if len(str_array)==0:
        str_array.append(temp['0'])
    else:
        if temp['0'] not in str_array:
            str_array.append(temp['0'])
    List.append([list(key['xs'].values()), list(key['ys'].values())])
for i in range(len(List)):
    List[i][0]=List[i][0]
    List[i][1]=[ASCII(element) for element in List[i][1]]

#frequency of each case
y = {}
for j in str_array:
    counter = 0
    for i in range(len(List)):
        if chr(List[i][1][0]) == j:
            counter += 1
    y[j] = counter

#split List into training and testing set by class label
train=[]
test=[]
start=0
end=0
counter=1
for key in y:
    if counter==1:
        end=y[key]-1
    else:
        end+=y[key]
    Train, Test=train_test_split(List[start:end], test_size=0.33)
    start+=y[key]
    for i in Train:
        train.append(i)
    for i in Test:
        test.append(i)
    counter+=1

#replace ascii values with 0 to (n-1) values where n is the number of classes
num=list(range(0,len(str_array)))
dic={}
index=0
for elem in str_array:
    dic[ASCII(elem)]=num[index]
    index+=1
for i in range(len(train)):
    temp=train[i][1][0]
    train[i][1]=dic[temp]
for i in range(len(test)):
    temp=test[i][1][0]
    test[i][1]=dic[temp]

#batch the data [10] randomized for each batch -- test set
batch=10
batch_n=int(len(train)/batch)
#print(len(train))
train_indices=list(range(0,len(train)))
train_set=[]

for n in range(batch_n):
    train_x=[]
    train_y=[]
    for i in range(batch):
        rand_index=choice(train_indices)
        train_x.append(train[rand_index][0])
        train_y.append(train[rand_index][1])
        train_indices.remove(rand_index)
    #turn into tensors
    train_set.append([torch.Tensor(train_x),torch.Tensor(train_y).long()])

#batch the data [10] randomized for each batch -- train set
batch=10
batch_n=int(len(test)/batch)
test_indices=list(range(0,len(test)))
test_set=[]
for n in range(batch_n):
    test_x=[]
    test_y=[]
    for i in range(batch):
        rand_index=choice(test_indices)
        test_x.append(test[rand_index][0])
        test_y.append(test[rand_index][1])
        test_indices.remove(rand_index)
    #turn into tensors
    test_set.append([torch.Tensor(test_x),torch.Tensor(test_y).long()])

#Train our network using the training set
net=Net()
optimizer = optim.Adam(net.parameters(), lr=0.001)
EPOCHS = 100
for epoch in range(EPOCHS):
    for data in train_set:
        X, y = data
        net.zero_grad()
        output = net(X)
        loss = F.nll_loss(output, y)
        loss.backward()
        optimizer.step()
#print loss once training is complete
#print(loss)

#classification
def classify(coord):
    inp = torch.Tensor(list(coord)).reshape(1, 34)
    idx = int(torch.argmax(net(inp)).numpy())
    return str_array[idx]
"""
#check accuracy of network
correct=0
total=0

with torch.no_grad():
    for data in test_set:
        X, y=data
        output=net(X)
        for idx, i in enumerate(output):
            if torch.argmax(i)== y[idx]:
                correct+=1
            total+=1
#print("Accuracy: ", round(correct/total, 3))
"""