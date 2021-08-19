import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import random

import socket
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(('3.142.167.4', 15271))
# client.connect(('127.0.0.1', 60000))
import rss22

pid = 2

def rssrd(r, xy):

    f = {}
    g = {}
    R = {}
    esend = {}
    epk1 = {}

    for j in range(rss22.n):
        if j+1==pid:
            continue

        f[pid,(j+1)] = round(random.uniform(3*(10**7),4*(10**7)),6)
        g[pid,(j+1)] = round(random.uniform(3*(10**7),4*(10**7)),6)
        R[pid,(j+1)] = random.uniform(11*(10**18),19*(10**18))

    for j in range(rss22.n):
        if j+1==pid:
            continue

        prod = f[pid,(j+1)] * r
        esend[pid,(j+1)] = ( rss22.public_key.encrypt(prod) , f[pid,(j+1)] )

    for j in range(1,4):
        if j == pid:
            rss22.client_send(esend, client)
            print('sent')
        else:
            print("Ready to receive")
            rss22.client_receive(pid, client)

    print("Received data")
    print(rss22.erecive)

    fj = {}

    for i in rss22.erecive.keys():
        epk1[i[0],i[1]]=( rss22.erecive[i][0] * g[i[1],i[0]] * xy + R[i[1],i[0]] , g[i[1],i[0]] )

        fj[i] = rss22.erecive[i][1]

    print("fj ",fj,"\n")
    print()

    for j in range(1,4):
        if j == pid:
            rss22.epk_send(epk1, client)
        else:
            rss22.epk_receive(pid, client)

    print("Received dat 01a")
    print(rss22.epkfinal)

    share1 = {}
    share2 = {}

    for i in rss22.epkfinal.keys():
        nr = rss22.private_key.decrypt(rss22.epkfinal[i][0])

        dr = rss22.epkfinal[i][1] * f[i]

        share1[i] = nr/dr

        share2[i] = - R[i] / ( fj[(i[1],i[0])] * g[i] )

        print('ok')

    t = round(random.uniform((-0.5),(0.5)),6)

    si = 0

    for i in share1.keys():
        si += share1[i] + share2[i] + ( r + t ) * xy

    rss22.s = []

    for j in range(1,4):
        if j == pid:
            rss22.si_send(si, client)
        else:
            rss22.si_receive(client)

    rss22.s.append(si)
    print(rss22.s)

    return sum(rss22.s)

def rss(d):

    x, y = d['x'], d['y']

    alphax = round(random.uniform((-0.5),(0.5)),6)
    alphay = round(random.uniform((-0.5),(0.5)),6)

    x = x + alphax
    y = y + alphay

    r = round(random.uniform(3000,4000),6)

    sx = rssrd(r, x)

    sy = rssrd(r, y)

    return sx/sy

def fdrt(dat, alpha, beta):
    
    cos_alpha = round(math.cos(alpha),4)
    sin_alpha = round(math.sin(alpha), 4)
    
    cos_beta = round(math.cos(beta),4)
    sin_beta = round(math.sin(beta), 4)
    
    x = [[cos_alpha,-sin_alpha,        0,         0],
        [ sin_alpha, cos_alpha,        0,         0],
        [         0,         0, cos_beta, -sin_beta],
        [         0,         0, sin_beta,  cos_beta]
        ]
    
    length=len(dat.columns)
    cnt = length//4
    if  length % 4>0:
        cnt+=1

    i=0
    j=4
    k=0
    
    while k<cnt:
        if j<length:
            dat4 = dat.iloc[:,i:j]
            
        else:
            dat4 = dat.iloc[:,-4:]
    
            i=length-4
            j=length
        
        dat4 = dat4.values.tolist()
    
        prod = np.dot(dat4,x)
        
        dat.iloc[:,i:j] = prod
        
        i=j
        j+=4
        
        k+=1
        
    return dat

###########################################################################


data = pd.read_csv('Test/cryo3.csv')

ndata = data.iloc[:,:-1].select_dtypes(include=np.number)
print(ndata)

dat = ndata.apply(lambda x: 5*(x - x.min()) / (x.max() - x.min()))
print(dat)

length=len(dat.columns)
cnt = length//4
if  length % 4>0:
    cnt+=1

i=0
j=4
k=0

while k<cnt:
    if j<length:
        dat4 = dat.iloc[:,i:j]
        i=j
        j+=4
    else:
        dat4 = dat.iloc[:,-4:]
        
    rot = pd.DataFrame(columns=[0,1,2,3])
    
    dat4 = dat4.values.tolist()
    orgdata = pd.DataFrame(dat4)
   
    for a in range( 360):
        rad   = math.radians(a)
        cos= round(math.cos(rad),4)
        sin = round(math.sin(rad), 4)
        
        x = [[cos, -sin,   0,    0],
            [ sin,  cos,   0,    0],
            [   0,    0, cos, -sin],
            [   0,    0, sin,  cos]
            ]
        
        prod = np.dot(dat4,x)
        
        rotdata = pd.DataFrame(prod)
        
        osr = orgdata.subtract(rotdata)
        
        rtf = osr.var()
        
        rot.loc[len(rot.index)]=[round(rtf[0],5),round(rtf[1],5),round(rtf[2],5),round(rtf[3],5)]
        
    print()
    print("Attribute set ",k+1)
    print()

    print(rot)
    
    fig = plt.figure(figsize=(10, 5))
    
    plt.plot(rot[0],label='1')
    plt.plot(rot[1],label='2')   
    
    plt.ylabel('Variance')
    plt.xlabel('alpha')
    plt.legend(loc='upper left')
    #plt.show()  

    fig = plt.figure(figsize=(10, 5))
    
    plt.plot(rot[2],label='3')
    plt.plot(rot[3],label='4')
    
    plt.ylabel('Variance')
    plt.xlabel('beta')
    plt.legend(loc='upper left')
    #plt.show()
    
    k+=1

alpha = math.radians(int(input("Enter angle alpha : ")))

beta = math.radians(int(input("Enter angle beta : ")))

dat = ndata.apply(lambda x: 5*(x - x.min()) / (x.max() - x.min()))

dat = fdrt(dat, alpha, beta)

dat = dat.round(decimals=1)

print()

cols = dat.columns

dat['y'] = data.iloc[:,-1]

print("Data after rotation",'\n')

print('\n',dat)

x_train = dat

x_t = pd.read_csv('Test/cryo_test.csv')

x_test = x_t.iloc[:1,:-1].apply(lambda x: 5*(x - x.min()) / (x.max() - x.min()))
y_test = x_t.iloc[:1,-1]

x_test = fdrt(x_test, alpha, beta)

x_test = x_test.round(decimals=1)

dic = x_train['y'].value_counts()

UserorgData={}
print("**************************************************************\n")
for i in cols:
    UserorgData[i]=float(input("Enter "+i+" : "))
print("***************************************************************\n")

orgdf=pd.DataFrame(UserorgData,index=[0])
print(pd.DataFrame(orgdf))

for i in range(len(cols)):
    mx = x_train.iloc[:,i].max()
    mn = x_train.iloc[:,i].min()

    orgdf.iloc[0,i] = 5 * (orgdf.iloc[0,i] - mn) / (mx - mn)

cnt = length//4
if  length % 4>0:
    cnt+=1

i=0
j=4
k=0

while k<cnt:
    if j<length:
        dat4 = orgdf.iloc[:,i:j]  
    else:
        dat4 = orgdf.iloc[:,-4:]
        i=length-4
        j=length
    
    dat4 = dat4.values.tolist()

    dat4 = pd.DataFrame(dat4, index=[0])
    
    prod = fdrt(dat4,alpha,beta) #Roating data about a given angle
    orgdf.iloc[:,i:j] = prod

    i=j
    j+=4
      
    k+=1
    
UserROtdat = orgdf.round(decimals=1)
print("*******************Rotated input DATA*******************\n",UserROtdat)

x_test = UserROtdat.round(decimals=1)

dic = x_train['y'].value_counts()

pc = {}
pxc = {}
    
for key in sorted(dic.keys()):
    
    d = {}
    d['x'] = dic[key]
    d['y'] = len(x_train)

    pc[key] = d
    tr = x_train.loc[ x_train['y'] == key]

    col = []
    for j in range(length):
        temp = tr.loc[ tr.iloc[:,j] == x_test.iloc[0,j] ]
        d1 = {}
        d1['x'] = len(temp)
        d1['y'] = dic[key]
        col.append(d1)

    pxc[key] = col

cls_ratio = {}

for key in sorted(dic.keys()):
    cls_ratio[key] = rss(pc[key])

attr_ratio = {}

for key in sorted(dic.keys()):
    col = pxc[key]

    col_lst = []
    for c in col:
        col_lst.append(rss(c))

    attr_ratio[key] = col_lst

final = {}

for key in sorted(dic.keys()):
    lst = attr_ratio[key]

    mul = 1
    for j in range(length):
        mul *= lst[j]

    final[key] = cls_ratio[key] * mul

y_pred = []

d = {}
for key in dic.keys():
    d[key] = final[key]
key = max(d, key=d.get)  
y_pred = key

print("Predicted Output : ",y_pred)