from pickle import TRUE
from flask import * 
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import random
import socket
import os
import time

import rss21

pid = 1

def rssrd(r, xy,client):

    f = {}
    g = {}
    R = {}
    esend = {}
    epk1 = {}

    for j in range(rss21.n):
        if j+1==pid:
            continue

        f[pid,(j+1)] = round(random.uniform(3*(10**7),4*(10**7)),6)
        g[pid,(j+1)] = round(random.uniform(3*(10**7),4*(10**7)),6)
        R[pid,(j+1)] = random.uniform(11*(10**18),19*(10**18))

    for j in range(rss21.n):
        if j+1==pid:
            continue

        prod = f[pid,(j+1)] * r
        esend[pid,(j+1)] = ( rss21.public_key.encrypt(prod) , f[pid,(j+1)] )

    for j in range(1,4):
        if j == pid:
            rss21.client_send(esend, client)
        else:
            print("Ready to receive")
            rss21.client_receive(pid, client)

    print("Received data")
    print(rss21.erecive)

    fj = {}

    for i in rss21.erecive.keys():
        epk1[i[0],i[1]]=( rss21.erecive[i][0] * g[i[1],i[0]] * xy + R[i[1],i[0]] , g[i[1],i[0]] )

        fj[i] = rss21.erecive[i][1]

    print("fj ",fj,"\n")
    print()

    for j in range(1,4):
        if j == pid:
            rss21.epk_send(epk1, client)
        else:
            rss21.epk_receive(pid, client)

    print("Received dat 01a")
    print(rss21.epkfinal)

    share1 = {}
    share2 = {}

    for i in rss21.epkfinal.keys():
        nr = rss21.private_key.decrypt(rss21.epkfinal[i][0])

        dr = rss21.epkfinal[i][1] * f[i]

        share1[i] = nr/dr

        share2[i] = - R[i] / ( fj[(i[1],i[0])] * g[i] )

        print('ok')

    t = round(random.uniform((-0.5),(0.5)),6)

    si = 0

    for i in share1.keys():
        si += share1[i] + share2[i] + ( r + t ) * xy

    rss21.s = []

    for j in range(1,4):
        if j == pid:
            rss21.si_send(si, client)
        else:
            rss21.si_receive(client)

    rss21.s.append(si)
    print(rss21.s)

    return sum(rss21.s)

def rss(d2,client):
    print("**********************102********************")
    print(type(d2))
    x, y = d2['x'], d2['y']

    alphax = round(random.uniform((-0.5),(0.5)),6)
    alphay = round(random.uniform((-0.5),(0.5)),6)

    x = x + alphax
    y = y + alphay

    r = round(random.uniform(3000,4000),6)

    sx = rssrd(r, x,client)

    sy = rssrd(r, y,client)

    return sx/sy

pat=os.getcwd() #to get working directory
print("CurrentPath : "+pat)
upfile=""

cols=[] #to store column names
TstCols=[]
Origional_data=[]
normalized_data=[] #to store normalized data
rotateded_data=[] #to store rotateded data about a given angle
alpha_graph=[] #This will hold file-names of alpha graph images
beta_graph=[] #This will hold file-names of beta graph images
Test_Data=[]
Normallized_Test_Data=[]
Origional_test_data=[]
rotateded_test_data=[]

#DATAFRAMES initialization
ndata= pd.DataFrame()
dat = pd.DataFrame()
dat4 = pd.DataFrame()

#Function to rotate data
def rotate_mult(dat,a,b):
    cos_a = round(math.cos(a),4)
    sin_a = round(math.sin(a), 4)

    cos_b = round(math.cos(b),4)
    sin_b = round(math.sin(b),4)

    x = [[cos_a,-sin_a,0,0],[ sin_a, cos_a,0,0],[0,0, cos_b, -sin_b],[0,0, sin_b,  cos_b]]
    
    prod=np.dot(dat,x) #Rotating data (Dot product of data with x)
    return prod

def clear():
    #Clear all lists
    angles.clear()
    alpha_graph.clear()
    beta_graph.clear()
    cols.clear()
    Origional_data.clear()
    normalized_data.clear()
    rotateded_data.clear()
    Test_Data.clear()
    TstCols.clear()
    Normallized_Test_Data.clear()
    Origional_test_data.clear()
    rotateded_test_data.clear()

    #delete older graphs
    dir = pat+'/static'
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))
    
app = Flask(__name__)   

@app.route('/')
def intro():
    return render_template("intro.html")  

@app.route('/getdata')
def upload():
    clear()
    os.chdir(pat) #Switch to working directory
    return render_template("file_upload_form.html")  

@app.route('/Showdata', methods = ['POST'])  
def success():  
    if request.method == 'POST':  
        f = request.files['file']  
        f.save(f.filename)
        upfile = f.filename

        data = pd.read_csv(upfile)
        for i in data.values.tolist():
            Origional_data.append(i)
        print ("*********Origional data******************")
        print(pd.DataFrame(Origional_data))
        #ndata = data.select_dtypes(include=np.number)
        
        ndata = data.iloc[:,:-1].select_dtypes(include=np.number)

        os.remove(upfile)
        print ("*********Origional Numeric data******************")
        for i in data.values.tolist():
            Origional_data.append(i)
        print(ndata)
        for col_name in ndata.columns: 
            cols.append(col_name)
        print(cols)

        #Normallized data

        dat = ndata.apply(lambda x: 5*(x - x.min()) / (x.max() - x.min()))
        print ("*********Normallized data******************")
        for row in dat.values.tolist():
            normalized_data.append(row)

        print(pd.DataFrame(normalized_data,columns=cols))
        length=len(dat.columns)
    
        cnt = length//4
        if  length % 4>0:
            cnt+=1
            
        print("length = ",length)
        print('count = ',cnt)
    
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

            for a in range(360):
                R0   = math.radians(a)
                
                prod = rotate_mult(dat4,R0,R0) #Roating data
                
                rotdata = pd.DataFrame(prod)
                osr = orgdata.subtract(rotdata)
                rtf = osr.var()
                rot.loc[len(rot.index)]=[round(rtf[0],5),round(rtf[1],5),round(rtf[2],5),round(rtf[3],5)]
   
            print("upfile : "+upfile)
            
            os.chdir(pat+"\static")

            i=0
            fig = plt.figure(figsize=(10,5))
            plt.plot(rot[0],label='1')
            plt.plot(rot[1],label='2')     
            plt.ylabel('Variance')
            plt.xlabel('alpha')
            plt.legend(loc='upper left')
            #plt.show() 
            plt.savefig(upfile+'_Alpha'+str(k)+'.png')
            alpha_graph.append(upfile+'_Alpha'+str(k)+'.png')
            
            fig = plt.figure(figsize=(10,5))
            plt.plot(rot[2],label='3')
            plt.plot(rot[3],label='4')  
            plt.ylabel('Variance')
            plt.xlabel('beta')
            plt.legend(loc='upper left')
            #plt.show()
            plt.savefig(upfile+'_Beta'+str(k)+'.png')
            beta_graph.append(upfile+'_Beta'+str(k)+'.png')

            k+=1
        print(len(normalized_data))   
        return render_template("OrgData.html",r=ndata.values.tolist(),cols=cols,dat=dat.values.tolist(),zip=zip,round=round,l=len(normalized_data))

@app.route('/rotate')  
def graph():
    print(alpha_graph)
    print(beta_graph)
    p=os.getcwd()
    print("Graph directory : ",p)
    return render_template("graphs.html",Ag=alpha_graph,Bg=beta_graph,zip=zip)

angles=[]
@app.route('/continue',methods = ['POST'])  
def enter_angle():
    #Rotate about a given angle
    alpha = math.radians(int(request.form['alpha']))
    angles.append(int(request.form['alpha']))
    beta = math.radians(int(request.form['beta']))
    angles.append(int(request.form['beta']))

    dat=pd.DataFrame(normalized_data,columns=cols)
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
            
        prod = rotate_mult(dat4,alpha,beta) #Roating data about a given angle
        dat.iloc[:,i:j] = prod

        i=j
        j+=4
                
        k+=1
    
    dat = dat.round(decimals=1)
    data=pd.DataFrame(Origional_data)
    dat['y'] = data.iloc[:,-1]
    cols.append('y')

    #Store rotated data in a list
    for row in dat.values.tolist():
        rotateded_data.append(row)

    print(pd.DataFrame(rotateded_data))
    return render_template("RotatedData.html",cols=cols,r=rotateded_data,round=round,l=len(rotateded_data),z=zip) 
    
@app.route('/Show-Test',methods = ['POST'])
def showTestData():
    if request.method == 'POST':  
        f = request.files['file']  
        f.save(f.filename)
        upfile = f.filename

        data = pd.read_csv(upfile)
        for i in data.values.tolist():
            Origional_test_data.append(i)
        print ("*********Origional Test data******************")
        print(pd.DataFrame(Origional_test_data))
        ndata = data.iloc[:,:-1].select_dtypes(include=np.number)

        os.remove(upfile)

        print ("*********Origional Test Numeric data******************")
        for i in ndata.values.tolist():
            Test_Data.append(i)
        print(ndata)

        for col_name in ndata.columns: 
            TstCols.append(col_name)
        print(TstCols)

        #Normallized data

        dat = ndata.apply(lambda x: 5*(x - x.min()) / (x.max() - x.min()))
        print ("*********Normallized Test data******************",dat)
        for row in dat.values.tolist():
            Normallized_Test_Data.append(row)
        return render_template("TstOrgNorm.html",r=Test_Data,cols=TstCols,dat=Normallized_Test_Data,zip=zip,round=round,l=len(Normallized_Test_Data),angles=angles)

@app.route('/rotateAng')  
def RotateAngle():
    #Rotate about a given angle
    alpha = math.radians(int(angles[0]))
    beta = math.radians(int(angles[1]))

    dat=pd.DataFrame(Normallized_Test_Data,columns=TstCols)

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
            
        prod = rotate_mult(dat4,alpha,beta) #Roating data about a given angle
        dat.iloc[:,i:j] = prod

        i=j
        j+=4
                
        k+=1
    
    dat = dat.round(decimals=1)
    data=pd.DataFrame(Origional_test_data)
    dat['y'] = data.iloc[:,-1]
    TstCols.append('y')

    #Store rotated data in a list
    for row in dat.values.tolist():
        rotateded_test_data.append(row)

    print(pd.DataFrame(rotateded_test_data))
    return render_template('loader.html')

@app.route('/Naive-bayes')
def nvb():
    dat=pd.DataFrame(Normallized_Test_Data,columns=TstCols[:len(TstCols)-1])
    length=len(dat.columns)

    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(('18.189.106.45', 12704))

    x_train=pd.DataFrame(rotateded_data,columns=cols)
    x_test=pd.DataFrame(rotateded_test_data,columns=TstCols).iloc[:2,:-1]
    y_test=pd.DataFrame(Origional_test_data).iloc[:2,-1]

    # x_test=pd.DataFrame(rotateded_test_data,columns=TstCols).iloc[:,:-1]
    # y_test=pd.DataFrame(Origional_test_data).iloc[:,-1]

    x_test = x_test.round(decimals=1)

    pc = {}
    pxc = {}

    dic = x_train['y'].value_counts()
    
    for key in sorted(dic.keys()):
            
        d = {}
        d['x'] = dic[key]
        d['y'] = len(x_train)

        pc[key] = d
        tr = x_train.loc[ x_train['y'] == key]
        row = []
        for r in range(len(x_test)):
            col = []
            for j in range(length):
                temp = tr.loc[ tr.iloc[:,j] == x_test.iloc[r,j] ]
                d1 = {}
                d1['x'] = len(temp)
                d1['y'] = dic[key]
                col.append(d1)
            row.append(col)
        pxc[key] = row

    cls_ratio = {}

    for key in sorted(dic.keys()):
        cls_ratio[key] = rss(pc[key],client)

    attr_ratio = {}

    for key in sorted(dic.keys()):
            d = pxc[key]
            row_lst = []
            
            for r in range(len(x_test)):
                row = d[r]
                col_lst = []
                for j in range(length):
                    col_lst.append(rss(row[j],client))
                row_lst.append(col_lst)
            attr_ratio[key] = row_lst

    final = {}

    for key in sorted(dic.keys()):
        lst = attr_ratio[key]
        row_lst = []
        for r in range(len(x_test)):
            row = lst[r]
            mul = 1
            for j in range(length):
                mul *= row[j]
            row_lst.append(cls_ratio[key] * mul)
        final[key] = row_lst

    y_pred = []

    for r in range(len(x_test)):
        d = {}
        for key in dic.keys():
            d[key] = final[key][r]
        key = max(d, key=d.get)  
        y_pred.append(key)
    accuracy = round(accuracy_score( y_pred, y_test)*100,2)
    print("Accuracy is :", accuracy)
    return render_template("RotatedTestData.html",cols=TstCols,r=rotateded_test_data,round=round,l=len(rotateded_test_data),z=zip,a=accuracy)

if __name__ == '__main__':  
    app.run(debug = True) 