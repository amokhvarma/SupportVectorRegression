#!/usr/bin/env python
# coding: utf-8

# In[1701]:


import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import cvxopt
import pandas as pd
from cvxopt import matrix,solvers
import math
import matplotlib.pyplot as plt


# In[1702]:


# Loading Input
df = pd.read_csv('BostonHousing.csv',sep=',',header=None)
X = df.values

#Removing the First row which contains name of features

X = X[1:]
X = X.astype(float)

# Changing 0-1 binary classification to -1-1 classification so that similarity matrix
# is actually indicative of similarity, otherwise (0,0) will give 0

for i in range(X.shape[0]):
    if(X[i][3]==0):
        X[i][3]=-1

y = X[:,-1]
X = X[:,:-1]
#Normalisation
for i in range(X.shape[1]):
    X[:,i] = (X[:,i]-np.mean(X[:,i]))/(np.std(X[:,i])+1e-7)

mean_y = np.mean(y)
std_y = np.std(y)
y = (y-mean_y)/(std_y)


# In[1703]:


# Calculate Mean Square Error
def pred_loss(y_pred,y):
    n=y.shape[0]
    c = 1/n * sum(abs(y_pred-y))
    return c

# Calculate R-2 Score 
def pred_score(y_pred,y):
    m = np.mean(y)
    a = sum((y_pred-m)**2)
    b = sum((y-m)**2)
    return math.sqrt(a/b)


# In[1704]:


# K-cross validation for Sklearn. This calculates the Score 
# and Mean Square Error and Prints it as well

def kcross_sklearn(X,y,k=5):

# groups is a dictionary which contains the data
# of each of k pieces mapped to index. 
# outputs contains the respective outputs
    
    
    score=[]
    loss=[]
    i=0
    n=X.shape[0]
    size=n//k
    index = 0
    groups = {}
    outputs = {}
    
    for i in range(k):
        if(index+size>= n):
            groups[i] = X[index:n]
            outputs[i] = y[index:n]
        else:
            groups[i] = X[index:index+size]
            outputs[i] = y[index:index+size]
        index=index+size
        
    
    hold_out = 0
# At each stage, one of the k pieces is held out and the model
# is trained on the rest. Then it is tested on the held out data
    
    
    for hold_out in range(k):
        inp=np.array([])
        out = np.array([])
        test_data = groups[hold_out]
        test_data_output = outputs[hold_out]
        
        for i in range(k):
            if(i==hold_out):
                continue
            else:
               
                if(inp.shape[0]==0):
                    inp = groups[i]
                    out=np.append(out,outputs[i])
                else:
                    inp = np.append(inp,groups[i],axis=0)
                    out = np.append(out,outputs[i])
        
        
#SCI-KIT IMPLEMENTATION :-

        svr_lin = SVR(kernel='linear',C=0.1, gamma='auto',epsilon = 0.025)
        svr_lin.fit(inp,out)
        score.append(pred_score(svr_lin.predict(test_data),test_data_output))  #test_data
        
        loss.append(pred_loss(svr_lin.predict(test_data),test_data_output))
        
    print(np.around(score,decimals=4),sum(score)/k)
    print(np.around(loss,decimals=4),sum(loss)/k)
        


# In[1705]:


kcross_sklearn(X,y)


# In[1706]:


def kcross_MySVR(X,y,k=5):
    score=[]
    loss=[]
    i=0
    n=X.shape[0]
    size=n//k
    index = 0
    groups = {}
    outputs = {}
    
    for i in range(k):
        if(index+size>= n):
            groups[i] = X[index:n]
            outputs[i] = y[index:n]
        else:
            groups[i] = X[index:index+size]
            outputs[i] = y[index:index+size]
        index=index+size
        
    
    hold_out = 0
    for hold_out in range(k):
        inp=np.array([])
        out = np.array([])
        test_data = groups[hold_out]
        test_data_output = outputs[hold_out]
        
        for i in range(k):
            if(i==hold_out):
                continue
            else:
               
                if(inp.shape[0]==0):
                    inp = groups[i]
                    out=np.append(out,outputs[i])
                else:
                    inp = np.append(inp,groups[i],axis=0)
                    out = np.append(out,outputs[i])
        
        
# OWN IMPLEMENTATION :-

        svr_lin = SupportVectorRegression(inp,out,'linear', 0.00035,0.025)
        svr_lin.fit()
        score.append(pred_score(svr_lin.pred(test_data),test_data_output))
        loss.append(pred_loss(svr_lin.pred(test_data),test_data_output))
        
    print(np.around(score,decimals=4),sum(score)/k)
    print(np.around(loss,decimals=4),sum(loss)/k)


# In[1707]:


#Own Implementation

class SupportVectorRegression:
    #epsilon = 0.1
    gamma = 0.1
    #c = 0.00035
    bias=0
    #kernel = 'linear'
    
# Different Kernel functions can be
# specified at the time of initialisation

    def kern(self,x1,x2):
        
        if(self.kernel=='linear'):
            #print(1)
            return np.dot(np.transpose(x1),x2)
        if(self.kernel == 'poly'):
            return (np.dot(np.transpose(x1),x2)+1)**2
        if(self.kernel == 'rbf'):
            #print(np.exp(-self.gamma*np.dot(np.transpose(x1-x2),(x1-x2))))
            
            return np.exp(-self.gamma*np.dot(np.transpose(x1-x2),(x1-x2)))
        else:
            print("Wrong Kernel")
            return 0
    
    
    def __init__(self,X,y,kernel = 'rbf',c=0.00035,epsilon = 0.25):
        self.X = X
        self.y = y
        self.epsilon = epsilon
        self.kernel = kernel
        self.c = c
        self.mean = np.mean(y)
        self.std = np.std(y)
        self.n = X.shape[0]
        
        
    
    def fit(self):
        n=self.n
        
        P = np.zeros([2*n,2*n])
# The vector to be optimised is a [2*n,1] vector in which the 
# first n elements are alpha and next n are alpha*
# The formulation can be seen from the report
        
        for i in range(2*n):
            for j in range(2*n):
                if((i<n and j<n)):
                    P[i][j] = self.kern(X[i],X[j])
                if(i>=n and j<n):
                    P[i][j] = -1*self.kern(X[i-n],X[j])
                if(i<n and j>=n):
                    P[i][j] = -1*self.kern(X[i],X[j-n])
                else:
                    P[i][j] = self.kern(X[i-n],X[j-n])
                    
        
        
        
        
        q = np.zeros([2*n,1])
        for i in range(2*n):
            if(i<n):
                q[i] = self.epsilon-self.y[i]
            else:
                q[i] = self.epsilon + self.y[i-n]
        
        
        G = np.zeros([4*n,2*n])
        G[0:(2*n),:] = np.diag([-1]*(2*n))
        G[2*n:,:] = np.diag([1]*(2*n))
        
        h = np.zeros([4*n,1])
        h[0:2*n] = 0
        h[2*n:] = self.c
        
        A = np.zeros([1,2*n])
        A[0,0:n] = 1
        A[0,n:]=-1
                
        b = 0
                
        P_matrix = matrix(P,tc = 'd')
        q_matrix = matrix(q,tc = 'd')
        G_matrix = matrix(G,tc = 'd')
        h_matrix = matrix(h,tc = 'd')
        A_matrix = matrix(A,tc = 'd')
        b_matrix = matrix(b,tc = 'd')
        
        
        
        sol = solvers.qp(P_matrix,q_matrix,G_matrix,h_matrix,A_matrix,b_matrix)
        self.alphas = sol['x']
        #print(sol['status'])
        #print(self.alphas)
         
# Bias is calculated by using either the inequality or by 
# averaging over the errors with b=0.
    
        temp = []
        
        for i in range(n):
            if(self.alphas[i] > 0 and self.alphas[i]<self.c):
                temp.append(-(self.mult(self.X[i])+self.y[i]-self.epsilon))
        
        self.bias = sum(temp)/len(temp)
        return 
    
#M = -1*math.inf
#       m = math.inf
        
#         for i in range(n):
#             if(self.alphas[i]<self.c or self.alphas[n+i]>0):
#                 if(M<(-1*self.epsilon + self.y[i] - self.mult(X[i]))):
#                     M = (-1*self.epsilon + self.y[i] - self.mult(X[i]))
                    
#             if(self.alphas[i+n]<self.c or self.alphas[i]>0):
#                 if( m > (-1*self.epsilon + self.y[i] - self.mult(X[i]))) :
#                     m = (-1*self.epsilon + self.y[i] - self.mult(X[i]))
                    
#             if((self.alphas[i]<self.c or self.alphas[n+i]>0) and (self.alphas[i+n]<self.c or self.alphas[i]>0) ):
#                 self.bias = (-1*self.epsilon + self.y[i] - self.mult(X[i]))
                
#         print(M,m)
                
       # if(self.bias == 0):
        #    self.bias = (M+m)/2
            
        #print(self.X[0:2],self.y[0:5])
        
        #print(std_y*std_y*pred_loss ((self.pred(X[300:])),y[300:]))
        
        #print(self.mult(X[100])+self.bias,self.y[100])
        
        
# This multiplies out weight matrix with a 
# vector X . 

    def mult(self,X):
        n = self.n
        alpha = np.zeros([n])
        
        for i in range(n):
            alpha[i] = self.alphas[i] - self.alphas[n+i] 
        
        temp = np.zeros([n])
        
        for i in range(n):
            #print(X,self.X[i],self.kern(X,self.X[i]))
            temp[i] = self.kern(X,self.X[i])
        #print(temp[0:5],alpha[0:5])
        #print(temp@alpha)
        
        
        return temp@alpha

#Prediction
   
    def pred(self,X):
        n = X.shape[0]
        res = np.zeros([n])
        for i in range(n):
            res[i] = self.mult(X[i]) + self.bias
            
        return res


# In[1708]:


kcross_MySVR(X,y)

