# SupportVectorRegression
This repository contains the code required to train different Support Vector Regression models to predict housing prices using boston housing dataset. 
## Index
1) [Requirements](#requirements)
2) [Introduction](#introduction)
3) [Implemetation details](#implementation-details)
4) [Running The Code](#usage)
5) [Results](#results)
### Requirements 
The code works on python >= 3.0  
To install all the required libraries run :

```
! pip install -r requirements.txt
```

### Introduction
Support Vector Regression is based on ***Support Vector Machines*** , first introduced by Vapnik and Cortes. SVMs are powerfull classification models which try to maximise the margin length between positive and negative labels. In the process of doing so, they only some of the elements of datasets are finally used for the predictions (For the optimal value of weight matrix), and these are called Support Vectors and hence the name. 

For a very basic idea of SVM, please visit http://web.mit.edu/6.034/wwwbob/svm-notes-long-08.pdf. 
SVR is a variation of SVM used for Regression and the math behind it can be found at https://alex.smola.org/papers/2004/SmoSch04.pdf

### Implementation details
The self implementation has been done using numpy and cvxopt. **CVXOPT** is a library used for convex optimisation All the details for solving of quadratic optimisation problems can be found at https://courses.csail.mit.edu/6.867/wiki/images/a/a7/Qp-cvxopt.pdf.

I have also implemented my own K-cross Validation test for the model to ensure a better selection of the required hyperparameters. The hyperparameters involved here are :-
```
epsilon = 0.1  // The value of soft margin
```
Another possibility of change is using the **kernel trick**.
```
self.kernel = linear // or poly or rbf (radial basis function)
```

### Usage
To run the code , use
```
! python3 SVR.py
```
However , please refer to the python notebook for better understanding.

### Results
Results can be found [here](Report.pdf)
