#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 14:19:31 2019

@author: pohsuanh
"""

import sys
sys.path.insert(0,"/home/pohsuanh/Documents/Itti Lab/SEMISUPERVISED_LABELING/RANSAC_exp/RANSAC/")
from ransac_processor import RansacProcessor
import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import pathos.multiprocessing as mp
from multiprocessing import Process, Queue
from copy import deepcopy
from sklearn.metrics import precision_recall_curve, average_precision_score
import time

#%% Global Matching : Random Sample and Concensus (RANSAC)

class RANSAC_SVC(object):
    
    def __init__(self, gamma = 'auto', C = 1, class_weight = 'balanced', estimator=None, kernel = 'svm'):
        
        self.id = 1
                
        self.results = []
        
        self.result_queue = Queue()
        
        self.Xtrain_init = None
            
        self.match_thres = 0.2
    
        self.learning_constant = 0.05
    
        self.n_iter = 100
            
        self.gamma = gamma
        
        self.C =C
        
        self.class_weight = class_weight
        
        if estimator == None :
                
            if kernel == 'svc' :
            
                self.estimator = SVC( C = self.C, gamma = self.gamma,  class_weight = self.class_weight, probability=True)
                
            elif kernel == 'knn' :
                self.estimator = KNeighborsClassifier()
        else :
            
            self.estimator = estimator

        
        self.best_estimator = None
        
        self.best_AP = 0.
        
        self.final_sample_sets = [] 
        
        self.num_estimates = 10
        
        self.__cv__  = True
        
        self.estimators = []
            
                
    def predict(self, X):
        
        self.pred = self.best_estimator.predict(X)
        
        return self.pred
    
    def predict_class(self, X):
        
        return self.predict(X)
    
    def predict_proba(self,X):
        
        self.y_score = self.best_estimator.predict_proba(X)
        
        return self.y_score
    
    def score(self, Xval, Yval):
        
        return self.estimator.score(Xval, Yval)
    
    def eval_score(self, Xval, Yval):
        
        pred = self.predict_class(Xval)
        
        self.validation_average_precision = average_precision_score(Yval, pred)
        
#        print('validation_average_precision: ', average_precision_score(Yval, pred) )
    
        y_score = self.predict_score(Xval)
        
        self.average_precision = average_precision_score(Yval, y_score)
        
#        print('Average precision-recall score: {0:0.2f}'.format(average_precision))
                
        self.precision, self.recall, self.thresholds = precision_recall_curve(Yval, y_score)
    
    def sample(self, Xtrain, Ytrain) :
            
        if  type(self.Xtrain_init) != np.ndarray :
            
            self.Xtrain_init = Xtrain.copy()
        
            self.train_size_init = Xtrain.shape[0]
                    
        self.train_size = Xtrain.shape[0]
        
        self.num_samples = int( 0.5 * len(Xtrain) )
        
        self.idx_sample = np.arange(self.train_size)
         
        np.random.shuffle(self.idx_sample)
        
        self.idx_sample = self.idx_sample[:self.num_samples]
        
        self.Xsample, self.Ysample = Xtrain[self.idx_sample], Ytrain[self.idx_sample]
        
        self.idx_all = np.arange(self.train_size) 
        
        self.idx_res = np.setdiff1d(self.idx_all, self.idx_sample) 
        
        self.Xres, self.Yres = Xtrain[list(self.idx_res)], Ytrain[list(self.idx_res)]
            
    def fit(self, X, Y):
        
        """ X is the input data, Y is the class annotation. 
        Unlabelled data are annotated '-1'.
        """
        
        def _work(Xtrain, Ytrain, Xelse, result_queue): 
            
            result = RansacProcessor().fit(Xtrain, Ytrain, Xelse)
                  
            result_queue.put(result)
     
        Xtrain = X[np.argwhere(Y != -1)].squeeze()
        
        Xelse = X[np.argwhere(Y == -1)].squeeze()
        
        Ytrain = Y[np.argwhere(Y != -1)].squeeze().astype(int)
                
        jobs = []

        global start0

        start0 = time.time()

        i = 0   
        
        while i < (self.n_iter) :
            
            if  self.result_queue.qsize() < mp.cpu_count()-4 :
                                    
                p = Process(target = _work, args = (Xtrain, Ytrain, Xelse, self.result_queue))
                        
                jobs.append(p)
                
                p.start()
                
                print( 'ChildProcess...',i)
                
                i += 1 
            
            else : 
                
                time.sleep(1)
            
            
        for p in jobs:
            
            p.join()
           
            p.close()
            
        self.results = [ self.result_queue.get() for i in range(self.result_queue.qsize())]

        best_aps, best_estimators = list(zip(*self.results))
        
        self.best_estimator = best_estimators[ np.argmax(best_aps) ]
        
        self.best_AP = np.max(best_aps)     