#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 13:42:52 2019

@author: pohsuanh

ransac_processor is the worker class for multiprocessing RANSAC when n_job >1



"""
import numpy as np
from multiprocessing import Process, Queue
from copy import deepcopy
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier



class SVC():

    def __init__(self, gamma = 'auto', C = 1, class_weight = 'balanced', estimator=None, kernel = 'svm'):
        
        self.id = 1
                
        self.results = []
        
        self.result_queue = Queue()
        
        self.Xtrain_init = None
            
        self.match_thres = 0.2
    
        self.learning_constant = 0.05
    
        self.n_iter = 20
            
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
        



class RansacProcessor(SVC):
    
    def __init__(self) :
        
        super(RansacProcessor, self).__init__()
        
        
        print('Initializing...')
        
        assert self.best_AP == 0., 'not a deep copy.'
        
#        self.id += self.id 
#        
#        print('selfid :', self.id)
        
        if self.estimator == None :
            
            k = input('no estimator, continue ?') 
            
            if k == 'n' :
                
                import os
                
                os.sys.exit()
#        else :
            
#            print('estimator :', self.estimator)
                    
    def sample(self, Xtrain, Ytrain) :
                
        if  type(self.Xtrain_init) != np.ndarray :
            
            self.Xtrain_init = Xtrain.copy()
    
            self.train_size_init = Xtrain.shape[0]
                    
        self.train_size = Xtrain.shape[0]
        
        self.num_samples = 4  # 10
        
        self.idx_sample = np.arange(self.train_size)
         
        np.random.shuffle(self.idx_sample)
        
        self.idx_sample = self.idx_sample[:self.num_samples]
        
        self.Xsample, self.Ysample = Xtrain[self.idx_sample], Ytrain[self.idx_sample]
        
        self.idx_all = np.arange(self.train_size) 
    
        self.idx_res = np.setdiff1d(self.idx_all, self.idx_sample) 
        
        self.Xres, self.Yres = Xtrain[list(self.idx_res)], Ytrain[list(self.idx_res)]
                
    def fit(self, Xtrain, Ytrain, Xelse):
        
        self.Ysample = 1
        
        next_step = True
        
        
        # first we cross-validate to find high clean labels
        
        for i in range(self.num_estimates) :  
            
            if self._cv_  == False :
            
                while next_step == False :
                    
                    while len(np.unique(self.Ysample)) == 1 : # ensure drawn samples contains both classes
                        
    #                    print('Sampling...')
                        
                        self.sample(Xtrain, Ytrain)
                            
    #                print('label_propagation...')
                    # Find lablled data that matches this fit
                    
                    self.estimator.fit(self.Xsample, self.Ysample)
                                
                    pred = self.estimator.predict(self.Xres)
                    
                    AP = average_precision_score(self.Yres, pred)
                    
                    # if enough mathces are found, declare it to be a good estimate, 
                    
                    if AP >= self.match_thres  :
                                                                        
                        if AP > self.best_AP :
                            
                            self.best_AP = AP
                        
    #                        print('Best average precision :',self.best_AP)
                        
                            self.estimators.append( deepcopy(self.estimator)) # deep copy
                            
                            next_step  = True
                    else :
                        
                        next_step = False
                        
            else : 
                
                while len(np.unique(self.Ysample)) == 1 : # ensure drawn samples contains both classes
                        
    #                    print('Sampling...')
                        
                        self.sample(Xtrain, Ytrain)
                        
                self.estimator.fit(self.Xsample, self.Ysample)
                
                self.estimators.append( deepcopy(self.estimator)) # deep copy
                
                    
                    
        for estimator in self.estimators :  
            
                self.estimator = estimator
                    
                print(' Unitilizing Unlabelled Data...')
                
                Yelse = self.estimator.predict(Xelse)
                
                confidence = self.estimator.decision_function(Xelse) # calculate confidence score of the unlablled data
                
                confidence = (confidence - np.mean(confidence) )/np.std(confidence)
                
                X_inliers, Y_inliers = Xelse[  np.abs(confidence) >= 0.9 ], Yelse[  np.abs(confidence) >= 0.9 ]
                
                X_outliers, Y_outliers = Xelse[ np.abs(confidence) < 0.1  ], Yelse[ np.abs(confidence) < 0.1 ]
                    
                # expande the dataset with the new lablled data
                
                train_size = Xtrain.shape[0]
                
                Xtrain, Ytrain = np.concatenate((Xtrain,X_inliers)), np.concatenate((Ytrain, Y_inliers))
                
                self.final_sample_sets.append((Xtrain, Ytrain))
                
#                Xtrain, Ytrain = np.concatenate((Xtrain,X_outliers)), np.concatenate((Ytrain, Y_outliers))
                
#                expanded_size = Xtrain.shape[0]
                
#                self.idx_sample = np.union1d(self.idx_sample, np.arange(train_size, expanded_size)) 
                
#                self.Xsample, self.Ysample = Xtrain[self.idx_sample], Ytrain[self.idx_sample]
                
        # keep the consensus data.\
        
        x0 = np.empty(1)
        y0 = np.empty(1)
        
        for x,y in self.final_sample_sets :
    
            x0 = np.concatenate((x0, x))
            
            y0 = np.concatenate((y0, y))
                        
        x1, x_ids, x_counts = np.unique(x0, return_index = True, return_counts = True)
        
        args = np.argwhere(x_counts > 5 )
        
        x_concensus = x1[args]
        
        y_concensus = []
        
        for x in x_concensus :
        
            i = np.argwhere(x0==x)
            
            if np.unique(y0[i]) == 1:
                
                y_concensus.append(y0[i][0])
            
            else :
                
                x_concensus.delete(x)
        
        y_concensus = np.asarray(y_concensus)
        
        self.best_estimator.fit(x_concensus, y_concensus)
        
        return self.estimator
