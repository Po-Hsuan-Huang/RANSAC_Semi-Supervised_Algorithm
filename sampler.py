#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 14:35:03 2019

@author: pohsuanh

RANSAC functions for 
"""
import numpy as np
from copy import deepcopy
from sklearn.metrics import average_precision_score
import RANSAC.dcs_fn_ransac_class as dcs_fn_ransac_class


class ransac_processor(dcs_fn_ransac_class.RANSAC_SVC):
    
    def __init__(self) :
        
        super(ransac_processor, self).__init__()
        
    def sample(self, Xtrain, Ytrian) :
        
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
        
        self.results =[]
        
        
    def fit(self, Xtrain, Ytrain, Xelse):
        
        Ysample = 1
            
        while len(np.unique(Ysample)) == 1 : # ensure drawn samples contains both classes
        
            self.__sample__(Xtrain, Ytrain)
            
        n = 0
        
        while True :
            
            # Find lablled data that matches this fit
            self.estimator.fit(self.Xsample, self.Ysample)
                        
            pred = self.estimator.predict(self.Xres)
            
            AP = average_precision_score(self.Yres, pred)
            
            # if enough mathces are found, declare it to be a good estimate, 
            # refit the estimator to the expanded set.
            
            if AP >= self.match_thres + n * self.learning_constant :
                
                n += 1
                                
                if AP > self.best_AP :
                    
                    self.best_AP = AP
                
                    print('Best average precision :',self.best_AP)
                
                    self.best_estimator = deepcopy(self.estimator) # deep copy
                
                # Rest of the labelled data
                
                confidence = self.estimator.decision_function(self.Xres)
                
                self.idx_sample = np.union1d(self.idx_sample ,self.idx_res[ (confidence >= 0.9) | (confidence <= -0.9) ])
                
                self.idx_res = np.setdiff1d( self.idx_res ,self.idx_res[ (confidence >= 0.9) | (confidence <= -0.9) ] )
                
                self.Xsample, self.Ysample = Xtrain[self.idx_sample], Ytrain[self.idx_sample]
                
                self.Xres, self.Yres = Xtrain[self.idx_res], Ytrain[self.idx_res]
    
#                print('training %',(train_size_init - len(group1_idx_res))/ train_size_init)
                
                if(self.train_size_init - len(self.idx_res))/ self.train_size_init == 1.0 : # training finished.
                    
                    break
                
                if  Xelse.size != 0 and (self.train_size_init - len(self.idx_res))/ self.train_size_init > 0.0 :
                    # Unlabelled Data
                    
                    Yelse = self.estimator.predict(Xelse)
                    
                    confidence = self.estimator.decision_function(Xelse) # calculate confidence score of the unlablled data
                    
                    X_inliers, Y_inliers = Xelse[  np.abs(confidence) >= 0.9 ], Yelse[   np.abs(confidence) >= 0.9 ]
                    
                    X_outliers, Y_outliers = Xelse[ np.abs(confidence) < 0.1  ], Yelse[ np.abs(confidence) < 0.1 ]
                        
                    # expande the dataset with the new lablled data
                    
                    train_size = Xtrain.shape[0]
                    
                    Xtrain, Ytrain = np.concatenate((Xtrain,X_inliers)), np.concatenate((Ytrain, Y_inliers))
                    
#                    Xtrain, Ytrain = np.concatenate((Xtrain,X_outliers)), np.concatenate((Ytrain, Y_outliers))
                    
                    expanded_size = Xtrain.shape[0]
                    
                    self.idx_sample = np.union1d(self.idx_sample, np.arange(train_size, expanded_size)) 
                    
                    self.Xsample, self.Ysample = Xtrain[self.idx_sample], Ytrain[self.idx_sample]
                
            else :     
                
                break
            
            return (self.best_estimator, self.best_AP)
            
