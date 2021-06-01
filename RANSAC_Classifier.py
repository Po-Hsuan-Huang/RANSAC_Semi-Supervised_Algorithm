#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 20:11:40 2019

@author: pohsuanh
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 10:18:08 2019

@author: pohsuanh

(Originally dcs_fn_ransac_svc_class.py)

Implement RANSAC that supports classification estimator
    
"""

import numpy as np
from sklearn.svm import SVC
import pathos.multiprocessing as mp
from multiprocessing import Process, Queue
from copy import deepcopy
from sklearn.metrics import precision_recall_curve, average_precision_score
import time

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

class SVC_(object):

    def __init__(self):
        
        self.estimator = SVC(gamma = 1.0, C = 100.0, class_weight = 'balanced',probability=True)
    
    def fit(self, X, Y):
        
        Xtrain = X[np.argwhere(Y != -1)].squeeze()
        
        Xelse = X[np.argwhere(Y == -1)].squeeze()
        
        Ytrain = Y[np.argwhere(Y != -1)].squeeze()
        
        return self.estimator.fit(Xtrain,Ytrain)
    
    def predict(self, X):
        
        self.pred = self.estimator.predict(X)
        
        return self.pred
    
    def predict_prob(self, X):
        
        self.y_score =self.estimator.predict_proba(X)[:,0]
        self.y_score =self.estimator.decision_function(X)
                
        return self.y_score
    
        
    def fit_predict(self, Xtrain,Ytrain, Xval):
        
        self.pred = self.estimator.fit(Xtrain,Ytrain).predict(Xval)
        
        return self.pred
        
    def score(self, Yval):
        
        return average_precision_score(Yval, self.pred)
    
#%% Global Matching : Random Sample and Concensus (RANSAC)

class RANSAC_Classifier(object):
    
    def __init__(self, estimator = 'svm', error_tolerance = 0.2, max_attempts =100 ):
        
        self.id = 1
                
        self.results = []
        
        self.result_queue = Queue()
        
        self.Xtrain_init = None
            
        self.match_thres = error_tolerance
    
        self.learning_constant = 0.05
    
        self.n_iter = max_attempts
        
        self.estimator = estimator
        
        if estimator =='svm' :
            
            self.gamma = 1.0
            
            self.C = 100
            
            self.class_weight = 'balanced'
            
            self.estimator = SVC( C = self.C, gamma = self.gamma,  class_weight = self.class_weight, probability=True)
        
        self.best_estimator = None
        
        self.best_AP = 0.
        
    def fit_paralall(self, X, Y):
        
        """ X is the input data, Y is the class annotation. 
        Unlabelled data are annotated '-1'.
        """
        
        def _work(Xtrain, Ytrain, Xelse, result_queue): 
            
            result = ransac_processor().fit(Xtrain, Ytrain, Xelse)
                  
            result_queue.put(result)
     
        Xtrain = X[np.argwhere(Y != -1)].squeeze()
        
        Xelse = X[np.argwhere(Y == -1)].squeeze()
        
        Ytrain = Y[np.argwhere(Y != -1)].squeeze().astype(int)
        
        print(Xtrain.shape, Ytrain.shape, Xelse.shape)
        
        jobs = []

        global start0

        start0 = time.time()

        i = 0   
        
        while i < (self.n_iter) :
            
            if  self.result_queue.qsize() < 2*mp.cpu_count() - 2 :
                                    
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
    
        y_score = self.predict_proba(Xval)[:][1]
        
        self.average_precision = average_precision_score(Yval, y_score)
        
#        print('Average precision-recall score: {0:0.2f}'.format(average_precision))
                
        self.precision, self.recall, self.thresholds = precision_recall_curve(Yval, y_score)
    
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
            
    def fit_sub(self, Xtrain, Ytrain, Xelse):
    
        self.Ysample = 1
            
        while len(np.unique(self.Ysample)) == 1 : # ensure drawn samples contains both classes
            
#            print('Sampling...')
            
            self.sample(Xtrain, Ytrain)
        
        n = 0
        
        while True :
            
#            print('label_propagation...')
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
                
                confidence =  normalize(confidence) + 0.5
                
                self.idx_sample = np.union1d(self.idx_sample ,self.idx_res[ (confidence >= 0.9) | (confidence <= -0.9) ])
                
                self.idx_res = np.setdiff1d( self.idx_res ,self.idx_res[ (confidence >= 0.9) | (confidence <= -0.9) ] )
                
                self.Xsample, self.Ysample = Xtrain[self.idx_sample], Ytrain[self.idx_sample]
                
                self.Xres, self.Yres = Xtrain[self.idx_res], Ytrain[self.idx_res]
        
        #                print('training %',(train_size_init - len(group1_idx_res))/ train_size_init)
                
                if(self.train_size_init - len(self.idx_res))/ self.train_size_init == 1.0 : # training finished.
                    
                    print('Propagation finished')
                    
                    break
                
                if  Xelse.size != 0 and (self.train_size_init - len(self.idx_res))/ self.train_size_init > 0.0 :
                    
                    print(' Utilizing Unlabelled Data...')
                    
                    Yelse = self.estimator.predict(Xelse)
                    
                    confidence = self.estimator.decision_function(Xelse) # calculate confidence score of the unlablled data
                    
                    confidence = (confidence - np.mean(confidence) )/np.std(confidence) + 0.5
                    
                    X_inliers, Y_inliers = Xelse[  np.abs(confidence) >= 0.9 ], Yelse[  np.abs(confidence) >= 0.9 ]
                    
                    X_outliers, Y_outliers = Xelse[ np.abs(confidence) < 0.1  ], Yelse[ np.abs(confidence) < 0.1 ]
                        
                    # expande the dataset with the new lablled data
                    
                    train_size = Xtrain.shape[0]
                    
                    Xtrain, Ytrain = np.concatenate((Xtrain,X_inliers)), np.concatenate((Ytrain, Y_inliers))
                    
        #                    Xtrain, Ytrain = np.concatenate((Xtrain,X_outliers)), np.concatenate((Ytrain, Y_outliers))
                    
                    expanded_size = Xtrain.shape[0]
                    
                    self.idx_sample = np.union1d(self.idx_sample, np.arange(train_size, expanded_size)) 
                    
                    self.Xsample, self.Ysample = Xtrain[self.idx_sample], Ytrain[self.idx_sample]
                        
            else :     
#                print(' cannnot find good estimator, best average precision : ', self.best_AP)
                break
                
        return (self.best_AP, self.best_estimator, )

    def fit(self, X, Y) :   
        
        Xtrain = X[np.argwhere(Y != -1)].squeeze()
        
        Xelse = X[np.argwhere(Y == -1)].squeeze()
        
        Ytrain = Y[np.argwhere(Y != -1)].squeeze()
                
        for i in range(self.n_iter) :
                                    
            result = self.fit_sub(Xtrain, Ytrain, Xelse)

            self.result_queue.put(result)
                             
        self.results = [ self.result_queue.get() for i in range(self.result_queue.qsize())]

        best_aps, best_estimators = list(zip(*self.results))
        
        self.best_estimator = best_estimators[ np.argmax(best_aps) ]
        
        self.best_AP = np.max(best_aps)                    

    
    
class ransac_processor(RANSAC_SVC):
    
    def __init__(self) :
        
        super(ransac_processor, self).__init__()
        
        print('Initializing...')
        
        assert self.best_AP == 0., 'not a deep copy.'
        
#        self.id += self.id 
#        
#        print('selfid :', self.id)
        
        if self.estimator == None :
            
            print('no estimator, continue ?') 
            
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
            
        while len(np.unique(self.Ysample)) == 1 : # ensure drawn samples contains both classes
            
#            print('Sampling...')
            
            self.sample(Xtrain, Ytrain)

        n = 0
        
        while True :
            
#            print('label_propagation...')
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
                
#                    print('Best average precision :',self.best_AP)
                
                    self.best_estimator = deepcopy(self.estimator) # deep copy
                
                # Rest of the labelled data
                
                confidence = self.estimator.decision_function(self.Xres)
                
#                confidence = (confidence - np.mean(confidence) )/np.std(confidence) + 0.5
                
                self.idx_sample = np.union1d(self.idx_sample ,self.idx_res[ (confidence >= 0.9) | (confidence <= -0.9) ])
                
                self.idx_res = np.setdiff1d( self.idx_res ,self.idx_res[ (confidence >= 0.9) | (confidence <= -0.9) ] )
                
                self.Xsample, self.Ysample = Xtrain[self.idx_sample], Ytrain[self.idx_sample]
                
                self.Xres, self.Yres = Xtrain[self.idx_res], Ytrain[self.idx_res]
    
#                print('training %',(train_size_init - len(group1_idx_res))/ train_size_init)
                
                if(self.train_size_init - len(self.idx_res))/ self.train_size_init == 1.0 : # training finished.
                    
                    print('Propagation finished')
                    
                    break
                
                if  Xelse.size != 0 and (self.train_size_init - len(self.idx_res))/ self.train_size_init > 0.0 :
                    
#                    print(' Unitilizing Unlabelled Data...')
                    
                    Yelse = self.estimator.predict(Xelse)
                    
                    confidence = self.estimator.decision_function(Xelse) # calculate confidence score of the unlablled data
                    
#                    confidence = (confidence - np.mean(confidence) )/np.std(confidence) + 0.5
                    
                    X_inliers, Y_inliers = Xelse[  np.abs(confidence) >= 0.9 ], Yelse[  np.abs(confidence) >= 0.9 ]
                    
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
                
        return (self.best_AP,self.best_estimator)