#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 10:18:08 2019

@author: pohsuanh

Class Implementaion of the RANSAC-SVM.

(Originally dcs_fn_ransac_svc_module.py)


    '''
    RBF SVM parameters
    Intuitively, the gamma parameter defines how far the influence of a single 
    training example reaches, with low values meaning ‘far’ and high values meaning
     ‘close’. The gamma parameters can be seen as the inverse of the radius of 
     influence of samples selected by the model as support vectors.
    
    The C parameter trades off correct classification of training examples against 
    maximization of the decision function’s margin. For larger values of C, a 
    smaller margin will be accepted if the decision function is better at 
    classifying all training points correctly. A lower C will encourage a larger
    margin, therefore a simpler decision function, at the cost of training 
    accuracy.In other words``C`` behaves as a regularization parameter in the SVM.
    '''
    
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
        
        self.y_score =self.estimator.predict_proba(X)[:,1]
        self.y_score =self.estimator.decision_function(X)
                
        return self.y_score
    
        
    def fit_predict(self, Xtrain,Ytrain, Xval):
        
        self.pred = self.estimator.fit(Xtrain,Ytrain).predict(Xval)
        
        return self.pred
        
    def score(self, Yval):
        
        return average_precision_score(Yval, self.pred)
    
    
class iter_counter():

    def __init__(self, X_size):
        """
        If the resampling in the while loop doesn't yeild better seed training samples 5 times stright,
        the iteration terminates.
        
        Max iteration is empriricaly defined as sqarued root of sample set size, but it depends on actual data distribution.
        
        Stopping signal is returned if count reaches max iterations.
        """
        self.max_iter  = int(np.sqrt(X_size))
        self.count = 1
        self.stop = False
        
    def add_one(self):
        
        self.count += 1
        
        if self.count >= self.max_iter:
            
            self.stop = True
            
            print('Iteration stopped because no improvement in the last {:d} iterations'.format(self.max_iter))
    
    def get_stop_signal(self):
        
        return self.stop
            
    def reset(self):
        
        self.count =1
#%% Global Matching : Random Sample and Concensus (RANSAC)

class RANSAC_Classifier(object):
    
    def __init__(self, estimator = 'svm', error_tolerance = 0.2, max_attempts =100, class_weight= None):
        
        self.id = 1
                
        self.results = []
        
        self.result_queue = Queue()
        
        self.train_size_init = None
            
        self.match_thres = error_tolerance
    
        self.learning_constant = 0.05
    
        self.n_iter = max_attempts
                
        self.estimator = estimator
        
        self.class_weight = class_weight
        
        if estimator =='svm' :
            
            self.gamma = 1.0
            
            self.C = 100
            
            self.class_weight = 'balanced'
            
            self.estimator = SVC( C = self.C, gamma = self.gamma,  class_weight = self.class_weight, probability=True)
        
        self.best_estimator = None
        
        self.best_AP = 0.

        
        
    def fit_parallel(self, X, Y):
        
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
            
            if  self.result_queue.qsize() < 2*mp.cpu_count() -1 :
                                    
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
    
        y_score = self.predict_proba(Xval)[:,1]
        
        self.average_precision = average_precision_score(Yval, y_score)
        
#        print('Average precision-recall score: {0:0.2f}'.format(average_precision))
                
        self.precision, self.recall, self.thresholds = precision_recall_curve(Yval, y_score)
    
    def sample(self, Xtrain, Ytrain) :
            
        if not self.train_size_init : 
            
            self.Xtrain_init = Xtrain.copy()
        
            self.train_size_init = Xtrain.shape[0]
            
            print('train_size_init')
                    
        self.train_size = Xtrain.shape[0]
        
        self.num_samples = max(int(0.9*len(Ytrain)),10)
        
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
                        
#            pred = self.estimator.predict(self.Xres)
#            
#            AP = average_precision_score(self.Yres, pred)
            
            AP = self.estimator.score(self.Xres, self.Yres)
                        
#            print('AP:', AP)
            
            # if enough mathces are found, declare it to be a good estimate, 
            # refit the estimator to the expanded set.
            
            if AP >= self.match_thres + n * self.learning_constant :
                
                print(self.match_thres + n * self.learning_constant)
                
                n += 1
                                
                if AP > self.best_AP :
                    
                    self.best_AP = AP
                
                    print('Best average precision :',self.best_AP)
                
                    self.best_estimator = deepcopy(self.estimator) # deep copy
                
                # Rest of the labelled data
                
                confidence = self.estimator.decision_function(self.Xres)
                
                down, up = np.percentile(confidence, [10, 90])
                
                interval = max( max(abs(down), abs(up)),  0.1)
                
                if not np.sign(up) == 1 : print('Negative sign of up percentile.'); break
                    
                if not np.sign(down)== -1: print('Positive sign of down percentile.'); break
                
                idx_inlier_up = confidence >= interval 
                
                idx_inlier_down = confidence <= -interval
                
                idx_inlier = idx_inlier_down | idx_inlier_up
                        
                print(' ground truth validation phase')
                
                print('percentage of inliers_down {:f}'.format(sum(idx_inlier_down)/len(confidence)))

                print('percentage of inliers_up {:f}'.format(sum(idx_inlier_up)/len(confidence)))
                    
                print('percentage of inliers {:f}'.format(sum(idx_inlier)/len(confidence)))
                
                if len(self.Yres[idx_inlier]) == 0:
                    print('no new inlier found')
                    
                    break

                AP2 = self.estimator.score(self.Xres[idx_inlier], self.Yres[idx_inlier])
                
                if AP2 < AP :
                    
                    print('inliers disagrees with groundtruths.')
                    
                    break
                    
                self.idx_sample = np.union1d(self.idx_sample ,self.idx_res[idx_inlier])
                
                self.idx_res = np.setdiff1d( self.idx_res ,self.idx_res[ idx_inlier] )
                
                self.Xsample, self.Ysample = Xtrain[self.idx_sample], Ytrain[self.idx_sample]
                
                self.Xres, self.Yres = Xtrain[self.idx_res], Ytrain[self.idx_res]
                
                self.estimator.fit(self.Xsample, self.Ysample)
                
                print('training %', 100*(self.train_size_init - len(self.idx_res))/ self.train_size_init)
                
                if(self.train_size_init - len(self.idx_res))/ self.train_size_init == 1.0 : # training finished.
                    
                    print('Propagation finished')
                    
                    break
                
                if  Xelse.size != 0 and (self.train_size_init - len(self.idx_res))/ self.train_size_init > 0.0 :
                    
#                    print(' Utilizing Unlabelled Data...')
                    
                    self.counter.reset()

                    
                    Yelse = self.estimator.predict(Xelse)
                    
                    confidence = self.estimator.decision_function(Xelse) # calculate confidence score of the unlablled data
                    
                    down, up = np.percentile(confidence, [10, 90])
                    
                    interval = max( max(abs(down), abs(up)), 0.1)
                    
                    if not np.sign(up) == 1 : print('Negative sign of up percentile.'); break
                    
                    if not np.sign(down)== -1: print('Positive sign of down percentile.'); break
                    
                    print('up {:f} , down {:f}'.format(up,down))
                    
                    idx_inlier_up = confidence >= interval 
                
                    idx_inlier_down = confidence <= -interval
                
                    idx_inlier = idx_inlier_down | idx_inlier_up
                    
                    print('percentage of inliers_down {:f}'.format(sum(idx_inlier_down)/len(confidence)))

                    print('percentage of inliers_up {:f}'.format(sum(idx_inlier_up)/len(confidence)))
                    
                    print('percentage of inliers {:f}'.format(sum(idx_inlier)/len(confidence)))
                                        
                    X_inliers, Y_inliers = Xelse[ idx_inlier ], Yelse[  idx_inlier ]
                    
#                    X_outliers, Y_outliers = Xelse[ ~idx_inlier ], Yelse[ ~idx_inlier]
                        
                    # expande the dataset with the new lablled data
                    
                    Xtrain, Ytrain = np.concatenate((Xtrain,X_inliers)), np.concatenate((Ytrain, Y_inliers))
                    
        #                    Xtrain, Ytrain = np.concatenate((Xtrain,X_outliers)), np.concatenate((Ytrain, Y_outliers))
                    
                    expanded_size = Xtrain.shape[0]
                    
                    self.idx_sample = np.union1d(self.idx_sample, np.arange(self.train_size, expanded_size)) 
                    
                    self.Xsample, self.Ysample = Xtrain[self.idx_sample], Ytrain[self.idx_sample]
                        
            else :     
#                print(' cannnot find good estimator, best average precision : ', self.best_AP)
                self.counter.add_one()
                break
                
        return (self.best_AP, self.best_estimator, )

    def fit(self, X, Y) :   
        
        Xtrain = X[np.argwhere(Y != -1)].squeeze()
        
        Xelse = X[np.argwhere(Y == -1)].squeeze()
        
        Ytrain = Y[np.argwhere(Y != -1)].squeeze()
        
        self.counter = iter_counter(Xtrain.shape[0])

                
        for i in range(self.n_iter) : 
            
            if not self.counter.stop :
                
                print(self.counter.count)
                                    
                result = self.fit_sub(Xtrain, Ytrain, Xelse)
    
                self.result_queue.put(result)
            
            else :
                
                break
                             
        self.results = [ self.result_queue.get() for i in range(self.result_queue.qsize())]

        best_aps, best_estimators = list(zip(*self.results))
        
        self.best_estimator = best_estimators[ np.argmax(best_aps) ]
        
        self.best_AP = np.max(best_aps)                    

    
    
class ransac_processor(RANSAC_Classifier):
    
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
                
        if  not self.Xtrain_init :
            
            self.Xtrain_init = Xtrain.copy()
    
            self.train_size_init = Xtrain.shape[0]
                    
        self.train_size = Xtrain.shape[0]
        
        self.num_samples = max(int(0.9*len(Ytrain)),10)
        
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
                
                down, up = np.percentile(confidence, [10, 90])
                
                interval = max( max(abs(down), abs(up)), 0.1)
                
                idx_inlier_up = confidence >= interval 
                
                idx_inlier_down = confidence <= -interval
            
                idx_inlier = idx_inlier_down | idx_inlier_up
                                    
                self.idx_sample = np.union1d(self.idx_sample ,self.idx_res[ idx_inlier])
                
                self.idx_res = np.setdiff1d( self.idx_res ,self.idx_res[ idx_inlier ] )
                
                self.Xsample, self.Ysample = Xtrain[self.idx_sample], Ytrain[self.idx_sample]
                
                self.Xres, self.Yres = Xtrain[self.idx_res], Ytrain[self.idx_res]
    
                print('training %',(self.train_size_init - len(self.idx_res))/ self.train_size_init)
                
                if(self.train_size_init - len(self.idx_res))/ self.train_size_init == 1.0 : # training finished.
                    
                    print('Propagation finished')
                    
                    break
                
                if  Xelse.size != 0 and (self.train_size_init - len(self.idx_res))/ self.train_size_init > 0.0 :
                    
#                    print(' Unitilizing Unlabelled Data...')
                    
                    Yelse = self.estimator.predict(Xelse)
                    
                    confidence = self.estimator.decision_function(Xelse) # calculate confidence score of the unlablled data
                    
#                    confidence = (confidence - np.mean(confidence) )/np.std(confidence) + 0.5
                    
                    down, up = np.percentile(confidence, [10, 90])
                    
                    interval = max( max(abs(down), abs(up)), 0.1)
                
                    idx_inlier_up = confidence >= interval 
                
                    idx_inlier_down = confidence <= -interval
            
                    idx_inlier = idx_inlier_down | idx_inlier_up
                    
                    X_inliers, Y_inliers = Xelse[  idx_inlier ], Yelse[ idx_inlier ]
                    
                    X_outliers, Y_outliers = Xelse[ ~idx_inlier  ], Yelse[ ~idx_inlier ]
                        
                    # expande the dataset with the new lablled data
                                        
                    Xtrain, Ytrain = np.concatenate((Xtrain,X_inliers)), np.concatenate((Ytrain, Y_inliers))
                    
#                    Xtrain, Ytrain = np.concatenate((Xtrain,X_outliers)), np.concatenate((Ytrain, Y_outliers))
                    
                    expanded_size = Xtrain.shape[0]
                    
                    self.idx_sample = np.union1d(self.idx_sample, np.arange(self.train_size, expanded_size)) 
                    
                    self.Xsample, self.Ysample = Xtrain[self.idx_sample], Ytrain[self.idx_sample]
                        
            else :     
            
                break
                
        return (self.best_AP,self.best_estimator)