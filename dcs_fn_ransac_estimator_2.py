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
    
    This class allows various classifiers
    
"""
import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from copy import deepcopy
from sklearn.metrics import precision_recall_curve, average_precision_score

class Queue:
    
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def enqueue(self, item):
        self.items.insert(0,item)

    def dequeue(self):
        return self.items.pop()

    def size(self):
        return len(self.items)

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

class SVC_(object):

    def __init__(self):
        
        self.estimator = SVC(gamma = 1.0, C = 100.0, class_weight = 'balanced', probability=True)
    
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
    
    def __init__(self, gamma = 'auto', C = 1, class_weight = 'balanced', estimator=None, kernel = 'svm'):
        
        self.id = 1
                
        self.results = []

        self.Xtrain_init = None
            
        self.match_thres = 0.2 # error tolerance
    
        self.learning_constant = 0.05
    
        self.n_iter = 5 # Maximum #attempts
        
        if estimator == None :
                
            if kernel == 'svm' :
                
                self.gamma = gamma
        
                self.C = C
        
                self.class_weight = class_weight
            
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
        
        self.result_queue = Queue()
            
                
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
        
    def sample_init(self):
        
        self.train_size = None
        
        self.num_samples = None
        
        self.Xtrain_init = None
        
        self.train_size_init = None
        
        self.idx_sample = None
        
        self.Xsample = None
        
        self.Ysample = None  
        
        self.idx_all = None
        
        self.idx_res = None
        
        self.Xres = None
        
        self.Yres = None
    
    def sample(self, Xtrain, Ytrain) :
        
        import time
        
        print('timeseed:',int(np.mod(time.time()*1000,1000)))
        
        np.random.seed(int(np.mod(time.time()*1000,1000)))
            
        if  type(self.Xtrain_init) != np.ndarray :
            
            self.Xtrain_init = Xtrain.copy()
        
            self.train_size_init = Xtrain.shape[0]
                    
        self.train_size = Xtrain.shape[0]
        
        self.num_samples = int( 0.5 * len(Xtrain) )
        
        self.idx_sample = np.arange(self.train_size)
         
        np.random.shuffle(self.idx_sample)
        
        self.idx_sample = self.idx_sample[:self.num_samples]
        
        #print(self.idx_sample[:10])
        
        self.Xsample, self.Ysample = Xtrain[self.idx_sample], Ytrain[self.idx_sample]
        
        self.idx_all = np.arange(self.train_size) 
        
        self.idx_res = np.setdiff1d(self.idx_all, self.idx_sample) 
        
        self.Xres, self.Yres = Xtrain[list(self.idx_res)], Ytrain[list(self.idx_res)]
        
            
    def fit_sub(self, Xtrain, Ytrain, Xelse):
        
        """
        Args : Xtrian :  training data 
               Ytrain :  labels
               Xelse  :  unlabled X, if not given then return input
        
        """
            
        self.Ysample = 1
        
        # first we cross-validate to find high clean labels
        
        for i in range(self.num_estimates) :  
            
            self.sample_init()
            
            if self.__cv__  == True :
                
                sampling_loop_count = 0

                NO_GOOD_SAMPLES_POSSIBLE = True
                                
                while hasattr(self, 'Ysample') :
                    
                    while len(np.unique(self.Ysample)) == 1 : # ensure drawn samples contains both classes
                        
                        print('Sampling...')
                        
                        self.sample(Xtrain, Ytrain)
                                                
                    print('label_propagation...')
                    # Find lablled data that matches this fit
                    
                    print(self.Xsample)
                    
                    
                    
                    self.estimator.fit(self.Xsample, self.Ysample)
                                
                    pred = self.estimator.predict(self.Xres)
                    
                    print('Yres : ', self.Yres)
                    
                    print('pred : ', pred)
                    
                    AP = average_precision_score(self.Yres, pred)
                    
                    # if enough mathces are found, declare it to be a good estimate, 
                    
                    # print('AP:')
                    # print(AP)
                    
                    sampling_loop_count += 1
                    
                    print('sampling loop count :', sampling_loop_count)
                    
                    if sampling_loop_count > 5:
                        
                        import time
                                                
                        print('reseed...')
                        
                        np.random.seed( int(np.mod(time.time()*100,1000)))
                        
                        sampling_loop_count = 1
                        
                        if NO_GOOD_SAMPLES_POSSIBLE :
                            
                            break
                        
                        NO_GOOD_SAMPLES_POSSIBLE = True 
                        
                        
                    
                    if AP >= self.match_thres  :
                        
                        print('in loop')
                                                                
                        if AP > self.best_AP :
                            
                            self.best_AP = AP
                            
                            self.best_estimator = deepcopy(self.estimator)
                        
                            print('Best average precision :',self.best_AP)
                        
                        self.estimators.append( deepcopy(self.estimator)) # deep copy
                            
                        break

                        
            else : 
                
                while len(np.unique(self.Ysample)) == 1 : # ensure drawn samples contains both classes
                        
                        print('Sampling...')
                        
                        self.sample(Xtrain, Ytrain)
                        
                self.estimator.fit(self.Xsample, self.Ysample)
                
                self.estimators.append( deepcopy(self.estimator)) # deep copy
                
            # print('estimators',self.estimators)
                    
                    
        for estimator in self.estimators :  
            
                self.estimator = estimator
                    
                print(' Unitilizing Unlabelled Data...')
                
                if len(Xelse) == 0 :  
                    
                    # print('Unlebeled data does not exist.')
                    
                    return (self.best_AP , self.estimator )
                
                Yelse = self.estimator.predict(Xelse)
                            
                if hasattr(self.estimator, "decision_function"):
                    
                    confidence = self.estimator.decision_function(Xelse)
                                        
                else:  # Note: predict_proba function in libsvm requires cross-validation, leading to
                       #  high computational cost but low return. Consider prioritize decision_function. 
                       #  predict_proba implements Platt Scaling (sigmoid scaling), and the output is between [0,1]     
                    confidence = self.estimator.predict_proba(Xelse).T
                    
                                                            
                down, up = np.percentile(confidence, [10, 90])
                
#                import matplotlib.pyplot as plt
#                plt.figure()
#                plt.hist(confidence)
                                                
                self.Xsample, self.Ysample = Xtrain[self.idx_sample], Ytrain[self.idx_sample]
                
                self.Xres, self.Yres = Xtrain[self.idx_res], Ytrain[self.idx_res]
                
                # top 10 percentile is considered confident classification.
                X_inliers, Y_inliers = Xelse[  confidence >= up | confidence <= down], Yelse[ confidence >= up | confidence <= down]
                    
                # print(X_inliers)                
                # expande the dataset with the new lablled data
                                
                Xtrain, Ytrain = np.concatenate((Xtrain,X_inliers)), np.concatenate((Ytrain, Y_inliers))
                
                self.final_sample_sets.append((Xtrain, Ytrain))
                
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
        
            i = np.argwhere(x0==x) # a numpy array indices of x 
                        
            if len(np.unique(y0[i])) == 1: # all concensus x share the same prediciton label
                                
                y_concensus.append(y0[i][0])
            
            else : # remove the concensus x labels disagree with each other
                                
                x_concensus.delete(x)
        
        y_concensus = np.asarray(y_concensus)
                
        assert len(y_concensus) > 0, 'no concensus!{:d}'.format(len(y_concensus))
        
        self.estimator.fit(x_concensus, y_concensus)
        
        self.best_estimator = self.estimator
        
        pred = self.predict(self.Xres)
        
        self.best_AP  = average_precision_score(pred, self.Yres)
        
        return (self.best_AP , self.estimator )
                
    def fit(self, X, Y) :   
        
        Xtrain = X[np.argwhere(Y != -1)].squeeze()
        
        Xelse = X[np.argwhere(Y == -1)].squeeze()
        
        Ytrain = Y[np.argwhere(Y != -1)].squeeze()
                
        for i in range(self.n_iter) :
                                    
            result = self.fit_sub(Xtrain, Ytrain, Xelse)

            self.result_queue.enqueue(result)
                             
        self.results = [ self.result_queue.dequeue() for i in range(self.result_queue.size())]

        best_aps, best_estimators = list(zip(*self.results))
        
        self.best_estimator = best_estimators[ np.argmax(best_aps) ]
        
        self.best_AP = np.max(best_aps)  

             

    
    
