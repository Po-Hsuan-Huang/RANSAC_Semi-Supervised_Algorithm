#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 15:03:20 2019

@author: pohsuanh

Random Concensus Label Denoise Version 2

Improvement in features :
    
1. Iterative training of the MLP:
        The original version only fit MLP with one training set, and reinstantiate a new MLP 
        every time new labels are added. This version the MLP is recycled through the label selection process.

2. Max_count_pooling :
        The origianl version only take unanimous consensus as new labels. This version takes
        the absolute majority rule. Desition can be achived without unanimous agreement if  
        at least 50% of the estimators have conssensus. 
        
3. Supports multiple epochs :
        After a fulll cycle of label spreading, the model can be trained futher by warm-starting the 
        label spreading with the trained MLP.        

"""
import numpy as np
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from copy import deepcopy
from sklearn.metrics import precision_recall_curve, average_precision_score, accuracy_score
from sklearn.model_selection import train_test_split


def sample_gen( X, Y):
        
        rate = 0.9

        num_totoal = len(X)

        sample_ids = np.arange(num_totoal)
        
        np.random.shuffle(sample_ids)
        
        sample_ids = sample_ids[:int(num_totoal*rate)]     
        
        return X[sample_ids], Y[sample_ids]
    
def intersect_id(a,b): 
    
    a1_rows = a.view([('', a.dtype)] * a.shape[1])
    a2_rows = b.view([('', b.dtype)] * b.shape[1])
    sec, comm1, comm2 = np.intersect1d(a1_rows, a2_rows, return_indices=True)
    
    return comm1, comm2

def setdiff(a, b):
    """
    Return the unique values in ar1 that are not in ar2.
    
    """
    a1_rows = a.view([('', a.dtype)] * a.shape[1])
    a2_rows = b.view([('', b.dtype)] * b.shape[1])
    
    return np.setdiff1d(a1_rows, a2_rows).view(a.dtype).reshape(-1, a.shape[1])

def max_proba_pool(X, preds):
    """
    predict probability of each class for each sample, and max pooling the samples
    above probability threshold. Threshold should be learned through logit layer or
    softmax layer. However, sciki-learn doesn't allow building neural netwrok layers.
    Temperary we use fixed threshold.
    
    args : 
        X : numpy array [ sample_size, np.shape(sample) ] ; the traing data
        preds : numpy array [ num_estimator, sample_size, num_class,] ; the probability prediciton of each class of each training data
    """
    preds = np.mean(np.asarray(preds), axis = 0 ) # mean scores of each class of each data 
    thres = 0.7
    
    x_consensus = []
    y_consensus = []
    
    for i in range(len(X)):
        if  np.max(preds[i]) >= thres :
            x_consensus.append(X[i])
            y_consensus.append(np.argmax(preds[i]))
    
    x_consensus = np.asarray(x_consensus)
    y_consensus = np.asarray(y_consensus)[:,np.newaxis]    
            
    return x_consensus, y_consensus
    
def majority_vote(X, preds, num_voters):
    """
    pool the most voted prediciton from the estimators (absolute majority vote)
    
    A : Array. Each row is a estimator's predicition of samples.
               Each column is a sample's predicitons from estimators.
               
    return : Vector. Mode of the predicion of each samples
    """
        
    majority_votes_count = []
    
    majority_votes = []
    
    for n in range(len(preds[0])) : # for each sample
        
        cl , cnts = np.unique(preds[:,n], return_counts= True) # majority vote of estimators

        majority_votes_count.append(np.max(cnts))
        
        majority_votes.append(cl[np.argmax(cnts)])
        
    max_major = max(majority_votes_count)
    
    if  max_major > int(num_voters/2): # at least 50% agree
    
        idxs = np.where(np.asarray(majority_votes_count) == max_major) # consensus idex
            
        y_consensus = np.asarray(majority_votes)[idxs[0]]
        
        x_consensus = X[idxs]
            
        y_consensus = np.asarray(y_consensus)[:,np.newaxis]

        return x_consensus, y_consensus
    
    else :
        
        print('no consensus')
        
        return np.asarray([]), np.asarray([])
    
def unanimus_consensus(X, preds):
    
    x_consensus = []
    y_consensus = []
    preds = np.asarray(preds)
    for n in range(len(X)) : 
        pred = np.unique(preds[:,n])
        if len(pred) == 1 : # censensus on search data point    
            x_consensus.append(X[n])
            y_consensus.append(pred)            
            
    return np.asarray(x_consensus), np.asarray(y_consensus)

def max_count_pool( preds, num_voters):
    """
    pool the most voted prediciton from the estimators (absolute majority vote)
    
    A : Array. Each row is a estimator's predicition of samples.
               Each column is a sample's predicitons from estimators.
               
    return : Vector. Mode of the predicion of each samples
    """
    
    x_consensus, y_consensus = unanimus_consensus(preds)
    
    if len(x_consensus) > 0 and len(y_consensus) > 0 :
        
        return x_consensus, y_consensus
    
    else  :  # absolute majaority

        return majority_vote(preds, num_voters)



class RANSAC(object):
    """ Randomly sample data from the training set and fit N estimators.
    The consensus of the predictions of the N estimators are used as labels on the
    unlablled data set X.
    """
    
    def __init__(self, estimator, n=10):
        
        self.num_estimators = n
        
        self.num_iter = 500
        
        self.num_inner_epochs = 0
        
        self.num_outer_epochs = 0
        
        self.estimator = estimator
        
        self.estimators = []
                        
        self.x_consensus = [] 
        
        self.y_consensus = []
        
    def _fit(self, Xtrain, Ytrain) :
        """fit the estimators within iteration
        """
        # Sampling Phase
        for i in range(self.num_estimators) :
            x, y = sample_gen(Xtrain,Ytrain)
            
            if len(self.estimators) < self.num_estimators :
                self.estimator.fit(x,y.ravel())
                self.estimators.append(deepcopy(self.estimator))
                
            elif len(self.estimators) == self.num_estimators :
                self.estimators[i].fit(x,y.ravel())
    
    def _predict(self, X, mode = 'max_proba_pool') :
        """ predict on unlabled data with estmators
        """
        # Consensus Phase
        preds = []
        if mode =='unamimous_consnsesus' :
            for i in range(self.num_estimators):
                estimator = self.estimators[i]
                preds.append(estimator.predict(X))
            
            return unanimus_consensus(X, preds)
        
        elif mode == 'max_count_pool' :
            for i in range(self.num_estimators):
                estimator = self.estimators[i]
                preds.append(estimator.predict(X))
            
            return max_count_pool(X, preds)
        
        elif mode == 'max_proba_pool' :
            for i in range(self.num_estimators):
                estimator = self.estimators[i]
                preds.append(estimator.predict_proba(X))
                
            return max_proba_pool(X, preds)
            
    
    def fit(self, Xtrain, Ytrain):
        
        self.Xtrain_init = Xtrain
        self.Ytrain_init = Ytrain[:,np.newaxis]
        self.Xtrain_epoch = Xtrain
        self.Ytrain_epoch = Ytrain[:,np.newaxis]
        
    def predict_pass(self, X):
        
        self.X_epoch = X
        
        # propagation
        
        for _ in range(self.num_iter):  # add the consensus to training set     
            
            if  len(self.X_epoch) == 0 : # no more unlabeled data
                break

            self._fit(self.Xtrain_epoch, self.Ytrain_epoch)
            x, y = self._predict(self.X_epoch)
            
            if len(y) == 0 : # no more consensus
                break
            
            else :
                self.Xtrain_epoch = np.concatenate((self.Xtrain_epoch, x), axis = 0)   # update training set
                self.Ytrain_epoch = np.concatenate((self.Ytrain_epoch, y), axis = 0)
                self.X_epoch = setdiff(self.X_epoch,x)       #update unlabeled set
                
                
        return self.Xtrain_epoch[len(self.Xtrain_init):], self.Ytrain_epoch[len(self.Ytrain_init):]  
                    
    def predict_proba_pass(self, X):
        
        X_, y_  = self.predict_pass(X)
        
        preds = []
        for i in range(self.num_estimators):
            estimator = self.estimators[i]
            preds.append(estimator.predict_proba(X))
            
        preds = np.mean(np.asarray(preds), axis = 0 )
        
        return preds
    
    def predict(self, X): 
        
        X, y = self.predict_pass(X)
        
        for i in range(self.num_inner_epochs):
        
            X, y = self.predict_pass(X)
        
            return X, y
        
        for in range( self.num_outer_epochs):

            X_in = np.concatenate((self.Xtrain_init, X), axis = 0)
            y_in = np.concatenate((self.Ytrain_init, y), axis = 0)
            
            for k in range(self.num_epochs): # unlabel the mislabelled data in training set.
    
                X_train, X_test, y_train, y_test = train_test_split(X_in, y_in, test_size= 0.1)
                
                X_pred, y_pred = self.predict_pass(X_train)
                
                X_in, y_in = X_pred, y_pred
            
            return X_in, y_in
            
    def predict_proba(self, X):
        
        X_, y_  = self.predict(X)
        
        preds = []
        for i in range(self.num_estimators):
            estimator = self.estimators[i]
            preds.append(estimator.predict_proba(X))
            
        preds = np.mean(np.asarray(preds), axis = 0 )
        
        return preds
    
    
    def score(self, X):
        
        preds = self.predict_proba(X)
        
        return np.argmax(preds, axis = 1)

if __name__ == '__main__' :
    from sklearn.datasets import make_moons, make_circles, make_classification
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import precision_recall_curve, average_precision_score
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    
    n_samples = 300

    X, y = make_classification(n_samples = n_samples, n_features=2, n_redundant=0, n_informative=2,
                                 random_state=1, n_clusters_per_class=1)
    rng = np.random.RandomState(2)
    X += 2 * rng.uniform(size=X.shape)
    linearly_separable = (X, y)


    datasets = [make_moons(n_samples = n_samples, noise=0.1, random_state=0),
            make_circles(n_samples = n_samples, noise=0.1, factor=0.5, random_state=1),
            linearly_separable
            ]
    
        ## corrutped annotations
    def add_noise(targets):
        
        y_noise = targets
        
        noise_indices = []
        
        classes = np.unique(targets)
       
        # level of noise in targets    
        noise_percent = np.ones(len(classes)) * 0.5
        
        if all(noise_percent != 0.) : 
        
            for i, c in enumerate(classes) :
                
               indices = np.where(targets == c)[0]
                      
               indices = np.random.choice( indices, int(len(indices)*noise_percent[i]))
               
               noise_indices.extend(indices.tolist())
               
               other_labels = [ cl for cl in classes if cl != c]
                                     
               for j in np.asarray(indices, dtype= int) :
                       
                   y_noise[j] = np.random.choice(other_labels)
                   
            return y_noise, noise_indices, noise_percent
        
        else :
            
            return targets, None,  0 
           
        
    i = 1 
    figure = plt.figure(figsize=(9, 9))

    for ds_cnt, ds in enumerate(datasets):

        X, y = ds
        
        X = StandardScaler().fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.95, random_state=42)
            
        y_train_gt = deepcopy(y_train)    
        y_train, noise_indices, noise_percent = add_noise(y_train)
        
#        est =SVC(gamma='auto', C=1, probability =True)
        est = MLPClassifier(hidden_layer_sizes=(2,10,10,2), solver='lbfgs' )
        
        est.fit(X_train, y_train_gt)
        score_clean = est.score(X_test, y_test)
        
        est.fit(X_train, y_train)
        score_noise = est.score(X_test, y_test)
        
        rac = RANSAC(est)
        rac.fit(X_train, y_train)
        
        # predict() classification
        x_pred, y_pred = rac.predict(X_test)
        ind1, ind2 = intersect_id(X_test, x_pred)
            
        score_denoise = accuracy_score(y_test[ind1].ravel(),y_pred.ravel())
        
        # probability prediction
        
#        preds = rac.predict_proba(X_test)
#                
#        score_denoise = accuracy_score(y_test.ravel(), np.argmax(preds, axis = 1).ravel())
        
        h = .02
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        cm = plt.cm.RdBu
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])
        
        
        # Plot the training points
        ax = plt.subplot(len(datasets), 3, i*3 -2)
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        if ds_cnt == 0:
            ax.set_title("training data")
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                   alpha = 0.6)
        
        # Plot the testing points

        ax = plt.subplot(len(datasets), 3, i*3 - 1 )
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        if ds_cnt == 0:
            ax.set_title("testing data")
        ax.scatter(X_test[:, 0], X_test[:, 1], facecolors =None, cmap=cm_bright,
               edgecolors='k',alpha = 0.6)
        
        # Plot the prdiction points
        ax = plt.subplot(len(datasets), 3, i*3 )
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())        
        if ds_cnt == 0:
            ax.set_title("prediction")
            
#         non-concensus    
        X_no = setdiff(X_test, x_pred)
        ax.scatter(X_no[:,0], X_no[:,1], facecolors =None, cmap=cm_bright,
               edgecolors='k',alpha = 0.2)
        
#         concensus    
        ax.scatter(x_pred[:, 0], x_pred[:, 1], c = y_pred.ravel()  ,cmap=cm_bright,
               edgecolors='k',alpha = 0.2)
#        ax.scatter(X_test[:, 0], X_test[:, 1], c = np.argmax(preds, axis =1).ravel()  ,cmap=cm_bright,
#               edgecolors='k',alpha = 0.2)
        
        ax.text(xx.max() - 3.8, yy.min() + 0.3, ('%.2f' % score_clean).lstrip('0'), c = 'k',
        size=15, horizontalalignment='right')
        
        ax.text(xx.max() - 2.8, yy.min() + 0.3, ('%.2f' % score_noise).lstrip('0'), c = 'b',
        size=15, horizontalalignment='right')
        
        ax.text(xx.max() - 1.8, yy.min() + 0.3, ('%.2f' % score_denoise).lstrip('0'), c = 'purple',
        size=15, horizontalalignment='right')
        
        i+=1
            
        

