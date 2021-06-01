#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 02:15:14 2020

@author: pohsuanh

Ransac Simple Algorithm2 (fork from ransac_simple.py)

RANSAC semisupervised learning utilizing unlabeled data.
"""
import numpy as np
from sklearn.svm import SVC
from copy import deepcopy
from sklearn.metrics import precision_recall_curve, average_precision_score, accuracy_score
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import logging
import argparse

def intersect_id(a,b):

    a1_rows = a.view([('', a.dtype)] * a.shape[1])
    a2_rows = b.view([('', b.dtype)] * b.shape[1])
    sec, comm1, comm2 = np.intersect1d(a1_rows, a2_rows, return_indices=True)

    return comm1, comm2

def setdiff(a, b):
    a1_rows = a.view([('', a.dtype)] * a.shape[1])
    a2_rows = b.view([('', b.dtype)] * b.shape[1])

    return np.setdiff1d(a1_rows, a2_rows).view(a.dtype).reshape(-1, a.shape[1])

class RANSAC(object):
    """ Randomly sample data from the training set and fit N estimators.
    The consensus of the predictions of the N estimators are used as labels on the
    unlablled data set X.
    """

    def __init__(self, estimator, n=30, Epochs = 30, sampling_rate = 0.5 ,verbose = False, save_iteration_consensus = False, save_decision_maps = False ):
        '''

        Parameters
        ----------
        estimator : sk-learn classifier
            DESCRIPTION.
        n : int, optional
            numer of base estimators. The default is 30.
        verbose : TYPE, optional
            print debugging information. The default is False.
        save_iteration_consensus : Boolean, optional
            Used for plotting 2D decision contours. Save inliers of each iteration of the best epoch. The default is False.
        save_decision_maps : TYPE, optional
            Used for plotting 2D decision contours. Save decision maps and grids of deicision maps of each iteration of the best epoch. The default is False.

        Returns
        -------
        None.

        '''
        self.num_estimators = n

        self.sampling_rate = sampling_rate

        self.Epochs = Epochs

        self.epoch = 0

        self.estimator = estimator

        self.estimators = []

        self.x_consensus = []

        self.y_consensus = []

        if verbose:
            def verboseprint(self,*args):
                # Print each argument separately so caller doesn't need to
                # stuff everything to be printed into a single string
                for arg in args:
                   print(arg),
                print()
        else:
            self.verboseprint = lambda *a: None      # do-nothing function

        if save_iteration_consensus :

           self.iteration_consensus = []

        if save_decision_maps :

           self.decision_maps = []
           self.decision_maps_grids = []


    def sample_gen( self, X, Y):

        resample = True

        rate = self.sampling_rate

        num_totoal = len(X)

        total_ids = np.arange(num_totoal)

        while resample == True :

            np.random.shuffle(total_ids)

            sample_ids = total_ids[:int(num_totoal*rate)]

            X_s, Y_s = X[sample_ids], Y[sample_ids]

            if len(np.unique(Y_s)) < 2: # only one class in the samples, resample
                self.verboseprint('resample training set because only one class presented.')
                u, c  = np.unique(Y, return_counts = True)
                self.verboseprint('training set')
                self.verboseprint('class{} : class{}'.format(u[0],u[1]))
                self.verboseprint('counts {} : {}'.format(c[0], c[1]))

                self.verboseprint('in the samples')
                u, c = np.unique(Y_s, return_counts = True)
                self.verboseprint('class{} '.format(u[0]))
                self.verboseprint('counts {} '.format(c[0]))

            else :

                resample = False

        return X_s, Y_s

    def _fit(self, Xtrain, Ytrain) :
        """fit the base estimators of the ensemble within iteration
        """
        # Sampling Phase
        if self.epoch == 0 :
            for i in range(self.num_estimators) :
                x, y = self.sample_gen(Xtrain,Ytrain)
                self.estimator.fit(x,y.ravel())
                self.estimators.append(deepcopy(self.estimator))
        else :
            for i in range(self.num_estimators)  :
                self.estimators[i].fit(Xtrain,Ytrain.ravel())



    def _predict(self, X) :
        """ predict on unlabled data with estmators
        """
        self.verboseprint('_predict: ',X.shape[0])
        # Consensus Phase
        preds = []
        for i in range(self.num_estimators):
            estimator = self.estimators[i]
            pred = estimator.predict(X)
            preds.append(pred)
        self.verboseprint('done predict')
        x_consensus = []
        y_consensus = []
        preds = np.asarray(preds)
        for n in range(len(X)) :
            pred = np.unique(preds[:,n])
            if len(pred) == 1 : # censensus on seach data point
                x_consensus.append(X[n])
                y_consensus.append(pred)

        return np.asarray(x_consensus), np.asarray(y_consensus)

    def fit_predict(self, Xtrain, Ytrain, Xelse ):
        """
        Set up X_init by sampling 90% of the labeled data.

        """
        self.Xtrain_init = Xtrain
        self.Xtrain_epoch = Xtrain

        if  len(Ytrain.shape) < 2 :
            self.Ytrain_init = Ytrain[:,np.newaxis]
            self.Ytrain_epoch = Ytrain[:,np.newaxis]
        else :
            self.Ytrain_init = Ytrain
            self.Ytrain_epoch = Ytrain

        self.X_epoch = Xelse

        try  :

            assert np.size(self.X_epoch) > 0 , 'unlabeled data array empty.'

        except AssertionError :

            self._fit(self.Xtrain_epoch, self.Ytrain_epoch)

            return self.Xtrain_epoch, self.Ytrain_epoch

        # propagation

        for self.epoch in range(self.Epochs):  # add the consensus to training set

            self._fit(self.Xtrain_epoch, self.Ytrain_epoch)

            x, y = self._predict(self.X_epoch)

            if len(y) == 0 : # no more consensus
                break

            else :

                if hasattr(self, 'iteration_consensus') :
                    self.verboseprint('append iter')
                    self.iteration_consensus.append((x,y))

                if hasattr(self, 'decision_maps') :
                    self.verboseprint('append decision map')
                     # Create mesh for contour plot
                    h = 0.06  # step size in the mesh
                    x1_min, x1_max = self.Xtrain_epoch[:, 0].min() - .5, self.Xtrain_epoch[:, 0].max() + .5
                    x2_min, x2_max = self.Xtrain_epoch[:, 1].min() - .5, self.Xtrain_epoch[:, 1].max() + .5
                    xx, yy = np.meshgrid(np.arange(x1_min, x1_max, h),
                                         np.arange(x2_min, x2_max, h))
                    if hasattr(self.estimator, "decision_function"):
                        decision_map = self.decision_function(np.c_[xx.ravel(), yy.ravel()])
                    else:
                        decision_map = self.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1] - 0.5

                    decision_map= decision_map.reshape(xx.shape)
                    self.decision_maps.append(decision_map)
                    self.decision_maps_grids.append((xx,yy))


                self.Xtrain_epoch = np.concatenate((self.Xtrain_epoch, x), axis = 0)   # update training set
                self.Ytrain_epoch = np.concatenate((self.Ytrain_epoch, y), axis = 0)
                self.X_epoch = setdiff(self.X_epoch,x)       #update unlabeled set
#                self.verboseprint('size of x epoch', self.X_epoch,shape)

            if len(self.X_epoch) == 0:
                break

        return self.Xtrain_epoch[len(self.Xtrain_init):], self.Ytrain_epoch[len(self.Ytrain_init):]

    def predict_proba(self, X):
        '''
        Predict class probabilities for unlabled data.

        Parameters
        ----------
        X : TYPE
            Unlabeled samples to be predicted.

        Returns
        -------
        None.

        '''

#        X_, y_  = self.predict(X)

        preds = []
        for i in range(self.num_estimators):
            estimator = self.estimators[i]
            preds.append(estimator.predict_proba(X))

        preds = np.mean(np.asarray(preds), axis = 0 )

        return preds

    def score(self, X):

        preds = self.predict_proba(X)

        return np.argmax(preds, axis = 1)

    def decision_function(self, X):
        '''
        Predict decision function value for unlabled data.

        Parameters
        ----------
        X : TYPE
            Unlabeled samples to be predicted.

        Returns
        -------
        None.

        '''

#        X_, y_  = self.predict(X)
        if hasattr(self.estimator, 'decision_function') :
            preds = []
            for i in range(self.num_estimators):
                estimator = self.estimators[i]
                preds.append(estimator.decision_function(X))

            preds = np.mean(np.asarray(preds), axis = 0 )

            return preds

        else :

            preds = self.predict_proba(X)[:,1]

            return preds -0.5

    def score(self, X):

        preds = self.predict_proba(X)

        return np.argmax(preds, axis = 1)

if __name__ == '__main__' :

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
        noise_percent = np.ones(len(classes)) * 0.4

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

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.9, random_state=42)

        y_train_gt = deepcopy(y_train)
        y_train, noise_indices, noise_percent = add_noise(y_train)

        est =SVC(gamma='auto', C=1, probability =True)
#        est = MLPClassifier(hidden_layer_sizes=(10,10), solver='lbfgs' )
        est.fit(X_train, y_train_gt)
        score_clean = est.score(X_test, y_test)

        est.fit(X_train, y_train)
        score_noise = est.score(X_test, y_test)

        rac = RANSAC(est)
        rac.fit_predict(X_train, y_train)

        # predict() classification
        x_pred, y_pred = rac.predict(X_test)
        ind1, ind2 = intersect_id(X_test, x_pred)

        score_denoise = accuracy_score(y_test[ind1].ravel(),y_pred.ravel())

        # probability prediction

        preds = rac.predict_proba(X_test)

        score_denoise = accuracy_score(y_test.ravel(), np.argmax(preds, axis = 1).ravel())

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

        # non-concensus
#        X_no = setdiff(X_test, x_pred)
#        ax.scatter(X_no[:,0], X_no[:,1], facecolors =None, cmap=cm_bright,
#               edgecolors='k',alpha = 0.2)

        # concensus
#        ax.scatter(x_pred[:, 0], x_pred[:, 1], c = y_pred.ravel()  ,cmap=cm_bright,
#               edgecolors='k',alpha = 0.2)

        ax.scatter(X_test[:, 0], X_test[:, 1], c = np.argmax(preds, axis =1).ravel()  ,cmap=cm_bright,
               edgecolors='k',alpha = 0.2)

        ax.text(xx.max() - 3.8, yy.min() + 0.3, ('%.2f' % score_clean).lstrip('0'), c = 'k',
        size=15, horizontalalignment='right')

        ax.text(xx.max() - 2.8, yy.min() + 0.3, ('%.2f' % score_noise).lstrip('0'), c = 'b',
        size=15, horizontalalignment='right')

        ax.text(xx.max() - 1.8, yy.min() + 0.3, ('%.2f' % score_denoise).lstrip('0'), c = 'purple',
        size=15, horizontalalignment='right')

        i+=1




