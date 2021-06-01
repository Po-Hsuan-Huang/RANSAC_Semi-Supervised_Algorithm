#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Random Concensus Algorityhm 2.3 utilizing unlabeled data. (fork form ransac_simple_v3.py)
"""
import numpy as np
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from copy import deepcopy
from sklearn.metrics import precision_recall_curve, average_precision_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import logging

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




class RANSAC(object):
    """ Randomly sample data from the training set and fit N estimators.
    The consensus of the predictions of the N estimators are used as labels on the
    unlablled data set X.
    """

    def __init__(self, estimator, n = 30, Epochs = 30,  sampling_rate = 0.5, mode = 'unanimous_consensus', verbose = False, save_iteration_consensus = False, save_decision_maps = False ):

        self.mode = mode

        self.num_estimators = n

        self.sampling_rate = sampling_rate

        self.num_iter = 10

        self.num_inner_epochs = 0

        self.num_outer_epochs = Epochs

        self.estimator = estimator

        self.estimators = []

        self.x_consensus = []

        self.y_consensus = []

        for i in range(self.num_estimators):
            self.estimators.append(deepcopy(estimator))

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

    def max_proba_pool( self, X, preds):
        """
        predict probability of each class for each sample, and max pooling the samples
        above probability threshold. Threshold should be learned through logit layer or
        softmax layer. However, sciki-learn doesn't allow building neural netwrok layers.
        Temperary we use fixed threshold.

        args :
            X : numpy array [ sample_size, np.shape(sample) ] ; the traing data
            preds : numpy array [ num_estimator, sample_size, num_class,] ; the probability prediciton of each class of each training data
        """


        self.verboseprint('max_praba_pool_preds_shape' , np.asarray(preds).shape)

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

    def majority_vote( self, X, preds, num_voters):
        """
        pool the most voted prediciton from the estimators (absolute majority vote)

        preds : Array. Each row is a estimator's predicition of samples.
                   Each column is a sample's predicitons from estimators.

        return : Vector. Mode of the predicion of each samples
        """

        self.verboseprint('majority_vote')

        majority_votes_count = []

        majority_votes = []

        preds = np.asarray(preds)

        for n in range(preds.shape[1]) : # for each sample

            cl , cnts = np.unique(preds[:,n], return_counts= True) # majority vote of estimators

            majority_votes_count.append(np.max(cnts))

            majority_votes.append(cl[np.argmax(cnts)])

        max_major = max(majority_votes_count)

        self.verboseprint('n_smaples', len(majority_votes), preds.shape)
        self.verboseprint('votes', majority_votes)
        self.verboseprint('counts', majority_votes_count)
        if  max_major > int(num_voters/2): # at least 50% agree

            idxs = np.where(np.asarray(majority_votes_count) == max_major) # consensus idex

            y_consensus = np.asarray(majority_votes)[idxs[0]]

            x_consensus = X[idxs]

            y_consensus = np.asarray(y_consensus)[:,np.newaxis]

            return x_consensus, y_consensus

        else :

            self.verboseprint('no consensus')

            return np.asarray([]), np.asarray([])

    def unanimus_consensus( self, X, preds):

        x_consensus = []
        y_consensus = []
        preds = np.asarray(preds)
        for n in range(len(X)) :
            pred = np.unique(preds[:,n])
            if len(pred) == 1 : # censensus on search data point
                x_consensus.append(X[n])
                y_consensus.append(pred)

        return np.asarray(x_consensus), np.asarray(y_consensus)

    def controversial(self, preds):
        """
        backward ransac to remove controversial samples when base estimators
        doesn't reach strong agreement.
        """
        indx =[]
        preds = np.asarray(preds)
        self.verboseprint('shape of preds:',preds.shape)
        for n in range(preds.shape[1]) :
            pred, cnts = np.unique(preds[:,n], return_counts = True)
            if np.max(cnts) < 6 : # no censensus on search data point
                indx.append(n)

        return indx

    def max_count_pool( self, X, preds, num_voters):
        """
        pool the most voted prediciton from the estimators (absolute majority vote)

        A : Array. Each row is a estimator's predicition of samples.
                   Each column is a sample's predicitons from estimators.

        return : Vector. Mode of the predicion of each samples
        """

        x_consensus, y_consensus = self.unanimus_consensus(X, preds)

        if len(x_consensus) > 0 and len(y_consensus) > 0 :

            return x_consensus, y_consensus

        else  :  # absolute majaority

            return self.majority_vote(X, preds, num_voters)


    def _add_to_labeled(self, Xtrain, Ytrain):
        """
        Add the pseudo labeled data to the label dataset before the start of each
        epoch.
        """
        if not hasattr(self, 'Xtrain_init') :
            self.Xtrain_init = Xtrain

        if not hasattr(self, 'Ytrain_init'):
            if  len(Ytrain.shape) < 2 :
                self.Ytrain_init = Ytrain[:,np.newaxis]
            else :
                self.Ytrain_init = Ytrain

        if not hasattr(self, 'Xtrain_epoch') :
            self.Xtrain_epoch = Xtrain
        else :
            self.Xtrain_epoch = Xtrain

        if not hasattr(self, 'Ytrain_epoch'):
            if len(Ytrain.shape) < 2 :
                self.Ytrain_epoch = Ytrain[:,np.newaxis]
            else :
                self.Ytrain_epoch = Ytrain
        else :
            if  len(Ytrain.shape) < 2 :
                self.Ytrain_epoch = Ytrain[:,np.newaxis]
            else :
                self.Ytrain_epoch = Ytrain


    def _fit(self, Xtrain, Ytrain) :
        """fit the base estimators of the ensemble within iteration
        """
        # Sampling Phase
        for i in range(self.num_estimators) :
            x, y = self.sample_gen(Xtrain,Ytrain)

#            self.verboseprint( 'sample set  class 1: {}, class 2: {}'.format(np.sum(y ==-1), np.sum(y==1)))

            if len(self.estimators) < self.num_estimators :
                self.estimator.fit(x,y.ravel())
                self.estimators.append(deepcopy(self.estimator))

            elif len(self.estimators) == self.num_estimators :
                try:
                    self.estimators[i].fit(x,y.ravel())
                except ValueError :
                    self.verboseprint('number of classes has to be greater than one; got one class.')
                    continue

    def _predict(self, X) :
        """ predict on unlabled data with estmators
        """
        # Consensus Phase
        preds = []
        if self.mode =='unanimous_consensus' :
            for i in range(self.num_estimators):
                estimator = self.estimators[i]
                preds.append(estimator.predict(X))

            return self.unanimus_consensus(X, preds)

        elif self.mode == 'max_count_pool' :
            for i in range(self.num_estimators):
                estimator = self.estimators[i]
                preds.append(estimator.predict(X))

            return self.max_count_pool(X, preds, self.num_estimators)

        elif self.mode == 'max_proba_pool' :
            for i in range(self.num_estimators):
                estimator = self.estimators[i]
                preds.append(estimator.predict_proba(X))

            return self.max_proba_pool(X, preds)

        else :

            raise NameError('Did not specify mode for concensus.')

    def _distill(self, X, y, mode = 'controversial'):
        " return distilled X,y "

        assert X.size != 0, print('Cannot distill empty array in self._distill() X.shape : ', X.shape)

        preds = []
        if mode =='controversial' :
            for i in range(self.num_estimators):
                estimator = self.estimators[i]
                preds.append(estimator.predict(X))
            self.verboseprint('list of base preds', len(preds))

            return self.controversial(preds)

    def fit_predict(self, Xtrain, Ytrain, Xelse):
        """ fit and predict of the RANSAC object with forward-backward RASNAC
            loop.

        args :
            Xtrain : array of shape (n_samples, n_features)

            Ytrain : array of shape (n_samples)

            Xelse : array of shape (n_samples, n_features)

        return:
            Xpred : X_consensus, a subset of Xelse where consensus are found

            Ypred : prediction of labels of X_consensus

        """
        self.verboseprint('Initial Xtrain size {}, initial Xelse size{}'.format(Xtrain.shape,Xelse.shape))

        self._add_to_labeled(Xtrain, Ytrain)

        self.X_epoch = np.copy(Xelse)

        if np.size(self.X_epoch) == 0 :

            if self.verboseprint :
                print('Unlabeled set empty.')

            return None, None

        # forward label spread
        self.verboseprint('first forward predict')
        x_pred, y_pred = self.predict_pass(Xelse)
        self.verboseprint('#labeled : {} #unlabeled:{}\n'.format(self.Xtrain_epoch.shape[0], self.X_epoch.shape[0]))

        for v in range(self.num_outer_epochs):
            self.verboseprint('epoch {}'.format(v+1))
            self.verboseprint('backward distillation')
            X_out, y_out = self.backward_pass(self.Xtrain_epoch, self.Ytrain_epoch) # backward distill
            self.verboseprint('#labeled : {} #unlabeled:{}\n'.format(self.Xtrain_epoch.shape[0], self.X_epoch.shape[0]))

            self.verboseprint('forward propagation')
            x_pred, y_pred = self.predict_pass(self.X_epoch) # forawrd spread
            self.verboseprint('#labeled : {} #unlabeled:{}\n'.format(self.Xtrain_epoch.shape[0], self.X_epoch.shape[0]))

        return x_pred, y_pred


    def predict_pass(self, X):
        """ forward label propagation of ensemble learning. Also known as forward RANSAC.

            input :
                X : unlabeled data
                12
            output:
                X: predicted labeled data
                y: predicted labels

        """

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
                self.verboseprint('unlabled samples: {}'.format(self.X_epoch.shape[0]))


        return self.Xtrain_epoch[len(self.Xtrain_init):], self.Ytrain_epoch[len(self.Ytrain_init):]

    def backward_pass(self, X, y):
        """ backward label distillation of ensemble learning. Also known as backward RANSAC.

        input :
            X : labeled data
            y : labels

        output:
            X : data after distillation
            y : labels after distillation
        """
        for _ in range(10):  # remvoe the controversial data from training set


#            u, c  = np.unique(self.Ytrain_epoch, return_counts = True)
#            self.verboseprint('before distill')
#            self.verboseprint('class{} : class{}'.format(u[0],u[1]))
#            self.verboseprint('counts {} : {}'.format(c[0], c[1]))
            indx= self._distill(self.Xtrain_epoch, self.Ytrain_epoch) # return indices of controversial samples

            if len(indx) == 0 : # no more controversail prediction
                break

            mask_cont = np.zeros(self.Ytrain_epoch.shape[0], dtype = bool)
            mask_cont[indx] = True
            self.X_epoch = np.concatenate([self.X_epoch, self.Xtrain_epoch[mask_cont]])
            mask = np.logical_not(mask_cont)
            self.Xtrain_epoch = self.Xtrain_epoch[mask]
            self.Ytrain_epoch = self.Ytrain_epoch[mask]

        return self.Xtrain_epoch, self.Ytrain_epoch

    def predict_proba_pass(self, X):
        """
        """

        X_, y_  = self.predict_pass(X)

        preds = []
        for i in range(self.num_estimators):
            estimator = self.estimators[i]
            preds.append(estimator.predict_proba(X))

        preds = np.mean(np.asarray(preds), axis = 0 )

        return preds

    def predict(self, X):
        """
        please run self._add_to_labeled(Xtrian, Ytrain) before this
        """
        # forward label spread
        x_pred, y_pred = self.predict_pass(X)

        X_in, y_in = self.Xtrain_epoch, self.Ytrain_epoch

        self.verboseprint('inside iteration Xtrain size', X_in.shape)

        for v in range(self.num_outer_epochs):
            X_out, y_out = self.backward_pass(X_in, y_in) # backward distill
            self._add_to_labeled(self.Xtrain_epoch, self.Ytrain_epoch)# must reset self.train_init to let predict_pass() correctly return new prediction
            x_pred, y_pred = self.predict_pass(X) # forawrd spread
            X_in, y_in = x_pred, y_pred

        return x_pred, y_pred


    def predict_proba(self, X):

        # X_, y_  = self.predict(X)

        preds = []
        for i in range(self.num_estimators):
            estimator = self.estimators[i]
            preds.append(estimator.predict_proba(X)) # only return probabilties of class [0,1]

        preds = np.mean(np.asarray(preds), axis = 0 )

        return preds

    def decision_function(self, X):

        # X_, y_  = self.predict(X)

        preds = []

        if hasattr(self.estimator, 'decision_function') :

            for i in range(self.num_estimators):
                estimator = self.estimators[i]
                preds.append(estimator.decision_function(X)) # only return probabilties of class [0,1]

            preds = np.mean(np.asarray(preds), axis = 0 )

            return preds

        else :

            return self.predict_proba(X)[:,1] -0.5

    def score(self, X):

        preds = self.predict_proba(X)

        return np.argmax(preds, axis = 1)

if __name__ == '__main__' :
    """
    RANSAC denoising example
    """

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
        noise_percent = np.ones(len(classes)) * 0.3

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

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.8, random_state=42)

        y_train_gt = deepcopy(y_train)
        y_train, noise_indices, noise_percent = add_noise(y_train)

#        est =SVC(gamma='auto', C=1, probability =True)
        est = MLPClassifier(hidden_layer_sizes=(2,10,10,2), solver='lbfgs', max_iter = 1000 )

        est.fit(X_train, y_train_gt)
        score_clean = est.score(X_test, y_test)

        est.fit(X_train, y_train)
        score_noise = est.score(X_test, y_test)

        rac = RANSAC(est)

        # predict() classification
        x_pred, y_pred = rac.fit_predict(X_test, y_test, X_test)

        ind1, ind2 = intersect_id(X_test, x_pred)

        score_denoise = accuracy_score(y_test[ind1].ravel(),y_pred.ravel())

        # probability prediction

        # preds = rac.predict_proba(X_test)

        # score_denoise = accuracy_score(y_test.ravel(), np.argmax(preds, axis = 1).ravel())

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

        # plot probability prediction
        preds = rac.predict_proba(X_test)
        ax.scatter(X_test[:, 0], X_test[:, 1], c = np.argmax(preds, axis =1).ravel()  ,cmap=cm_bright,
               edgecolors='k',alpha = 0.2)

        ax.text(xx.max() - 3.8, yy.min() + 0.3, ('%.2f' % score_clean).lstrip('0'), c = 'k',
        size=15, horizontalalignment='right')

        ax.text(xx.max() - 2.8, yy.min() + 0.3, ('%.2f' % score_noise).lstrip('0'), c = 'b',
        size=15, horizontalalignment='right')

        ax.text(xx.max() - 1.8, yy.min() + 0.3, ('%.2f' % score_denoise).lstrip('0'), c = 'purple',
        size=15, horizontalalignment='right')

        i+=1




