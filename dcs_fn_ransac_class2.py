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

2020 July 9.

The model is updated based on the Three-stage-learning : Supervision, Supervised-generalizatoin,
 Transductive-Generalization.

 Supervision : randomly samples training set and train a weak classifier.

 Supervised-generalization : validate the the weak model with remaining labels
 (which might not be necessary, because it prevents diversification)

 Inductive-Generalization : weakly-supervised model use decision boundary or
 probability estimate to assign confience values to unlabeled data. Under the assumption
 of sufficient training epochs. There exists a good subset of training samples that
 best generatlized to the unlabled data by excluding low confidence samples from
 the pseudo-labels. The other approach is to include high confidence samples to the
 pseudo-labels. The former is called substractive labeling, the later is called
 additive labeling.

"""
import numpy as np
from sklearn.svm import SVC
import pathos.multiprocessing as mp
from multiprocessing import Process, Queue
from copy import deepcopy
from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score
import time

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
       return v
    return v / norm

def verboseprint(*args, **kwargs):
    '''
    Verbose print for debugging. Equivalent to Logging.info()

    Parameters
    ----------
    *args : TYPE
        string to be printed on stdio.
    **kwargs : TYPE
        ''verbose' : Bool
            print the input string of verbose == True

    '''
    if 'verbose' in kwargs.keys() :
       verbose = kwargs['verbose']
       if verbose :
           for arg in args:
              print(arg)
           print()
    else:
       return lambda *a: None      # do-nothing function

def find_inliers(estimator, X_all, X, y, verbose = False):
    '''
    This part has major difference from previous RSVM1 confidence metric.
    Before we catch 10 and 90 percentile confidence interval from all unlabeled data.
    However, when unlabeled data are few and sparse. It is misleading that the
    90 percnetile of sampled unlabeled data distribution is the same as 90 percentile of
    gobal confidence landscape of the whole feature space. In face np.percentile grabs the nearest
    point to be the 90 percentile sample, even if that sample has negatvie confidence score.
    By estimating the gobal confidence score distribution, there's no longer sign issue for the
    confidence interval.

    Parameters
    ----------
    estimator : TYPE
        DESCRIPTION.
    X_all : TYPE
        Union of labled set and unlabel set.
    X : TYPE
        Data set to be worked on.
    y : TYPE
        Data set to be worked on.
    verbose : TYPE, optional
        DESCRIPTION. The default is False.

    Raises
    ------

        DESCRIPTION.
    ValueError
        DESCRIPTION.

    Returns
    -------
    X_inliers : TYPE
        DESCRIPTION.
    y_inliers : TYPE
        DESCRIPTION.
    idx_inliers : TYPE
        DESCRIPTION.

    '''

    # Evaluate the decision boundary of the whole feature space.
    x1_min, x1_max = X_all[:, 0].min() - .5, X_all[:, 0].max() + .5
    x2_min, x2_max = X_all[:, 1].min() - .5, X_all[:, 1].max() + .5
    h1 = (x1_max -x1_min)/10
    h2 = (x2_max -x2_min)/10
    xx, yy = np.meshgrid(np.arange(x1_min, x1_max, h1),
                         np.arange(x2_min, x2_max, h2))

    if hasattr(estimator, "decision_function"):
        decision_map = estimator.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        decision_map = estimator.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1] -0.5


    if hasattr(estimator, "decision_function"):
          confidence = estimator.decision_function(X)
    else:
          confidence = estimator.predict_proba(X)[:, 1]

          confidence = confidence - 0.5

    down, up = np.percentile(decision_map,[20,80])

    verboseprint('90 percentile : %.2f, 10 percentile : %.2f' % (up, down), verbose = verbose)

    confidence_interval = max( max(abs(down), abs(up)), 0.1)

    idx_inliers_up = confidence >= confidence_interval

    idx_inliers_down = confidence <= -confidence_interval

    idx_inliers = idx_inliers_down | idx_inliers_up

    X_inliers, y_inliers = X[idx_inliers], y[idx_inliers]

    if verbose:

        print(' ground truth validation phase')

        print('percentage of inliers_down {:f}'.format(sum(idx_inliers_down)/len(confidence)))

        print('percentage of inliers_up {:f}'.format(sum(idx_inliers_up)/len(confidence)))

        print('percentage of inliers {:f}'.format(sum(idx_inliers)/len(confidence)))

    return X_inliers, y_inliers, idx_inliers


class SVC_(object):

    def __init__(self):

        self.estimator = SVC(gamma = 1.0, C = 100.0, class_weight = 'balanced' )

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

        Max iteration is empriricaly defined as squared root of sample set size, but it depends on actual data distribution.

        Stopping signal is returned if count reaches max iterations.
        """
        self.max_iter  = max(int(np.sqrt(X_size)),10)
        self.count = 1
        self.stop = False

    def add_one(self):

        self.count += 1

    def reset(self):

        self.count =1
#%% Global Matching : Random Sample and Concensus (RANSAC)

class RANSAC_Classifier(object):



    def __init__(self, estimator = 'svm', error_tolerance = 0.5, sampling_rate = 0.9,
                 n_epoch = 30, class_weight= None, probability=True, save_iteration_inliers = False,
                 save_decision_maps = False, verbose = False):
        '''


        Parameters
        ----------
        estimator : sklearn classifier , optional
            DESCRIPTION. The default is 'svm'.
        error_tolerance : flaot, optional
            DESCRIPTION. The default is 0.2.
        sampling_rate : between 0, 1
            Percentage of training set being sampled.
        n_epoch : integer, optional
            Number of epoch of RANSAC. The more epochs, the better change for convergence.
            The default is sqrt(N), where N is the number of training samples.
        class_weight : Boolean, optional
            weigth_blance argument for estimator. The default is None.
        probability : Boolean, optional
            argument for estimator. The default is True.
        save_iteration_inliers : Boolean, optional
            Used for plotting 2D decision contours. Save inliers of each iteration of the best epoch. The default is False.
        save_decision_maps : TYPE, optional
            Used for plotting 2D decision contours. Save decision maps and grids of deicision maps of each iteration of the best epoch. The default is False.

        Returns
        -------
        None.

        '''
        self.id = 1

        self.results = []

        self.result_queue = Queue()

        self.train_size_init = None

        self.match_thres = error_tolerance

        self.learning_constant = 0.

        self.n_epoch = n_epoch

        self.estimator = estimator

        self.best_estimator = estimator

        self.sampling_rate = sampling_rate

        self.class_weight = class_weight

        self.verbose = verbose


        '''BEST RESULT OF ALL EPOCHS '''

        class Best(object):

            def __init__(self, save_iteration_inliers, save_decision_maps):

                self.AP = None

                self.estimator = None

                if save_iteration_inliers :

                    self.iteration_inliers = None

                if save_decision_maps :

                    self.decision_maps = None

                    self.decision_maps_grids = None

        self.best = Best(save_iteration_inliers, save_decision_maps)

        self.save_iteration_inliers = save_iteration_inliers # Boolean flag

        if save_iteration_inliers :

            self.iteration_inliers = []# Temporary storage of iteration inliers in a subprocess. Should not be visible nor callable as class object.


        self.save_decision_maps = save_decision_maps # Boolean flag

        if save_decision_maps :

            self.decision_maps = [] # Temporary storage of decision maps of all iterations in a subprocess. Should not be visible nor callable as class object.

            self.decision_maps_grids = [] # Temporary storage of grids of decision maps of all iterations in a subprocess. Should not be visible nor callable as class object.


        if estimator =='svm' :

            self.gamma = 1.0

            self.C = 100

            self.class_weight = 'balanced'

            self.estimator = SVC( C = self.C, gamma = self.gamma,  class_weight = self.class_weight, probability=True)


    def verboseprint(self, *args):
             if self.verbose:
                for arg in args:
                   print(arg)
                print()
             else:
               return lambda *a: None      # do-nothing function

    def fit_parallel(self, X, Y):

        """ X is the input data, Y is the class annotation.
        Unlabelled data are annotated '-1'.
        """

        def _work(Xtrain, Ytrain, Xelse, result_queue):

            result = ransac_processor(save_iteration_inliers =
                                      self.save_iteration_inliers,
                                      save_decision_maps =
                                      self.save_decision_maps
                                      ).fit(Xtrain, Ytrain, Xelse)

            result_queue.enqueue(result)

        Xtrain = X[np.argwhere(Y != -1)].squeeze()

        Xelse = X[np.argwhere(Y == -1)].squeeze()

        Ytrain = Y[np.argwhere(Y != -1)].squeeze().astype(int)

        jobs = []

        global start0

        start0 = time.time()

        i = 0

        # Define default size of epochs when not specified.
        if not self.n_epoch :

            self.n_epoch = int(np.floor(np.sqrt(X.shape[0])))

        while i < (self.n_epoch) :

            if  self.result_queue.size() <   2 * mp.cpu_count() -2 :

                p = Process(target = _work, args = (Xtrain, Ytrain, Xelse, self.result_queue))

                jobs.append(p)

                p.start()

                self.verboseprint( 'ChildProcess...',i)

                i += 1

            else :

                time.sleep(10)


        for p in jobs:

            p.join()

        self.results = [ self.result_queue.dequeue() for i in range(self.result_queue.qsize())]

        if self.save_iteration_inliers and not self.save_decision_maps:

            best_aps, best_estimators, collect_n_iteration_inliers = list(zip(*self.results))

        if self.save_decision_maps and not self.save_iteration_inliers:

             best_aps, best_estimators, decision_maps, decision_maps_grids = list(zip(*self.results))

        if self.save_iteration_inliers and self.save_decision_maps :

             best_aps, best_estimators, collect_n_iteration_inliers, collect_n_decision_maps, collect_n_decision_maps_grids = list(zip(*self.results))

        else :

             best_aps, best_estimators = list(zip(*self.results))




        self.best.AP = np.max(best_aps)

        best_epoch_id = np.argmax(best_aps)

        self.best.estimator = best_estimators[ best_epoch_id]

        if self.save_iteration_inliers :

            self.best.iteration_inliers = collect_n_iteration_inliers[best_epoch_id]

        if self.save_decision_maps :

            self.best.decision_maps = collect_n_decision_maps[best_epoch_id]

            self.best.decision_maps_grids = collect_n_decision_maps_grids[best_epoch_id]




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

#        verboseprint('validation_average_precision: ', average_precision_score(Yval, pred) )

        y_score = self.predict_proba(Xval)[:,1]

        self.average_precision = average_precision_score(Yval, y_score)

#        verboseprint('Average precision-recall score: {0:0.2f}'.format(average_precision))

        self.precision, self.recall, self.thresholds = precision_recall_curve(Yval, y_score)

    def save_training_info(self, X_all, X_inliers, Y_inliers) :

        '''
           save X_inliers and decision maps of every iteration for data visualization.

           X_all : Label set + Unlabel set. Used to define region to be plotted.

        '''
        if self.save_iteration_inliers:
            self.iteration_inliers.append((X_inliers, Y_inliers))

        if self.save_decision_maps:
             # Create mesh for contour plot
            h = 0.06  # step size in the mesh
            x1_min, x1_max = X_all[:, 0].min() - .5, X_all[:, 0].max() + .5
            x2_min, x2_max = X_all[:, 1].min() - .5, X_all[:, 1].max() + .5
            xx, yy = np.meshgrid(np.arange(x1_min, x1_max, h),
                                 np.arange(x2_min, x2_max, h))

            if hasattr(self.estimator, "decision_function"):
                decision_map = self.estimator.decision_function(np.c_[xx.ravel(), yy.ravel()])
            else:
                decision_map = self.estimator.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

            decision_map= decision_map.reshape(xx.shape)

            self.decision_maps.append(decision_map)
            self.decision_maps_grids.append((xx,yy))

    def sample(self, Xtrain, Ytrain) :

        if not self.train_size_init :

            self.Xtrain_init = Xtrain.copy()

            self.train_size_init = Xtrain.shape[0]

            self.verboseprint('train_size_init')

        self.train_size = Xtrain.shape[0]

        self.num_samples = max(int(self.sampling_rate*len(Ytrain)),10)

        idx_sample = np.arange(self.train_size)

        np.random.shuffle(idx_sample)

        idx_sample = idx_sample[:self.num_samples]

        Xsample, Ysample = Xtrain[idx_sample], Ytrain[idx_sample]

        idx_all = np.arange(self.train_size)

        idx_res = np.setdiff1d(idx_all, idx_sample)

        Xres, Yres = Xtrain[list(idx_res)], Ytrain[list(idx_res)]

        return Xsample, Ysample, Xres, Yres, idx_sample, idx_res

    def fit_sub(self, Xtrain, Ytrain, Xelse):

        self.best_AP = 0.

        Ysample = 1

        while len(np.unique(Ysample)) == 1 : # ensure drawn samples contains both classes

            self.verboseprint('Sampling...')

            Xsample, Ysample, Xres, Yres, idx_sample, idx_res = self.sample(Xtrain, Ytrain)

        '''' Supervised learning with training set '''

        self.estimator.fit(Xsample, Ysample)

        Xres_0 = Xres

        Yres_0 = Yres

        Xtrain_0 = Xtrain

        Ytrain_0 = Ytrain

        Xelse_0 = Xelse

        X_all = np.concatenate([Xtrain_0, Xelse_0])

        preds = self.estimator.predict(Xres)

        AP0 = average_precision_score(Yres, preds) #baseline average precision

        ''' while loop to re-evaluate the SSL improvement with the remaining labeled samples.'''

        while self.counter.count < self.counter.max_iter :

            self.verboseprint('Iteration : {:}'.format(self.counter.count))

            '''Sample validation with remaining training set.'''

            try :

                X_inliers, y_inliers, idx_inliers = find_inliers(self.estimator, X_all, Xres, Yres, verbose = self.verbose)

                if len(y_inliers) == 0:

                    if self.best_AP != 0 :

                        self.counter.add_one()

                        raise ValueError(' no more inliers in validation')

                    if self.best_AP == 0 :

                        self.counter.add_one()

                        raise ValueError(' bad sampling for validation')

                self.estimator.fit(Xsample, Ysample)

                preds = self.estimator.predict(Xres_0)

                AP0 = f1_score(Yres_0, preds)#average_precision_score(Yres_0, preds) #baseline average precision

                # preds = self.estimator.predict(X_inliers)

                # AP = average_precision_score(y_inliers, preds)


                preds = self.estimator.predict(Xres_0)

                AP = f1_score(Yres_0, preds) #average_precision_score(Yres_0, preds)

                # AP = self.estimator.score(Xres, Yres) don't use mean accuracy for multilabeling.

                self.verboseprint('AP:', AP)

                # if enough mathces are found, declare it to be a good estimate,
                # refit the estimator to the expanded set.

                # if AP >= self.match_thres + n * self.learning_constant :

                if AP >= AP0 : # augmented training set enhance prediction accuracy.

                    if AP >= self.best_AP : # Average precision improves over iterations.

                        self.best_AP = AP

                        self.best_estimator = deepcopy(self.estimator)

                        self.verboseprint('Best average precision :',self.best_AP)

                        idx_sample = np.union1d(idx_sample ,idx_res[idx_inliers])

                        idx_res = np.setdiff1d( idx_res ,idx_res[ idx_inliers] )

                        Xsample, Ysample = Xtrain[idx_sample], Ytrain[idx_sample]

                        Xres, Yres = Xtrain[idx_res], Ytrain[idx_res]

                        self.verboseprint('Xsample Ysample updated.')

                        self.counter.reset()

                        if len(idx_res) == 0:
                            # raise ValueError('Validation set emptied.')
                            self.verboseprint(' Validation set emptied. Training terminates.')
                            break

                        else :

                            self.verboseprint('Training progess : {:.2f}'.format(len(Ysample)/len(X_all[:,0])))

                    else :
                        self.counter.add_one()

                else :
                    self.counter.add_one()
                    self.verboseprint(' Xsample Ysample not updated.')
                    self.estimator = deepcopy(self.best_estimator)

                self.verboseprint('Go to Transductive Gerneralization.')


            except ValueError as inst:

                print(inst.args)

                if 'Negative sign of up percentile.' in inst.args:

                    pass# break

                if 'Positive sign of down percentile.' in inst.args:

                    pass# break

                self.verboseprint('Go to Transductive Generalization.')

                # raise KeyError('Unkown Bug')

                # import pdb

                # pdb.set_trace()

            try :

                self.verboseprint(' Utilizing Unlabelled Data...')

                Yelse = self.estimator.predict(Xelse)

                X_inliers, Y_inliers, idx_inliers = find_inliers(self.estimator, X_all, Xelse, Yelse, verbose = self.verbose)

                if Y_inliers.size == 0:

                    if self.best_AP != 0 :

                        raise ValueError(' no more inliers in unlabeled set.')

                    if self.best_AP == 0 :

                        raise ValueError(' bad sampling for transductive generalization.')

                preds = self.estimator.predict(Xres_0)

                AP0 = f1_score(Yres_0, preds)  #average_precision_score( Yres_0, preds)

                self.estimator.fit(np.concatenate((Xsample,X_inliers)), np.concatenate((Ysample, Y_inliers)))

                preds = self.estimator.predict(Xres_0)

                AP = f1_score(Yres_0, preds)  #average_precision_score(Yres_0, preds)

                if AP >= AP0 : # augmented training set enhance prediction accuracy.

                    if AP >= self.best_AP : # Average precision improves over iterations.

                       self.best_AP = AP

                       self.best_estimator = deepcopy(self.estimator)

                       self.verboseprint('Best average precision :', self.best_AP)

                       self.train_size = Xtrain.shape[0]

                       Xtrain, Ytrain = np.concatenate((Xtrain,X_inliers)), np.concatenate((Ytrain, Y_inliers))

                       Xelse, Yelse = Xelse[~idx_inliers], Yelse[~idx_inliers]

                       expanded_size = Xtrain.shape[0]

                       idx_pseudo_labeled = np.arange(self.train_size, expanded_size)

                       idx_sample = np.union1d(idx_sample, idx_pseudo_labeled)

                       Xsample, Ysample = Xtrain[idx_sample], Ytrain[idx_sample]

                       self.verboseprint('Xsample Ysample updated.')

                       self.save_training_info(X_all, X_inliers, Y_inliers)

                       if Yelse.size == 0:
                            # raise ValueError('Unlabeled set emptied.')
                            self.verboseprint(' Unlabled set emptied. Training terminates.')
                            break

                    else :

                        pass

                else :

                    self.verboseprint(' Xsample Ysample not updated.')

                    self.estimator = deepcopy(self.best_estimator)

                self.verboseprint('Go to Self Validation.')

            except ValueError as inst:

                print(inst.args)

                if 'Negative sign of up percentile.' in inst.args :

                    pass# break

                if'Positive sign of down percentile.' in inst.args :

                    pass# break

                self.verboseprint('Go to Self Validation.')

                # raise KeyError('Unkown Bug')

                # import pdb

                # pdb.set_trace()



        if self.save_iteration_inliers:

            if self.save_decision_maps :
                return (self.best_AP, self.best_estimator, self.iteration_inliers, self.decision_maps, self.decision_maps_grids)

            else :
                return(self.best_AP, self.best_estimator, self.iteration_inliers)

        else :
            return (self.best_AP,self.best_estimator)

    def fit(self, X, Y) :

        Xtrain = X[np.argwhere(Y != -1)].squeeze()

        Xelse = X[np.argwhere(Y == -1)].squeeze()

        Ytrain = Y[np.argwhere(Y != -1)].squeeze()

        self.counter = iter_counter(Xtrain.shape[0])

        for i in range(self.n_epoch) :


            self.iteration_inliers = []

            self.decision_maps, self.decision_maps_grids = [], []

            self.counter.reset()

            result = self.fit_sub(Xtrain, Ytrain, Xelse)

            self.result_queue.put(result)



        self.results = [ self.result_queue.get() for i in range(self.result_queue.qsize())]

        if self.save_iteration_inliers and not self.save_decision_maps:

            best_aps, best_estimators, collect_n_iteration_inliers = list(zip(*self.results))

        if self.save_decision_maps and not self.save_iteration_inliers:

             best_aps, best_estimators, collect_n_decision_maps, collect_n_decision_maps_grids = list(zip(*self.results))

        if self.save_iteration_inliers and self.save_decision_maps :

             best_aps, best_estimators, collect_n_iteration_inliers, collect_n_decision_maps, collect_n_decision_maps_grids = list(zip(*self.results))

        else :

             best_aps, best_estimators = list(zip(*self.results))


        self.best.AP = np.max(best_aps)

        best_epoch_id = np.argmax(best_aps)

        # best_epoch_id = np.argmax([len(n_iteration_inliers) for n_iteration_inliers in collect_n_iteration_inliers])

        self.best.estimator = best_estimators[ best_epoch_id]

        if self.save_iteration_inliers :

            self.best.iteration_inliers = collect_n_iteration_inliers[best_epoch_id]

        if self.save_decision_maps :

            self.best.decision_maps = collect_n_decision_maps[best_epoch_id]

            self.best.decision_maps_grids = collect_n_decision_maps_grids[best_epoch_id]



class ransac_processor(RANSAC_Classifier):

    def __init__(self, save_iteration_inliers = False, save_decision_maps = False) :

        super(ransac_processor, self).__init__( save_iteration_inliers = save_iteration_inliers,
                                               save_decision_maps = save_decision_maps)

        self.verboseprint('Initializing...')

        self.best_AP = 0

#        self.id += self.id
#
#        verboseprint('selfid :', self.id)

        if self.estimator == None :

            k = self.print('no estimator, continue ? (y/n)')

            if k == 'n' :

                import os

                os.sys.exit()
        # else :

        #     verboseprint('estimator :', self.estimator)

    def _sample(self, Xtrain, Ytrain) :
        '''
        Sampling form training set by sampling rate for RANSAC.

        Parameters
        ----------
        Xtrain : TYPE
            DESCRIPTION.
        Ytrain : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''

        if not self.train_size_init :

            self.Xtrain_init = Xtrain.copy()

            self.train_size_init = Xtrain.shape[0]

        self.train_size = Xtrain.shape[0]

        self.num_samples = max(int( self.sampling_rate*len(Ytrain)),10)

        idx_sample = np.arange(self.train_size)

        np.random.shuffle(idx_sample)

        idx_sample = idx_sample[:self.num_samples]

        Xsample, Ysample = Xtrain[idx_sample], Ytrain[idx_sample]

        self.idx_all = np.arange(self.train_size)

        idx_res = np.setdiff1d(self.idx_all, idx_sample)

        Xres, Yres = Xtrain[list(idx_res)], Ytrain[list(idx_res)]

        return Xsample, Ysample, Xres, Yres, idx_sample, idx_res

    def fit(self, Xtrain, Ytrain, Xelse):

        Ysample = 1

        while len(np.unique(Ysample)) == 1 : # ensure drawn samples contains both classes

#            verboseprint('Sampling...')

            Xsample, Ysample, Xres, Yres, idx_sample, idx_res = self._sample(Xtrain, Ytrain)

        n = 0

        while True :

#            verboseprint('label_propagation...')
            # Find lablled data that matches this fit

            self.estimator.fit(Xsample, Ysample)

            pred = self.estimator.predict(Xres)

            AP = average_precision_score(Yres, pred)

            # if enough mathces are found, declare it to be a good estimate,
            # refit the estimator to the expanded set.

            if AP >= self.match_thres + n * self.learning_constant :

                n += 1

                if AP > self.best_AP :

                    self.best_AP = AP

#                    verboseprint('Best average precision :',self.best_AP)

                    self.best_estimator = deepcopy(self.estimator) # deep copy

                # Rest of the labelled data

                confidence = self.estimator.decision_function(Xres)

#                confidence = (confidence - np.mean(confidence) )/np.std(confidence) + 0.5

                down, up = np.percentile(confidence, [10, 90])

                interval = max( max(abs(down), abs(up)), 0.1)

                idx_inlier_up = confidence >= interval

                idx_inlier_down = confidence <= -interval

                idx_inlier = idx_inlier_down | idx_inlier_up

                assert len(idx_inlier) <= idx_res.size, 'IndexError: too many indices for array {} > {}'.format(len(idx_inlier), idx_res.size)

                idx_sample = np.union1d(idx_sample ,idx_res[ idx_inlier])

                idx_res = np.setdiff1d( idx_res ,idx_res[ idx_inlier ] )

                Xsample, Ysample = Xtrain[idx_sample], Ytrain[idx_sample]

                Xres, Yres = Xtrain[idx_res], Ytrain[idx_res]

#                verboseprint('training %',(self.train_size_init - len(idx_res))/ self.train_size_init)

                if(self.train_size_init - len(idx_res))/ self.train_size_init == 1.0 : # training finished.

                    self.verboseprint('Propagation finished')

                    break

                if  Xelse.size != 0 and (self.train_size_init - len(idx_res))/ self.train_size_init > 0.0 :

#                    verboseprint(' Unitilizing Unlabelled Data...')

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

                    # save X_inliers and decision maps of every iteration for data visualization
                    if self.save_iteration_inliers :

                        self.iteration_inliers.append((X_inliers, Y_inliers))

                    if self.save_decision_maps:

                         # Create mesh for contour plot
                        h = 0.06  # step size in the mesh
                        x1_min, x1_max = Xtrain[:, 0].min() - .5, Xtrain[:, 0].max() + .5
                        x2_min, x2_max = Xtrain[:, 1].min() - .5, Xtrain[:, 1].max() + .5
                        xx, yy = np.meshgrid(np.arange(x1_min, x1_max, h),
                                             np.arange(x2_min, x2_max, h))

                        if hasattr(self, "decision_function"):
                            decision_map = self.decision_function(np.c_[xx.ravel(), yy.ravel()])
                        else:
                            decision_map = self.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

                            decision_map= decision_map.reshape(xx.shape)

                        self.decision_maps.append(decision_map)

                        self.decision_maps_grids.append((xx,yy))



                    # expande the dataset with the new lablled data

                    Xtrain, Ytrain = np.concatenate((Xtrain,X_inliers)), np.concatenate((Ytrain, Y_inliers))

#                    Xtrain, Ytrain = np.concatenate((Xtrain,X_outliers)), np.concatenate((Ytrain, Y_outliers))

                    expanded_size = Xtrain.shape[0]

                    idx_sample = np.union1d(idx_sample, np.arange(self.train_size, expanded_size))

                    Xsample, Ysample = Xtrain[idx_sample], Ytrain[idx_sample]

            else :

                break

        if self.save_iteration_inliers:

            if self.save_decision_maps :

                return (self.best_AP, self.best_estimator, self.iteration_inliers, self.decision_maps, self.decision_maps_grids)

            else :
                return(self.best_AP, self.best_estimator, self.iteration_inliers)

        else :
            return (self.best_AP,self.best_estimator)

