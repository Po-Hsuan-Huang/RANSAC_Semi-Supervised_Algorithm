# README #

Random Sample Concensus Models

### RANSAC1 Models ###

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

#### dcs_fn_ransac_estimator.py

Version 1 of RANSAC1

#### dcs_fn_ransac_estimator_2.py

Version2 

Improvements : 

* Resampling :

	When the initial sample set only contains 1 class, it is re-sampled. However, without modifying the time-based random seed. The samples will look similar
	and that causes probelm when doing multi-processing. This version handles that for multi-processing.

* Confidence Metric :

	* In version 1, the value of 0.1 and 0.9 does not make any sense since the probability is not normally distributed between [0,1].  
	
	
					if hasattr(self.estimator, "decision_function"): # calculate confidence score of the unlablled data

						confidence = self.estimator.decision_function(Xelse)

					else:

						confidence = self.estimator.predict_proba(Xelse)
                    
	
					confidence = (confidence - np.mean(confidence) )/np.std(confidence) + 0.5   
					
                    X_inliers, Y_inliers = Xelse[  np.abs(confidence) >= 0.9 ], Yelse[  np.abs(confidence) >= 0.9 ]
                    
                    X_outliers, Y_outliers = Xelse[ np.abs(confidence) < 0.1  ], Yelse[ np.abs(confidence) < 0.1 ]
					
					
	
	* In version 2, we use percentile: 


                down, up = np.percentile(confidence, [10, 90])
				
				X_inliers, Y_inliers = Xelse[  confidence >= up | confidence <= down], Yelse[ confidence >= up | confidence <= down]
	
                        
	
* Remove function predict_parallel() :

	* The function is removed from version 2. Another multi-prcessing model *dcs_fn_estimator_2_multiprocessing.py* is created.

#### dcs_fn_ransac_class2.py

* Built on top of dcs_fn_ransac_estimator_2.py

* Previous version does not chekc if the f1 score improves every iteration during inductive generalization. 
   
    	If the resampling in the while loop doesn't yeild better seed training samples 5 times stright, the iteration terminates.
    	Max iteration is empriricaly defined as squared root of sample set size, but it depends on actual data distribution.
    	Stopping signal is returned if count reaches max iterations.
   
* def find_inliers(estimator, X_all, X, y, verbose = False):
    
    	This part has major difference from previous RSVM1 confidence metric.
    	Before we catch 10 and 90 percentile confidence interval from all unlabeled data.
    	However, when unlabeled data are few and sparse. It is misleading that the
    	90 percnetile of sampled unlabeled data distribution is the same as 90 percentile of
    	gobal confidence landscape of the whole feature space. In face np.percentile grabs the nearest
   		point to be the 90 percentile sample, even if that sample has negatvie confidence score.
    	By estimating the gobal confidence score distribution, there's no longer sign issue for the
    	confidence interval.

### RANSAC2 Models ###

RANSAC2 combines bagging and active learning. Initially 10 base classifiers were trainined separately with irreplacable random samples from the training data.

The rest of training data are considered unlableld.

Then the base classifiers use 'bagging' to generate concensus as pseudo labels. 

The psudolabels then are added to the sample sets in the next iteration of all base classifiers.

The process is repeated untill all unlabeled data are labeled.

#### ransac_simple.py 

First Version 

The models takes fully labeled data set where labels are 'noisy'. 

The SSL model then 'denoise' the data set in the fashion of 'bagging' and active learning.

#### ransac_simple_v2.py

Version 2 is developed on top of verison 1.
	
Improvements :
    
* Iterative training of the MLP:
        
		The original version only fit MLP with one training set, and reinstantiate a new MLP 
        every time new labels are added. This version the MLP is recycled through the label selection process.

* Max_count_pooling :
        
		The origianl version only take unanimous consensus as new labels. This version takes
        the absolute majority rule. Desition can be achived without unanimous agreement if  
        at least 50% of the estimators have conssensus. 
        
* Supports multiple epochs :
        
		After a fulll cycle of label spreading, the model can be trained futher by warm-starting the 
        label spreading with the trained MLP.     
		
#### ransac_simple_v3.py

Version 3 is developed on top of verison 2.

Improvements :

* Supports backward pass
       
	    After label spreading, all labels(including initial labels) are distilled to 
        remove controvesial labels.
		
#### ransac_simople_ssl_v3.py (alpha version)

This version 3 is developed on top of version 3. (fork form ransac_simple_v3.py)


Improvements :

* Forward-backward loop :
		
		The code is drastically modified to avoid some logical bugs in forward-backward loop that was not captured before. 
		It is considered easier to read, maintain, and scale.  

* Take unlabled data : 
		
		Random Concensus Algorityhm 2.3 utilizing unlabeled data. 
		
#### ransac_simple_new.py
	
Ransac Simple Algorithm2 (fork from ransac_simple.py)

RANSAC semisupervised learning utilizing unlabeled data.