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
