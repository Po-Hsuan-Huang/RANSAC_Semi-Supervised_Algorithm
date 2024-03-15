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
		
