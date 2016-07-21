# TweetClassification
A quick python proof of concept to classify (Naïve Bayse) twitter texts (supervised learning). 

### Steps:
1. Get the tweets via API
2. Extract words
3. Compute frequency of each word
5. Train and Test by Naïve Bayes classifier
4. Identify significant words by computing chi-square

### Theoretical Concepts:
1. To evaluate performace, we've got the following measures"    
  * Accuracy: percentage of items in test set that the classifier correctly labeled.    
   (TP + TN)/(TP + TN + FP + FN)
  * Precision: percentage of correctly labled positive from all cases the classifier labeled positive.    
   TP / (TP + FP) 
  * Recall: percentage of correctly labled positive from all actually posivie cases.     
   TP / (TP + FN)    

Accuracy alone is not a good indicator of performance, for example when searching amongst huge pile of documents for one with specific features, it is very unlikely to find the correct document. So a classifer that always returns FALSE will get a high "Accuracy" whereas in fact it is not a good classifier. For such classifier we have TP=FP=0, FN = small number and TN = dataset size - FN. So Accuracy is a number very close to 100%.  It is easy to see how Precision = Recall = 0 for this classifier.    

TP: True Positive,   TN: True Negative,   FP: False Positive,   FN: False Negative   

### New Coding Tips:   
1. [Naive Bayes Classification] (https://en.wikipedia.org/wiki/Naive_Bayes_classifier "Wikipedia")
1. Tweepy
1. nltk.classify, nltk.metrics, nltk.probability

### References:
1. [Connceting to Twitter API](http://adilmoujahid.com/posts/2014/07/twitter-analytics/)
1. [Twitter Text Mining](https://gist.github.com/yanofsky/5436496)
1. [Text Classification](https://github.com/abromberg/sentiment_analysis_python/blob/master/sentiment_analysis.py)
1. [Chi-Square for Eliminating Low Information Features](http://streamhacker.com/tag/chi-square/)
