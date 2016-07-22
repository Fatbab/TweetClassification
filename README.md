# TweetClassification
A quick python proof of concept to classify (Naïve Bayse) twitter texts (supervised learning). 

### Steps:
1. Get the tweets via API
2. Extract all words
3. Remove [Stop Words](http://streamhacker.com/2010/05/24/text-classification-sentiment-analysis-stopwords-collocations/)
4. Identify significant words by computing chi-square
5. Train and Test by Naïve Bayes classifier


### Little Theory Brush Up:
1. To evaluate performace, we've got the following measures:    
  * Accuracy: percentage of items in test set that the classifier correctly labeled.    
   (TP + TN)/(TP + TN + FP + FN)
  * Pos Precision: percentage of correctly labled positive from all cases the classifier labeled positive. (similarly for Neg Precision)     
   TP / (TP + FP) 
  * Pos Recall: percentage of correctly labled positive from all actually posivie cases. (similarly for Neg Recall)      
   TP / (TP + FN)    
  Where, TP: True Positive,   TN: True Negative,   FP: False Positive,   FN: False Negative.    

  Accuracy alone is not a good indicator of performance, for example when searching amongst huge pile of documents for one with specific features, it is very unlikely to find the correct document. So a classifer that always returns FALSE will get a high "Accuracy" whereas in fact it is not a good classifier. For such classifier we have TP=FP=0, FN = small number and TN = dataset size - FN. So Accuracy is a number very close to 100%.  It is easy to see how Precision = Recall = 0 for this classifier.    


2. To identify high information features, we've used [chi-squared](http://streamhacker.com/tag/chi-square/) as measure for information gain.      
  The idea is to identify words that appear primarily in one class and not so often in other classes. To use `BigramAssocMeasures.chi_sq` function, we need to calcualte overall and class-based word frequecy for every word and, the total number of words. 

### Coding Remarks:   
1. [Naive Bayes Classification] (https://en.wikipedia.org/wiki/Naive_Bayes_classifier "Wikipedia")
1. Tweepy
1. nltk.classify, nltk.metrics, nltk.probability, BigramAssocMeasures.chi_sq
1. [Nested Function Call](http://stackoverflow.com/questions/38512596/nested-function-calls-and-missing-input-parameter-python)

### Future Work:
1. Try other feature selection methods to replace make_full_dict. E.g. pair of words, x-many words before and after a keyword for a list of keywords, others(?).
1. Remove stopwords before counting frequencies.

### References:
1. [Connceting to Twitter API](http://adilmoujahid.com/posts/2014/07/twitter-analytics/)
1. [Twitter Text Mining](https://gist.github.com/yanofsky/5436496)
1. [Text Classification](https://github.com/abromberg/sentiment_analysis_python/blob/master/sentiment_analysis.py)
1. [Chi-Square for Eliminating Low Information Features](http://streamhacker.com/tag/chi-square/)
1. [Stopwords and Collocations](http://streamhacker.com/2010/05/24/text-classification-sentiment-analysis-stopwords-collocations/)
