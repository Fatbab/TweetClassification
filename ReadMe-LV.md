# Linguistic Visualisation

Visualizing textual material requires deciding on a numeric measure that would adequately represent one text and can also be applied on other samples to provide a meaningful mean for comparison between multiple texts.  
One way to do this is to reduce every sentence to its underlysing linguistic structure. For example, “The cat sat on the mat” is transformed to this vector “<NP, VP, PP, NP>” where, NP: Noun Phrase, VP: Verb Phrase, PP: Proposition Phrase. 
Now, comparing similarity between vectors is not a difficult task and similarly, a numeric measure can be assigned to each sentence. Python's `pattern.en` packge includes the `parsetree` method which takes care of this transformation.

My work in [LinguisticVisulaization.py](https://github.com/Fatbab/TweetClassification/blob/master/LinguisticVisulaization.py) is an attempt to replicate [this sample work](http://nbviewer.jupyter.org/github/AYLIEN/headline_analysis/blob/06f1223012d285412a650c201a19a1c95859dca1/main-chunks.ipynb#A-primer-on-parse-trees), where we learn to diffrentiate between and visualize the headlines written by two different journalists. 


to be completed ...
