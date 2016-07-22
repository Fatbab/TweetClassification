## Text Classification - Supervised Learning
## Goal is to train the classifier to distinguish between Donald Trump and 'other' tweets. 
## Training set: 200 tweets from Donald Trump, 100 from Bill Clinton and 100 from Adele's account.
## Test on a new collection of 100 tweets from Donald Trump, 50 from Bill Clinton and 50 from Adele's.

import tweepy
import csv 
import json
import pandas as pd
import matplotlib.pyplot as plt
import re, math, collections, itertools, os
import nltk, nltk.classify.util, nltk.metrics
from nltk.classify import NaiveBayesClassifier
from nltk.metrics import BigramAssocMeasures
from nltk.metrics import scores
from nltk.metrics.scores import precision
from nltk.metrics.scores import recall
from nltk.probability import FreqDist, ConditionalFreqDist

## Connect to Twitter API
## Reference: http://adilmoujahid.com/posts/2014/07/twitter-analytics/
## Variables that contains user credentials to access Twitter API
## Replace with valid credentials
access_token = "" # "ENTER YOUR ACCESS TOKEN"
access_token_secret = "" # "ENTER YOUR ACCESS TOKEN SECRET"
consumer_key = "" # "ENTER YOUR API KEY"
consumer_secret = "" # "ENTER YOUR API SECRET"

## Script to get tweets for certain account
## Reference: https://gist.github.com/yanofsky/5436496
def get_all_tweets(screen_name):
	# Twitter only allows access to a users most recent 3240 tweets with this method
	# Authorize twitter, initialize tweepy
	auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
	auth.set_access_token(access_token, access_token_secret)
	api = tweepy.API(auth)

	# Initialize a list to hold all the tweepy Tweets
	alltweets = []

	# Make initial request for most recent tweets (200 is the maximum allowed count)
	new_tweets = api.user_timeline(screen_name = screen_name, count=200)

	# Save most recent tweets
	alltweets.extend(new_tweets)

	# Save the id of the oldest tweet less one
	oldest = alltweets[-1].id - 1

	# Keep grabbing tweets until there are no tweets left to grab
	while len(new_tweets) > 0:
		print "getting tweets before %s" % (oldest)

		# All subsiquent requests use the max_id param to prevent duplicates
		new_tweets = api.user_timeline(screen_name = screen_name,count=200,max_id=oldest)

		# Save most recent tweets
		alltweets.extend(new_tweets)

		# Update the id of the oldest tweet less one
		oldest = alltweets[-1].id - 1

		print "...%s tweets downloaded so far" % (len(alltweets))

	# Transform the tweepy tweets into a 2D array that will populate the csv
	outtweets = [[tweet.id_str, tweet.created_at, tweet.text.encode("utf-8")] for tweet in alltweets]

	# Write to csv generically
	with open('/PATH/TO/WRITE/%s_tweets.csv' % screen_name, 'wb') as f:
		writer = csv.writer(f)
		writer.writerow(["id","created_at","text"])
		writer.writerows(outtweets)

	pass


get_all_tweets("realDonaldTrump")
get_all_tweets("billclinton")
get_all_tweets("Adele")

## Read in the tweets
Adele =[]
with open('/PATH/TO/READ/Adele_tweets.csv', 'rb') as f:
    next(f) # skip header line
    reader = csv.reader(f,delimiter=',')
    for row in reader:
        Adele.append(row[2])

Clinton =[]
with open('/PATH/TO/READ/billclinton_tweets.csv', 'rb') as f:
    next(f) # skip header line
    reader = csv.reader(f,delimiter=',')
    for row in reader:
        Clinton.append(row[2])

Trump =[]
with open('/PATH/TO/READ/realDonaldTrump_tweets.csv', 'rb') as f:
    next(f) # skip header line
    reader = csv.reader(f,delimiter=',')
    for row in reader:
        Trump.append(row[2])

## Training set: 200 tweets from Donald Trump, 100 from Bill Clinton and 100 from Adele
## Divide the set into training n test subsets
Adele_train = Adele[0:100]
Adele_test = Adele[101:151]

Clinton_train = Clinton[0:100]
Clinton_test = Clinton[101:151]

Trump_train = Trump[0:200]
Trump_test = Trump[201:301]

## TEXT ANALYSIS 
## https://github.com/abromberg/sentiment_analysis_python/blob/master/sentiment_analysis.py

def evaluate_features(feature_select):
	## Label all Trump tweets with 'pos' and other tweets with 'neg'
	## Divide them into Train and Test subset 
	posFeatures_train =[]
	negFeatures_train =[]

	for i in Trump_train:
		posWords = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
		posWords = [feature_select(posWords), 'pos']
		posFeatures_train.append(posWords)

	for i in Adele_train + Clinton_train:
		negWords = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
		negWords = [feature_select(negWords), 'neg']
		negFeatures_train.append(negWords)

	posFeatures_test = []
	negFeatures_test = []

	for i in Trump_test:
		posWords = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
		posWords = [feature_select(posWords), 'pos']
		posFeatures_test.append(posWords)

	for i in Adele_test + Clinton_test:
		negWords = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
		negWords = [feature_select(negWords), 'neg']
		negFeatures_test.append(negWords)

	trainFeatures = posFeatures_train + negFeatures_train
	testFeatures = posFeatures_test + negFeatures_test

	## Trains a Naive Bayes Classifier
	## Read more here: https://en.wikipedia.org/wiki/Naive_Bayes_classifier
	classifier = NaiveBayesClassifier.train(trainFeatures)

	## Initiates referenceSets and testSets
	referenceSets = collections.defaultdict(set)
	testSets = collections.defaultdict(set)

	## Puts correctly labeled sentences in referenceSets and the predictively labeled version in testsets
	for i, (features, label) in enumerate(testFeatures):
		referenceSets[label].add(i)
		predicted = classifier.classify(features)
		testSets[predicted].add(i)

	## Prints metrics to show how well the feature selection did
	## Accuracy: percentage of items in test set that the classifier correctly labeled.
	## Precision: True_Positive / (True_Positive+False_Positive) 
	## Recall: True_Positive / (True_Positive+False_Negative) 
	print 'train on %d instances, test on %d instances' % (len(trainFeatures), len(testFeatures))
	print 'accuracy:', nltk.classify.util.accuracy(classifier, testFeatures)
	print 'pos precision:', precision(referenceSets['pos'], testSets['pos'])
	print 'pos recall:', recall(referenceSets['pos'], testSets['pos'])
	print 'neg precision:', precision(referenceSets['neg'], testSets['neg'])
	print 'neg recall:', recall(referenceSets['neg'], testSets['neg'])
	classifier.show_most_informative_features(10)

## Creates a feature selection mechanism that uses all words
## This function is passed on to evaluate_featues as input. One can change this to select every pair of words ot threesome, etc.
def make_full_dict(words):
	return dict([(word, True) for word in words])

## Using all words as the feature selection mechanism
print 'using all words as features'
evaluate_features(make_full_dict)

## Once word selection is done, we move onto identifying most informative features
## Scores words based on chi-squared test to show information gain 
## (http://streamhacker.com/2010/06/16/text-classification-sentiment-analysis-eliminate-low-information-features/)
def create_word_scores():
	posWords = []
	negWords = []

	for i in Trump:
			posWord = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
			posWords.append(posWord)

	for i in Adele+Clinton:
			negWord = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
			negWords.append(negWord)
	posWords = list(itertools.chain(*posWords))
	negWords = list(itertools.chain(*negWords))

	## Frequency Distibution: number of times each word has appeared in input list
	## Conditional Frequency Distribution: counts word frequency by genre, (genre, word). In this case genre = pos, neg.
	## Essentially counts the number of time a word has appeared paired with pos or neg tag, for every single word in list.
	## http://www.nltk.org/book/ch02.html#sec-conditional-frequency-distributions
	## Note that we cannot just borrow from pre-defined genres. We need to train based on our definition of "pos"/"neg"   
	word_fd = FreqDist()
	cond_word_fd = ConditionalFreqDist()
	for word in posWords:
		word_fd[word.lower()] += 1
		cond_word_fd['pos'][word.lower()] += 1
	for word in negWords:
		word_fd[word.lower()] += 1
		cond_word_fd['neg'][word.lower()] += 1

	## Finds the number of positive and negative words, as well as the total number of words
	## Wee need these elements for computing chi-sq 
	pos_word_count = cond_word_fd['pos'].N()
	neg_word_count = cond_word_fd['neg'].N()
	total_word_count = pos_word_count + neg_word_count

	## Builds dictionary of word scores based on chi-squared test
	## http://streamhacker.com/tag/chi-square/
	word_scores = {}
	for word, freq in word_fd.iteritems():
		pos_score = BigramAssocMeasures.chi_sq(cond_word_fd['pos'][word], (freq, pos_word_count), total_word_count)
		neg_score = BigramAssocMeasures.chi_sq(cond_word_fd['neg'][word], (freq, neg_word_count), total_word_count)
		word_scores[word] = pos_score + neg_score

	return word_scores

## Finds word scores
word_scores = create_word_scores()

## Finds the best 'number' of words based on word scores
def find_best_words(word_scores, number):
	best_vals = sorted(word_scores.iteritems(), key=lambda (w, s): s, reverse=True)[:number]
	best_words = set([w for w, s in best_vals])
	return best_words

## Creates feature selection mechanism that only uses best words
def best_word_features(words):
	return dict([(word, True) for word in words if word in best_words])

## List for top 'numbers' words after feature selection
numbers_to_test = [10, 100, 1000, 10000, 15000]
## Tries the best_word_features mechanism with each of the numbers_to_test of features
for num in numbers_to_test:
	print 'evaluating best %d word features' % (num)
	best_words = find_best_words(word_scores, num)
	evaluate_features(best_word_features)
