## Text Classification - Supervised Learning
## Will Get 100 tweets from Donald Trump, Bill Clinton and Adele's account as
## training set (Trump YES/NO).
## Analysis text and defines identifying attributes for each.
## Test on a mix of 50 tweets randomly chosen from same accounts, but not been
## trained upon.

## Connect to Twitter API
## Reference: http://adilmoujahid.com/posts/2014/07/twitter-analytics/
#Import the necessary methods from tweepy library

import tweepy
#from tweepy.streaming import StreamListener
#from tweepy import OAuthHandler
#from tweepy import Stream

# Import libraries for text mining
import csv
import json
import pandas as pd
import matplotlib.pyplot as plt

#Variables that contains user credentials to access Twitter API
## Replace with valid credentials
access_token = "1171035650-l3MAgBH6peyUQoePBgivrooZMvrVxv85D8RcI6R" # "ENTER YOUR ACCESS TOKEN"
access_token_secret = "v5duEnw3kSN3B4goTXAF8byYTlB07uVBZ65uQEXyRNxbk" # "ENTER YOUR ACCESS TOKEN SECRET"
consumer_key = "G7K5UhTvtqo5OrYMusNk5WPGM" # "ENTER YOUR API KEY"
consumer_secret = "hlNoZOrDKCy1r40nHuqGK7HVa92qcGcPpVZZ2tsv8Qslneqzwn" # "ENTER YOUR API SECRET"

## Script to get tweets for certain account
## Reference: https://gist.github.com/yanofsky/5436496
def get_all_tweets(screen_name):
	#Twitter only allows access to a users most recent 3240 tweets with this method

	#authorize twitter, initialize tweepy
	auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
	auth.set_access_token(access_token, access_token_secret)
	api = tweepy.API(auth)

	#initialize a list to hold all the tweepy Tweets
	alltweets = []

	#make initial request for most recent tweets (200 is the maximum allowed count)
	new_tweets = api.user_timeline(screen_name = screen_name,count=200)

	#save most recent tweets
	alltweets.extend(new_tweets)

	#save the id of the oldest tweet less one
	oldest = alltweets[-1].id - 1

	#keep grabbing tweets until there are no tweets left to grab
	while len(new_tweets) > 0:
		print "getting tweets before %s" % (oldest)

		#all subsiquent requests use the max_id param to prevent duplicates
		new_tweets = api.user_timeline(screen_name = screen_name,count=200,max_id=oldest)

		#save most recent tweets
		alltweets.extend(new_tweets)

		#update the id of the oldest tweet less one
		oldest = alltweets[-1].id - 1

		print "...%s tweets downloaded so far" % (len(alltweets))

	#transform the tweepy tweets into a 2D array that will populate the csv
	outtweets = [[tweet.id_str, tweet.created_at, tweet.text.encode("utf-8")] for tweet in alltweets]

	#write the csv
	with open('/Users/fatemeh/Documents/Career/Concentra/TextAnalysis-SampleWork/%s_tweets.csv' % screen_name, 'wb') as f:
		writer = csv.writer(f)
		writer.writerow(["id","created_at","text"])
		writer.writerows(outtweets)

	pass


#if __name__ == '__main__':
	#pass in the username of the account you want to download
get_all_tweets("realDonaldTrump")
get_all_tweets("billclinton")
get_all_tweets("Adele")

## Training set: 200 tweets from Donald Trump, 100 from Bill Clinton and 100 from Adele
Adele =[]
with open('/Users/fatemeh/Documents/Career/Concentra/TextAnalysis-SampleWork/Adele_tweets.csv', 'rb') as f:
    next(f) # skip header line
    reader = csv.reader(f,delimiter=',')
    for row in reader:
        Adele.append(row[2])

Clinton =[]
with open('/Users/fatemeh/Documents/Career/Concentra/TextAnalysis-SampleWork/billclinton_tweets.csv', 'rb') as f:
    next(f) # skip header line
    reader = csv.reader(f,delimiter=',')
    for row in reader:
        Clinton.append(row[2])

Trump =[]
with open('/Users/fatemeh/Documents/Career/Concentra/TextAnalysis-SampleWork/realDonaldTrump_tweets.csv', 'rb') as f:
    next(f) # skip header line
    reader = csv.reader(f,delimiter=',')
    for row in reader:
        Trump.append(row[2])

# Divide the set into training n test subsets
Adele_train = Adele[0:100]
Adele_test = Adele[101:151]

Clinton_train = Clinton[0:100]
Clinton_test = Clinton[101:151]

Trump_train = Trump[0:200]
Trump_test = Trump[201:301]

## https://github.com/abromberg/sentiment_analysis_python/blob/master/sentiment_analysis.py
import re, math, collections, itertools, os
import nltk, nltk.classify.util, nltk.metrics
from nltk.classify import NaiveBayesClassifier
from nltk.metrics import BigramAssocMeasures
from nltk.metrics import scores
from nltk.metrics.scores import precision
from nltk.metrics.scores import recall
from nltk.probability import FreqDist, ConditionalFreqDist

def evaluate_features(feature_select):
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

	#print 'testFeatures: %s' %testFeatures
	print 'enumerate(testFeatures): %s' %enumerate(testFeatures)
	print 'testFeatures[:5]: %s' %testFeatures[:5]
	#trains a Naive Bayes Classifier
	classifier = NaiveBayesClassifier.train(trainFeatures)

	#initiates referenceSets and testSets
	referenceSets = collections.defaultdict(set)
	testSets = collections.defaultdict(set)

	#puts correctly labeled sentences in referenceSets and the predictively labeled version in testsets
	for i, (features, label) in enumerate(testFeatures):
		referenceSets[label].add(i)
		predicted = classifier.classify(features)
		testSets[predicted].add(i)

	#prints metrics to show how well the feature selection did
	## precision: fraction of test values that appear in the reference set
	## recall: fraction of reference values that appear in the test set
	print 'train on %d instances, test on %d instances' % (len(trainFeatures), len(testFeatures))
	print 'accuracy:', nltk.classify.util.accuracy(classifier, testFeatures)
	print 'pos precision:', precision(referenceSets['pos'], testSets['pos'])
	print 'pos recall:', recall(referenceSets['pos'], testSets['pos'])
	print 'neg precision:', precision(referenceSets['neg'], testSets['neg'])
	print 'neg recall:', recall(referenceSets['neg'], testSets['neg'])
	classifier.show_most_informative_features(10)

#creates a feature selection mechanism that uses all words
def make_full_dict(words):
	return dict([(word, True) for word in words])

#tries using all words as the feature selection mechanism
print 'using all words as features'
evaluate_features(make_full_dict)


#scores words based on chi-squared test to show information gain (http://streamhacker.com/2010/06/16/text-classification-sentiment-analysis-eliminate-low-information-features/)
def create_word_scores():
	#creates lists of all positive and negative words
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

	#build frequency distibution of all words and then frequency distributions of words within positive and negative labels
	word_fd = FreqDist()
	cond_word_fd = ConditionalFreqDist()
	for word in posWords:
		word_fd[word.lower()] += 1
		cond_word_fd['pos'][word.lower()] += 1
	for word in negWords:
		word_fd[word.lower()] += 1
		cond_word_fd['neg'][word.lower()] += 1

	#finds the number of positive and negative words, as well as the total number of words
	pos_word_count = cond_word_fd['pos'].N()
	neg_word_count = cond_word_fd['neg'].N()
	total_word_count = pos_word_count + neg_word_count

	#builds dictionary of word scores based on chi-squared test
	word_scores = {}
	for word, freq in word_fd.iteritems():
		pos_score = BigramAssocMeasures.chi_sq(cond_word_fd['pos'][word], (freq, pos_word_count), total_word_count)
		neg_score = BigramAssocMeasures.chi_sq(cond_word_fd['neg'][word], (freq, neg_word_count), total_word_count)
		word_scores[word] = pos_score + neg_score

	return word_scores

#finds word scores
word_scores = create_word_scores()
print word_scores

#finds the best 'number' words based on word scores
def find_best_words(word_scores, number):
	best_vals = sorted(word_scores.iteritems(), key=lambda (w, s): s, reverse=True)[:number]
	best_words = set([w for w, s in best_vals])
	return best_words

#creates feature selection mechanism that only uses best words
def best_word_features(words):
	return dict([(word, True) for word in words if word in best_words])

#numbers of features to select
numbers_to_test = [10, 100, 1000, 10000, 15000]
#tries the best_word_features mechanism with each of the numbers_to_test of features
for num in numbers_to_test:
	print 'evaluating best %d word features' % (num)
	best_words = find_best_words(word_scores, num)
	evaluate_features(best_word_features)
