import csv
import nltk
import re
from nltk.corpus import stopwords
tweets=[]
with open('C:/users/purusoth/desktop/trainingset2000.csv','rb') as f:
    reader=csv.reader(f)
    for row in reader:
        tweets.append(row)
#print tweets
tweetssenti=[]
for (a,b,c,d,e,f) in tweets:
    words_filtered=[e.lower() for e in f.split() if len(e)>=3]
    for i in words_filtered:
        if i in stopwords.words('english'):
            #print i
            words_filtered.remove(i)
    tweetssenti.append((words_filtered,a))
#print tweetssenti
print 'tweent senti done'

def get_words_in_tweets(tweetssenti):
    all_words = []
    for (words, sentiment) in tweetssenti:
      all_words.extend(words)
    return all_words
def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    #print wordlist
    word_features = wordlist.keys()
    return word_features
word_features = get_word_features(get_words_in_tweets(tweetssenti))
print 'word_features done'
#print word_features

def extract_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
        
    return features
print 'obtaining training set'   
training_set = nltk.classify.util.apply_features(extract_features, tweetssenti)
#print training_set
print 'training the classfier .....'
classifier = nltk.NaiveBayesClassifier.train(training_set)
print ' Trained!'
print classifier.show_most_informative_features(200)
def processTweet(tweet):
    # process the tweets

    #Convert to lower case
    tweet = tweet.lower()
    #Convert www.* or https?://* to URL
    tweet = re.sub('((www\.[\s]+)|(https?://[^\s]+))','',tweet)
    #Convert @username to AT_USER
    tweet = re.sub('@[^\s]+','',tweet)
    #Remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)
    #Replace #word with word
    tweet = re.sub(r'#([^\s]+)', '', tweet)
    #trim
    tweet = tweet.strip('\'"')
    return tweet
#end

test_tweet=[]
with open('C:/Users/Purusoth/Desktop/movies/gijoe.csv','rb') as s:
    reader=csv.reader(s)
    for row in reader:
       test_tweet.append(row)
    print test_tweet
testing=[]
for (d,e,f,g,h) in test_tweet:
    #print d
    #print 'preprocessing'
    filtered=processTweet(e)
    tweet_test=filtered.split()
    for i in tweet_test:
        if i in stopwords.words('english'):
            print i
            tweet_test.remove(i)
    #print filtered
    testing.append(tweet_test)
for tw in testing:
    print tw
    print classifier.classify(extract_features(tw))

