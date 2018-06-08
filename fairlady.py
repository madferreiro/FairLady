import nltk
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from string import punctuation
from nltk.collocations import *
from nltk.stem.lancaster import LancasterStemmer
from bs4 import BeautifulSoup
from collections import defaultdict
from heapq import nlargest

import urllib2
import codecs

text = 'Although we must be extremely cautious in concluding that any organ could not have been produced by successive, small, transitional gradations, yet undoubtedly serious cases of difficulty occur. One of the most serious is that of neuter insects, which are often differently constructed from either the males or fertile females; but this case will be treated of in the next chapter. The electric organs of fishes offer another case of special difficulty; for it is impossible to conceive by what steps these wondrous organs have been produced. But this is not surprising, for we do not even know of what use they are. In the gymnotus and torpedo they no doubt serve as powerful means of defence, and perhaps for securing prey; yet in the ray, as observed by Matteucci, an analogous organ in the tail manifests but little electricity, even when the animal is greatly irritated; so little that it can hardly be of any use for the above purposes. Moreover, in the ray, besides the organ just referred to, there is, as Dr. R. McDonnell has shown, another organ near the head, not known to be electrical, but which appears to be the real homologue of the electric battery in the torpedo. It is generally admitted that there exists between these organs and ordinary muscle a close analogy, in intimate structure, in the distribution of the nerves, and in the manner in which they are acted on by various reagents. It should, also, be especially observed that muscular contraction is accompanied by an electrical discharge; and, as Dr. Radcliffe insists, "in the electrical apparatus of the torpedo during rest, there would seem to be a charge in every respect like that which is met with in muscle and nerve during the rest, and the discharge of the torpedo, instead of being peculiar, may be only another form of the discharge which attends upon the action of muscle and motor nerve." Beyond this we cannot at present go in the way of explanation; but as we know so little about the uses of these organs, and as we know nothing about the habits and structure of the progenitors of the existing electric fishes, it would be extremely bold to maintain that no serviceable transitions are possible by which these organs might have been gradually developed.'

print('Part 1: Tools')

print('Part 1A: Tokenizing by sentence')
sents=sent_tokenize(text)
print(sents)

print('Part 1B: Tokenizing by words')
words = [ word_tokenize(sent) for sent in sents ]
print(words)


print('Part 1C: Removing stopwords')

customStopWords = set(stopwords.words('english')+list(punctuation))
wordsWOStopwords = [ word for word in word_tokenize(text) if word not in customStopWords ]
print(wordsWOStopwords)


print('Part 1D: Identify Bigrams')

#bigram_measures = nltk.collocations.BigramAssocMeasures()
finder = BigramCollocationFinder.from_words(wordsWOStopwords)
print sorted(finder.ngram_fd.items())


print('Part 1E: Stemmization')
st = LancasterStemmer()
stemmedWords = [ st.stem(word) for word in wordsWOStopwords ]
print(stemmedWords)


print('Part 1F: Part-of-Speech(POS) Tagging')
taggedWords = nltk.pos_tag(wordsWOStopwords)
print(taggedWords)

print('Part 2: Sumarization Techniques')

with codecs.open("hugetext.txt", encoding="utf-8") as inp:
    rawText = ( ' '.join( inp.read().splitlines() ) ).encode('ascii', errors='replace').replace('?', ' ')

print('Part 2A: Preprocessing text')

summarySents = sent_tokenize(rawText)
summaryStopWords = set(stopwords.words('english')+list(punctuation))
summaryWords = [ word for word in word_tokenize(rawText) if word not in summaryStopWords ]

print('Part 2B: Count word frequency')
freq = FreqDist(summaryWords)
print(freq.most_common(10))

print('Part 2C: Rank most valuable sentences')
ranking = defaultdict(int)

for i, summarySents in enumerate( summarySents ):
    for w in word_tokenize( summarySents.lower() ): 
        if( w in freq ):
            ranking[i] += freq[w]
unrankedSents = sent_tokenize(rawText)            
rankedSents = map(lambda x: unrankedSents[x], nlargest(4, ranking, key=ranking.get))
print(rankedSents)
