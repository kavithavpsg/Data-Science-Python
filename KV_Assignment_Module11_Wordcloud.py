# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 06:54:46 2020

@author: KavithaV
"""

import requests
from bs4 import BeautifulSoup  # for web scraping

import matplotlib.pyplot as plt
from wordcloud import WordCloud

#header={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.90 Safari/537.36'}

# Creating empty review list
applewatch_reviews = []

for i in range(1,30):
    phone = []
    url = "https://www.amazon.in/Apple-Watch-GPS-44mm-Aluminium/dp/B07XWYJDDW/ref=sr_1_13?dchild=1&qid=1593212912&refinements=p_89%3AApple%2Cp_6%3AA14CZOWI0VEHLG%7CAT95IG9ONZD7S&s=electronics&sr=1-13#customerReviews"+str(i)
    header={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.90 Safari/537.36'}
    response = requests.get(url,headers = header)
    # creating soup object to iterate over the extracted content
    #soup = bs(response.content, "html.parser")
    soup = BeautifulSoup(response.text,"lxml")
    # Extract the content under the specific tag
    reviews = soup.find_all("div",{"data-hook":"review-collapsed"})
    for i in range(len(reviews)):
        phone.append(reviews[i].text)
    # Adding the reviews of one page to empty list which in future contains all the reviews
    applewatch_reviews = applewatch_reviews + phone
    
# Writing reviews in a text file
with open('applewatch.txt','w', encoding = 'utf8') as output:
    output.write(str(applewatch_reviews))

# Joinining all the reviews into single paragraph 
ip_rev_string = " ".join(applewatch_reviews)

import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
wordnet = WordNetLemmatizer()
nltk.download('wordnet')

# Removing unwanted symbols incase if exists
ip_rev_string = re.sub("[^A-Za-z" "]+"," ",ip_rev_string).lower()
ip_rev_string = re.sub("[0-9" "]+"," ",ip_rev_string)

# words that contained in iphone XR reviews
ip_reviews_words = ip_rev_string.split(" ")

#TFIDF
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer=TfidfVectorizer(ip_reviews_words,use_idf=True,ngram_range=(1, 3))
X=vectorizer.fit_transform(ip_reviews_words)

with open("D:/Data science/Module 11 - Text mining/Assignment/Assignment/stopwords.txt","r") as sw:
    stop_words = sw.read()

stop_words = stop_words.split("\n")

stop_words.extend(["watch","mobile","ios","apple","phone","amazon","good","xr","product","great","camera","price"])
ip_reviews_words = [w for w in ip_reviews_words if not w in stop_words]


# Joinining all the reviews into single paragraph 
ip_rev_string = " ".join(ip_reviews_words)

# WordCloud can be performed on the string inputs. Hence all the reviews are merged into single string
# Corpus word cloud

wordcloud_ip = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(ip_rev_string)
plt.figure(1)
plt.imshow(wordcloud_ip)

# positive words # Choose the path for +ve words stored in system
with open("D:/Data science/Module 11 - Text mining/Assignment/Assignment/positive-words.txt","r") as pos:
  poswords = pos.read().split("\n")
  

# Positive word cloud
# Choosing the only words which are present in positive words
ip_pos_in_pos = " ".join ([w for w in ip_reviews_words if w in poswords])
wordcloud_pos_in_pos = WordCloud(
                      background_color='White',
                      width=1800,
                      height=1400
                     ).generate(ip_pos_in_pos)

plt.figure(2)
plt.imshow(wordcloud_pos_in_pos)

  
# negative words  Choose path for -ve words stored in system
with open("D:/Data science/Module 11 - Text mining/Assignment/Assignment/negative-words.txt","r") as neg:
  negwords = neg.read().split("\n")

# negative word cloud
# Choosing the only words which are present in negwords
ip_neg_in_neg = " ".join ([w for w in ip_reviews_words if w in negwords])

wordcloud_neg_in_neg = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(ip_neg_in_neg)
plt.figure(3)
plt.imshow(wordcloud_neg_in_neg)


