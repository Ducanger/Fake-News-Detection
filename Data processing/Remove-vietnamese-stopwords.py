# import lib
import pandas as pd
import re

# data after word_tokenize by underthesea
df = pd.read_csv(r'D:\Dat\Fake News Detection\Data\word-tokenize-data.csv') 

# clean data
df.drop(['stt','id','timestamp_post'], axis='columns', inplace=True)
df.post_message = df.post_message.str.replace("[0-9]","")
df.post_message = df.post_message.str.replace("[`~!@#$%^&*()-=?+><;:,.{}']","")
df.post_message = df.post_message.str.replace(" +"," ")
df.post_message = df.post_message.str.replace("_+","_")

# import vietnamese-stopwords
import io
f = io.open(r'D:\Dat\Fake News Detection\Code\vietnamese-stopwords.txt', "r", encoding="utf-8")
vn = f.read().split()

# remove vietnamese-stopwords
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk

ps = PorterStemmer()
corpus = []

for i in range (0, len(df)):
    review = df.post_message[i]
    review = review.split()
    review = [ps.stem(word) for word in review if not word in vn] 
    review = ' '.join(review)
    corpus.append(review)
df.post_message = corpus

# drop empty value
nan_value = float("NaN")
df.replace("", nan_value, inplace=True)
df = df.dropna(subset = ["post_message"], inplace=True)

# export file
df.to_csv(r"D:\Dat\Fake News Detection\Data\pre-train-data.csv")
