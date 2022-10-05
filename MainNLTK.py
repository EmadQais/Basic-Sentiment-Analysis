import string
from collections import Counter

import matplotlib.pyplot as plt
import nltk

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import RegexpTokenizer, PorterStemmer
from nltk.corpus import stopwords

# nltk.download('stopwords')

# text = open('opinions.txt', encoding='utf-8').read()
text = open('Reviews.json', encoding='utf-8').read()
lower_case: str = text.lower()
cleaned_text = lower_case.translate(str.maketrans('', '', string.punctuation))

# Using Tokenize for split the sentences to the words
words = nltk.wordpunct_tokenize(cleaned_text)

# Removing Stop Words
tokens_without_sw = [word for word in words if not word in stopwords.words()]


# Stemming : is the process of reducing inflection in words to their root
ps = PorterStemmer()
finalWord = []
for w in tokens_without_sw:
    rootWord = ps.stem(w)
    finalWord.append(rootWord)

# print(finalWord)

emotion_list = []
with open('emotions.txt', 'r') as file:
    for line in file:
        clear_line = line.replace("\n", '').replace(",", '').replace("'", '').strip()
        word, emotion = clear_line.split(':')
        # print("Word :" + word + " " + "Emotion :" + emotion)
        # print(word)
        if word in finalWord:
            emotion_list.append(emotion)


print(emotion_list)
w = Counter(emotion_list)
print(w)


def sentiment_analyse(sentiment_text):
    # print(sentiment_text)
    score = SentimentIntensityAnalyzer().polarity_scores(sentiment_text)
    print(score)
    if score['neg'] > score['pos']:
        print("Negative Sentiment")
    elif score['neg'] < score['pos']:
        print("Positive Sentiment")
    else:
        print("Neutral Sentiment")


sentiment_analyse(cleaned_text)

fig, ax1 = plt.subplots()
ax1.bar(w.keys(), w.values())
fig.autofmt_xdate()
plt.savefig('graph.png')
plt.show()
