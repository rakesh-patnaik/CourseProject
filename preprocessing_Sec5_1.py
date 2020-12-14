#!/usr/bin/env python

# This script implements section 5.1 of
# Hongning Wang, Yue Lu, and ChengXiang Zhai. 2011. Latent aspect rating analysis without aspect keyword supervision.
# In Proceedings of ACM KDD 2011, pp. 618-626. DOI=10.1145/2020408.2020505
# https://www.cs.virginia.edu/~hw5x/paper/p618.pdf
# 5.1 Data Sets and Preprocessing
# 1) remove the reviews with any missing aspect rating or document length less than 50 words (to keep the content coverage of all possible aspects);
# 2) convert all the words into lower cases; and
# 3) removing punctuations, stop words defined in [1], and the terms occurring in less than 10 reviews in the collection

import nltk
from nltk.corpus import stopwords
from collections import defaultdict
import json
import os
import re
import string

RAW_JSON_DATA_DIR = 'data/TripAdvisorData/raw_JSON'
CLEANED_JSON_DATA_DIR = 'data/TripAdvisorData/CleanData_JSON'
RATING_ASPECTS = ["Service", "Cleanliness", "Overall", "Value", "Location", "Rooms", "Sleep Quality"]

def prepareStopWords():
    stopwordsList = stopwords.words('english')
    stopwordsList.append('review')
    stopwordsList.append('dont')
    stopwordsList.append('didnt')
    stopwordsList.append('doesnt')
    stopwordsList.append('cant')
    stopwordsList.append('couldnt')
    stopwordsList.append('couldve')
    stopwordsList.append('im')
    stopwordsList.append('ive')
    stopwordsList.append('isnt')
    stopwordsList.append('theres')
    stopwordsList.append('wasnt')
    stopwordsList.append('wouldnt')
    stopwordsList.append('a')
    return stopwordsList

def preprocess_text(rawText):
    # Lowercase and tokenize
    rawText = rawText.lower()
    # Remove single quote early since it causes problems with the tokenizer.
    rawText = rawText.replace("'", "")

    tokens = nltk.word_tokenize(rawText)
    text = nltk.Text(tokens)

    # Load default stop words and add a few more.
    stopWords = prepareStopWords()

    # Remove extra chars and remove stop words.
    text_content = [''.join(re.split("[ .,;:!?‘’``''@#$%^_&*()<>{}~\n\t\\\-]", word)) for word in text]

    text_content = [word for word in text_content if (word not in stopWords and word not in string.punctuation)]

    # After the punctuation above is removed it still leaves empty entries in the list.
    # Remove any entries where the len is zero.
    text_content = [s for s in text_content if len(s) != 0]

    WNL = nltk.WordNetLemmatizer()
    text_content = [WNL.lemmatize(t) for t in text_content]

    return " ".join(text_content)

raw_datafiles = os.listdir(RAW_JSON_DATA_DIR)

hotel_dict_list = []
for raw_datafile in raw_datafiles:
    with open(RAW_JSON_DATA_DIR + '/' + raw_datafile) as raw_hoteldata_file:
        hotel_data_dict = json.load(raw_hoteldata_file)
        hotel_dict_list.append(hotel_data_dict)

word_frequency_counter = defaultdict(int)

for hotel_dict in hotel_dict_list:
    selected_reviews = []
    for review in hotel_dict['Reviews']:
        try:
          if 'Content' not in review or len(review['Content'].split()) < 50:
            continue

          for rating_aspect in RATING_ASPECTS:
            rating_value = float(review['Ratings'][rating_aspect])
            review['Ratings'][rating_aspect] = rating_value
        except:
            continue
        review['Content'] = preprocess_text(review['Content'])
        selected_reviews.append(review)

        for word in set(review['Content']):
            word_frequency_counter[word] += 1

    hotel_dict['Reviews'] = selected_reviews


words_more_than_10_occurances = { k: v for k, v in word_frequency_counter.items() if v >= 10 }
word_to_keep = set(words_more_than_10_occurances.keys())
for hotel_dict in hotel_dict_list:
    selected_reviews =  []
    for review in hotel_dict['Reviews']:
        content = ' '.join(list(filter(lambda x: x in word_to_keep, review['Content'])))
        if len(content.strip()) > 0:
            selected_reviews.append(review)

    hotel_dict['Reviews'] = selected_reviews

for hotel_dict in hotel_dict_list:
    if len(hotel_dict['Reviews']) == 0:
        hotel_dict_list.remove(hotel_dict)

for hotel_dict in hotel_dict_list:
    hotel_id = hotel_dict['HotelInfo']['HotelID']
    with open(CLEANED_JSON_DATA_DIR + '/' +  hotel_id + '.json', 'w+') as clean_data_file:
        json.dump(hotel_dict, clean_data_file, default=str, indent=2)