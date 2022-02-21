# Question-Pair-Text-Similarity

## Overview
The goal of this project is to predict which of the provided pairs of questions contain two questions with the same meaning. This could be useful to instantly provide answers to questions that have already been answered. We are tasked with predicting whether a pair of questions are duplicates or not

## Motivation
Over 100 million people visit Quora every month, so it's no surprise that many people ask similarly worded questions. Multiple questions with the same intent can cause seekers to spend more time finding the best answer to their question, and make writers feel they need to answer multiple versions of the same question. Quora values canonical questions because they provide a better experience to active seekers and writers, and offer more value to both of these groups in the long term.

## Data Overview
Data Source : https://www.kaggle.com/c/quora-question-pairs

Train.csv contains 5 columns:
- id - the id of a training set question pair
- qid1, qid2 - unique ids of each question (only available in train.csv)
- question1, question2 - the full text of each question
- is_duplicate - the target variable, set to 1 if question1 and question2 have essentially the same meaning, and 0 otherwise.


## New Feature Extraction

Basic Features - Extracted some features after cleaning of data as below.

- freq_qid1 = Frequency of qid1's
- freq_qid2 = Frequency of qid2's
- q1len = Length of q1
- q2len = Length of q2
- q1_n_words = Number of words in Question 1
- q2_n_words = Number of words in Question 2
- word_Common = (Number of common unique words in Question 1 and Question 2)
- word_Total =(Total num of words in Question 1 + Total num of words in Question 2)
- word_share = (word_common)/(word_Total)
- freq_q1+freq_q2 = sum total of frequency of qid1 and qid2
- freq_q1-freq_q2 = absolute difference of frequency of qid1 and qid2

Advanced Features - Did some preprocessing of texts and extracted some other features. i am giving some definitions which are used below. Token- You get a token by splitting sentence by space , Stop_Word - stop words as per NLTK, Word -A token that is not a stop_word.

- cwc_min = common_word_count / (min(len(q1_words), len(q2_words))
- cwc_max = common_word_count / (max(len(q1_words), len(q2_words))
- csc_min = common_stop_count / (min(len(q1_stops), len(q2_stops))
- csc_max = common_stop_count / (max(len(q1_stops), len(q2_stops))
- ctc_min = common_token_count / (min(len(q1_tokens), len(q2_tokens))
- ctc_max = common_token_count / (max(len(q1_tokens), len(q2_tokens))
- last_word_eq = Check if Last word of both questions is equal or not (int(q1_tokens[-1] == q2_tokens[-1]))
- first_word_eq = Check if First word of both questions is equal or not (int(q1_tokens[0] == q2_tokens[0]) )
- abs_len_diff = abs(len(q1_tokens) - len(q2_tokens))
- mean_len = (len(q1_tokens) + len(q2_tokens))/2
- fuzz_ratio = How much percentage these two strings are similar, measured with edit distance.
- fuzz_partial_ratio = if two strings are of noticeably different lengths, we are getting the score of the best matching lowest length substring.
- token_sort_ratio = sorting the tokens in string and then scoring fuzz_ratio.
- longest_substr_ratio = len(longest common substring) / (min(len(q1_tokens), len(q2_tokens))

### Text Vectorization
Vectorized text of question 1 and question 2 using tf-idf weighted word2vec

## ML Models
Our best performing model was Random Forest Classifier and we further hyperparameter tuned it using RandomizedSearchCV
