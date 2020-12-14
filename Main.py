#!/usr/bin/env python
# This script reads clean data output by preprocessing_Sec5_1.py and applies
# 1. extracts aspects
# 2. estimates rating per aspect
# 3. estimates aspect weight
# 4. calculates overall rating
# 5. Evaluates outputs
# 6. outputs results and stats

import json
import os
import shutil

import nltk
import numpy as np
import string
from nltk.corpus import stopwords
from scipy.special import digamma, gammaln

CLEANED_JSON_DATA_DIR = 'data/TripAdvisorData/CleanData_JSON'
RESULTS_DATA_DIR = 'results/TripAdvisorData'
RATING_ASPECTS = ["Service", "Cleanliness", "Overall", "Value", "Location", "Rooms", "Sleep Quality"]

hotelid_list = []
hotel_data_reviews_list = []
for file in os.listdir(CLEANED_JSON_DATA_DIR):
    with open(CLEANED_JSON_DATA_DIR + '/' + file, encoding='utf-8') as data_file:
        hotel_data_reviews_list.append(json.load(data_file))
        hotelid_list.append(file.split('.')[0])

all_aspect_reviews = []
all_terms = []
all_review_comment_list = []
all_review_freq_dist_list = []
all_hotel_id_list = []
all_review_ids = []
all_review_contents = []
all_overall_ratings = []
all_review_authors = []
for r in range(len(hotel_data_reviews_list)):
    stemmer = nltk.stem.porter.PorterStemmer()
    for review in hotel_data_reviews_list[r]['Reviews']:
        parsedWords = []
        for sentence in nltk.sent_tokenize(review['Content']):
            stemmedWords = [stemmer.stem(w.lower()) for w in nltk.word_tokenize(sentence) if
                            w not in string.punctuation]
            parsedWords += [v for v in stemmedWords if v not in stopwords.words('english')]
        reviewFrequency = dict(nltk.FreqDist(parsedWords))
        all_review_freq_dist_list.append(reviewFrequency)
        all_review_comment_list.append(parsedWords)
        all_review_ids.append(review['ReviewID'])
        all_hotel_id_list.append(hotelid_list[r])
        all_review_contents.append(review['Content'])
        all_aspect_reviews.append(review['Ratings']['Service'])
        all_aspect_reviews.append(review['Ratings']['Cleanliness'])
        all_overall_ratings.append(review['Ratings']['Overall'])
        all_aspect_reviews.append(review['Ratings']['Overall'])
        all_aspect_reviews.append(review['Ratings']['Value'])
        all_aspect_reviews.append(review['Ratings']['Sleep Quality'])
        all_aspect_reviews.append(review['Ratings']['Rooms'])
        all_aspect_reviews.append(review['Ratings']['Location'])
        all_review_authors.append(review['Author'])
        all_terms += parsedWords
termFrequency = nltk.FreqDist(all_terms)
vocab, cnt = [], []
vocabDict = {}
for k, v in termFrequency.items():
    if v > 5:
        vocab.append(k)
        cnt.append(v)
    else:
        for r in all_review_freq_dist_list:
            if k in r:
                del r[k]
        for i in range(len(all_review_comment_list)):
            all_review_comment_list[i] = filter(lambda a: a != k, all_review_comment_list[i])
vocab = np.array(vocab)[np.argsort(vocab)].tolist()
cnt = np.array(cnt)[np.argsort(vocab)].tolist()
vocabDict = dict(zip(vocab, range(len(vocab))))

shapeSize = 7
k = 4
M = len(all_review_freq_dist_list)
mu, sigma = 0.0, 0.0
phi, lmbda, sigmaSq = [], np.zeros(shape=(1, M)), np.zeros(shape=(1, M))
eta = np.zeros([M, k])
gamma = np.ones([M, k])
for m in range(0, M):
    wordsInDoc = list(all_review_freq_dist_list[m].keys())
    N = len(wordsInDoc)
    phi_temp = np.ones([N, k]) * 1 / float(k)
    for i in range(0, k):
        eta[m, i] = gamma[m, i] + N / float(k)
    phi.append(phi_temp)
    lmbda[0, m] = np.random.rand()
    sigmaSq[0, m] = np.random.rand()
lmbda = lmbda / lmbda.sum(axis=1, keepdims=1)
sigmaSq = sigmaSq / sigmaSq.sum(axis=1, keepdims=1)
epsilon = np.zeros([k, len(vocabDict)])
for i in range(0, k):
    tmp = np.random.uniform(0, 1, len(vocabDict))
    epsilon[i, :] = tmp / np.sum(tmp)
for d in range(0, M):
    mu += lmbda[0, d]
mu = mu / M
for d in range(0, M):
    sigma += (lmbda[0, d] - mu) ** 2 + sigmaSq[0, d] ** 2
sigma = sigma / M
likelihood, oldLikelihood, iteration = 0, 0, 1
while iteration <= 5 and (iteration <= 2 or np.abs((likelihood - oldLikelihood) / oldLikelihood) > 1e-4):
    oldLikelihood, oldPhi, oldEta, oldGamma, oldEpsilon, oldLambda, oldSigmaSq, oldMu, oldSigma = likelihood, phi, eta, gamma, epsilon, lmbda, sigmaSq, mu, sigma
    newLmbda, newSigmaSq = np.zeros(shape=(1, M)), np.zeros(shape=(1, M))
    likelihood, newMu, newSigma = 0.0, 0.0, 0.0
    convergence = np.zeros(M)
    for d in range(0, M):
        words = list(all_review_freq_dist_list[d].keys())
        N = len(words)
        p = phi[d]
        counter = 0
        while convergence[d] == 0 and d < len(convergence):
            oldPhi = p
            p = np.zeros([N, k])
            oldEta = eta[d, :]
            for n in range(0, N):
                if words[n] in list(vocabDict):
                    vocabIdx = list(vocabDict).index(words[n])
                    for i in range(0, k):
                        e = epsilon[i, vocabIdx]
                        p[n, i] = e * np.exp(digamma(eta[d, i]) - digamma(np.sum(eta[d, :])))
                    p[n, :] = p[n, :] / np.sum(p[n, :])
            eta[d, :] = gamma[d, :] + np.sum(p, axis=0)
            newLmbda[0, d] = 0.5 * (lmbda[0, d] - mu) ** 2
            newLmbda = newLmbda / newLmbda.sum(axis=1, keepdims=1)
            newSigmaSq[0, d] = sigmaSq[0, d] / sigma
            newSigmaSq = newSigmaSq / newSigmaSq.sum(axis=1, keepdims=1)
            counter += 1
            if np.linalg.norm(p - oldPhi) < 1e-3 and np.linalg.norm(
                    eta[d, :] - oldEta) < 1e-3:
                convergence[d] = 1
                phi[d] = p
                eta_slice = eta[d, :]
                gamma_slice = gamma[d, :]
                V = len(vocabDict)
                review = list(all_review_freq_dist_list[d].keys())
                N = len(review)
                gammaSum, phiEtaSum, phiLogEpsilonSum, entropySum, etaSum = 0.0, 0.0, 0.0, 0.0, 0.0
                gammaSum += gammaln(np.sum(gamma_slice))
                etaSum -= gammaln(np.sum(eta_slice))
                for i in range(0, k):
                    gammaSum += -gammaln(gamma_slice[i]) + (gamma_slice[i] - 1) * (
                                digamma(eta_slice[i]) - digamma(np.sum(eta_slice)))
                    for n in range(0, N):
                        if phi[d][n, i] > 0:
                            indicator = np.sum(np.in1d(len(vocabDict), review[n]))
                            phiEtaSum += phi[d][n, i] * (digamma(eta_slice[i]) - digamma(np.sum(eta_slice[:])))
                            entropySum += phi[d][n, i] * np.log(phi[d][n, i])
                            for j in range(0, V):
                                if epsilon[i, j] > 0:
                                    phiLogEpsilonSum += phi[d][n, i] * indicator * np.log(epsilon[i, j])
                    etaSum += gammaln(eta_slice[i]) - (eta_slice[i] - 1) * (
                                digamma(eta_slice[i]) - digamma(np.sum(eta_slice[:])))
                likelihood += (gammaSum + phiEtaSum + phiLogEpsilonSum - etaSum - entropySum)

    for d in range(0, M):
        newMu += newLmbda[0, d]
    mu = mu / M
    for d in range(0, M):
        newSigma += (newLmbda[0, d] - newMu) ** 2 + newSigmaSq[0, d] ** 2
    newSigma = newSigma / M

    mu, sigma = newMu, newSigma

    V = len(vocabDict)
    epsilon = np.zeros([k, V])
    for d in range(0, M):
        words = list(all_review_freq_dist_list[d].keys())
        for i in range(0, k):
            p = phi[d][:, i]
            for j in range(0, V):
                word = list(vocabDict)[j]
                indicator = np.in1d(words, word).astype(int)
                epsilon[i, j] += np.dot(indicator, p)
    epsilon = np.transpose(np.transpose(epsilon) / np.sum(epsilon, axis=1))  # the epsilon value
    iteration += 1

reviewLabelList = [[] for i in range(len(all_review_freq_dist_list))]
for i in range(len(all_review_freq_dist_list)):
    aspectWeights = np.zeros(shape=(shapeSize, len(list(all_review_freq_dist_list[i].keys()))))
    for j in range(shapeSize):
        aspectWeights[j] = np.random.normal(loc=mu, scale=sigma, size=len(list(all_review_freq_dist_list[i].keys())))
    aspectWeights = aspectWeights / aspectWeights.sum(axis=1, keepdims=1)
    for j in range(shapeSize):
        reviewLabels = [-1] * len(list(all_review_freq_dist_list[i].keys()))
        reviewLabels[np.where(aspectWeights[j] == max(aspectWeights[j]))[0][
            0]] = 1
        reviewLabelList[i].append(reviewLabels)

reviewMatrixList = []
for i in range(len(all_review_freq_dist_list)):
    review = list(all_review_freq_dist_list[i].keys())
    reviewMatrix = np.zeros((len(reviewLabelList[i]), len(review)))
    for j in range(len(reviewLabelList[i])):
        for k in range(len(review)):
            reviewMatrix[j, k] = all_review_freq_dist_list[i][review[k]] * reviewLabelList[i][j][k]
        reviewMatrix[j] = (reviewMatrix[j] - reviewMatrix[j].min(0)) / reviewMatrix[j].ptp(
            0)
    reviewMatrixList.append(reviewMatrix)

predList = []
for i in range(len(reviewMatrixList)):
    for j in range(len(reviewMatrixList[i])):
        predReviews = 0
        for k in range(len(reviewMatrixList[i][j])):
            review = list(all_review_freq_dist_list[i].keys())
            predReviews += all_review_freq_dist_list[i][review[k]] * reviewMatrixList[i][j][k]
        predReviews = predReviews / len(reviewMatrixList[i][j])
        predList.append(predReviews)
predList = [float(i) * 5 / max(predList) for i in predList]

totalMse = np.square(np.subtract(predList, all_aspect_reviews)).mean()
totalPearson = np.corrcoef(predList, all_aspect_reviews)[0, 1]

results_file = RESULTS_DATA_DIR + '/results.txt'

f = open(results_file, 'w')
for i in range(len(all_review_comment_list)):
    f.write(':'.join(
        [all_hotel_id_list[i], all_review_ids[i], all_review_contents[i], str(all_review_comment_list[i]), str(reviewMatrixList[i])]) + '\n')
total_annotated_reviews = 0
total_length_reviews = 0
labels_per_reviews = []
for i in range(len(all_review_comment_list)):
  total_length_reviews += len(all_review_contents[i])
  for j in range(len(reviewLabelList[i])):
    num_annotated_reviews = 0
    if reviewLabelList[i][j] != -1:
      num_annotated_reviews += 1
      labels_per_reviews.append(num_annotated_reviews)
    total_annotated_reviews += num_annotated_reviews

print('Total reviews: {}'.format(len(all_review_comment_list)))
print('MSE: {}'.format(totalMse))