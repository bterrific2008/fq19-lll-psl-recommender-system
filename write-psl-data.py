#!/usr/bin/env python3

import math
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

import iid
import pandas as pd

def main():
    os.makedirs('data', exist_ok=True)

    ratings = iid.getData()

    values = ratings[['ISBN','bookRating']]
    features = ratings[['ISBN','bookTitle', 'bookAuthor', 'yearOfPublication', 'publisher']]

    meanRating = values['bookRating'].mean()
    meanRating = pd.DataFrame([[0, meanRating]])

    sameAuthor = features[['ISBN', 'bookAuthor']]
    sameAuthor = sameAuthor.merge(sameAuthor, on = 'bookAuthor', suffixes = ('_left', '_right'))
    sameAuthor = sameAuthor[['ISBN_left', 'ISBN_right']]
    sameAuthor = sameAuthor[sameAuthor['ISBN_left'] < sameAuthor['ISBN_right']]

    samePublisher = features[['ISBN', 'publisher']]
    samePublisher = samePublisher.merge(samePublisher, on = 'publisher', suffixes = ('_left', '_right'))
    samePublisher = samePublisher[['ISBN_left', 'ISBN_right']]
    samePublisher = samePublisher[samePublisher['ISBN_left'] < samePublisher['ISBN_right']]

    sameAuthor.to_csv('data/same_author_obs.txt', sep = '\t', header = False, index = False)
    samePublisher.to_csv('data/same_publisher_obs.txt', sep='\t', header=False, index=False)
    meanRating.to_csv('data/mean_rating_obs.txt', sep='\t', header=False, index=False)

    trainFeatures, testFeatures, trainValues, testValues = train_test_split(features, values,
                                                                            random_state=4)
    trainValues.to_csv('data/rating_obs.txt', sep="\t", header=False, index=False)
    testValues.to_csv('data/rating_truth.txt', sep="\t", header=False, index=False)
    testValues['ISBN'].to_csv('data/rating_targets.txt', sep="\t", header=False, index=False)

    # Drop the id columns.
    trainFeatures = trainFeatures.drop('ISBN', 1)
    testFeatures = testFeatures.drop('ISBN', 1)
    trainValues = trainValues['bookRating']

    regressor = RandomForestRegressor(n_estimators=10, random_state=4, n_jobs=8)
    regressor.fit(trainFeatures, trainValues)

    testValues['bookRating'] = regressor.predict(testFeatures)

    testValues.to_csv('data/iid_obs.txt', sep="\t", header=False, index=False)


if (__name__ == '__main__'):
    main()
