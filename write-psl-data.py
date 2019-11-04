#!/usr/bin/env python3

import math
import os

import pandas
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

import iid

def main():
    print("lmao")
    os.makedirs('data', exist_ok = True)

    features, values = iid.getData()

    meanPrice = values['price'].mean()
    meanPrice = pandas.DataFrame([[0, meanPrice]])

    sameBurrow = features[['id', 'neighbourhood_group']]
    sameBurrow = features.merge(features, on = 'neighbourhood_group', suffixes = ('_left', '_right'))
    sameBurrow = sameBurrow[['id_left', 'id_right']]
    sameBurrow = sameBurrow[sameBurrow['id_left'] < sameBurrow['id_right']]

    sameNeighbourhood = features[['id', 'neighbourhood']]
    sameNeighbourhood = features.merge(features, on = 'neighbourhood', suffixes = ('_left', '_right'))
    sameNeighbourhood = sameNeighbourhood[['id_left', 'id_right']]
    sameNeighbourhood = sameNeighbourhood[sameNeighbourhood['id_left'] < sameNeighbourhood['id_right']]

    sameBurrow.to_csv('data/same_burrow_obs.txt', sep = "\t", header = False, index = False)
    sameNeighbourhood.to_csv('data/same_neighbourhood_obs.txt', sep = "\t", header = False, index = False)
    meanPrice.to_csv('data/mean_price_obs.txt', sep = "\t", header = False, index = False)

    trainFeatures, testFeatures, trainValues, testValues = train_test_split(features, values, random_state = 4)

    trainValues.to_csv('data/price_obs.txt', sep = "\t", header = False, index = False)
    testValues.to_csv('data/price_truth.txt', sep = "\t", header = False, index = False)
    testValues['id'].to_csv('data/price_targets.txt', sep = "\t", header = False, index = False)

    # Drop the id columns.
    trainFeatures = trainFeatures.drop('id', 1)
    testFeatures = testFeatures.drop('id', 1)
    trainValues = trainValues['price']

    regressor = RandomForestRegressor(n_estimators = 10, random_state = 4, n_jobs = 8)
    regressor.fit(trainFeatures, trainValues)

    testValues['price'] = regressor.predict(testFeatures)

    testValues.to_csv('data/iid_obs.txt', sep = "\t", header = False, index = False)

if (__name__ == '__main__'):
    main()
