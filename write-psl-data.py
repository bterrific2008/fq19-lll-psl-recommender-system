#!/usr/bin/env python3

import math
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

import iid


def main():
    os.makedirs('data', exist_ok=True)

    items, users, values = iid.getData()

    feature_vector = values.merge(items, on="ISBN").merge(users, on="userId")


    trainFeatures, testFeatures, trainValues, testValues = train_test_split(features, values,
                                                                            random_state=4)

    trainValues.to_csv('data/price_obs.txt', sep="\t", header=False, index=False)
    testValues.to_csv('data/price_truth.txt', sep="\t", header=False, index=False)
    testValues['id'].to_csv('data/price_targets.txt', sep="\t", header=False, index=False)

    # Drop the id columns.
    trainFeatures = trainFeatures.drop('id', 1)
    testFeatures = testFeatures.drop('id', 1)
    trainValues = trainValues['price']

    regressor = RandomForestRegressor(n_estimators=10, random_state=4, n_jobs=8)
    regressor.fit(trainFeatures, trainValues)

    testValues['price'] = regressor.predict(testFeatures)

    testValues.to_csv('data/iid_obs.txt', sep="\t", header=False, index=False)


if (__name__ == '__main__'):
    main()
