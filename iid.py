#!/usr/bin/env python3

import math
import os

import pandas
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


def getData():
    bookData = pandas.read_csv('BX-Books.csv', error_bad_lines=False, sep=';', encoding='utf-8')
    bookData = bookData.drop(columns=["Image-URL-S", "Image-URL-M", "Image-URL-L"])
    userData = pandas.read_csv('BX-Users.csv', error_bad_lines=False, encoding="utf-8", sep=";")
    bookRatingData = pandas.read_csv('BX-Book-Ratings.csv', error_bad_lines=False, encoding="utf-8",
                                     sep=";")

    masterData = pandas.merge(bookData, bookRatingData, on="ISBN", how='outer')
    masterData = pandas.merge(masterData, userData, on="User-ID", how='outer')

    # drop all data with null columns
    # could be problematic, as some ratings already have null values
    #   in them
    masterData = masterData.dropna()

    # remove books with under 10 reviews
    masterData = masterData[masterData.groupby("ISBN").ISBN.transform('count') > 10]

    print(len(masterData.index))

    values = masterData[['ISBN', 'Book-Rating']]
    features = masterData[
        ['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher', 'User-ID',
         'Location', 'Age']]

    # Clean up the data a bit.
    features = features.fillna(0)

    # Turn strings into ints.
    for stringColumn in ['ISBN', 'Book-Title', 'Book-Author', 'Location',
                         'Publisher']:
        encoder = preprocessing.LabelEncoder()
        features[stringColumn] = encoder.fit_transform(features[stringColumn])

    # TODO Normalize the ratings? to make eval more consistent.
    # Figure out if we should do that or not
    """minPrice = float(values['price'].min())
    maxPrice = float(values['price'].max())
    pandas.options.mode.chained_assignment = None
    values['price'] = values['price'].apply(
        lambda price: 0.25 + (price - minPrice) / (maxPrice - minPrice) / 2.0)"""

    return features, values


def evaluate(predictions, testValues):
    testValues = list(testValues)

    mse = 0.0

    for i in range(len(predictions)):
        # print("Prediction: %.2f vs Actual: %.2f" % (predictions[i], testValues[i]))

        mse += (predictions[i] - testValues[i]) ** 2

    mse /= len(predictions)

    print("MSE: %f" % (mse))


def main():
    features, values = getData()

    # Drop the id columns.
    features = features.drop('ISBN', 1)
    values = values['Book-Rating']

    trainFeatures, testFeatures, trainValues, testValues = train_test_split(features, values,
                                                                            random_state=4)

    regressor = RandomForestRegressor(n_estimators=10, random_state=12345, n_jobs=8)
    regressor.fit(trainFeatures, trainValues)

    predictions = regressor.predict(testFeatures)

    evaluate(predictions, testValues)


if (__name__ == '__main__'):
    main()
