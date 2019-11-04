#!/usr/bin/env python3

import math
import os

import pandas
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


def getData():
    """
    https://github.com/csaluja/JupyterNotebooks-Medium/blob/master/Book%20Recommendation%20System.ipynb?source=post_page-----5ec959c41847----------------------

    :return:
    """

    bookData = pandas.read_csv('BX-Books.csv', error_bad_lines=False, sep=';', encoding='utf-8')
    # remove the useless book cover columns
    bookData.drop(columns=["Image-URL-S", "Image-URL-M", "Image-URL-L"], inplace=True)
    bookData.columns = ["ISBN", "bookTitle", "bookAuthor", "yearOfPublication", "publisher"]

    userData = pandas.read_csv('BX-Users.csv', error_bad_lines=False, encoding="utf-8", sep=";")
    userData.columns = ['userId', 'userLocation', 'userAge']

    bookRatingData = pandas.read_csv('BX-Book-Ratings.csv', error_bad_lines=False, encoding="utf-8",
                                     sep=";")
    bookRatingData.columns = ['userId', 'ISBN', 'bookRating']


    def bookDataYear(bookData):
        """

        :param bookData:
        :return:
        """

        # ISBN 0789466953 is improperly formatted. Fix it
        bookData.loc[bookData.ISBN == "0789466953", 'yearOfPublication'] = 2000
        bookData.loc[
            bookData.ISBN == "0789466953", 'bookTitle'] = "DK Readers: Creating the X-Men, How Comic Books Come to Life (Level 4: Proficient Readers)"
        bookData.loc[bookData.ISBN == "0789466953", 'bookAuthor'] = "James Buckley"
        bookData.loc[bookData.ISBN == "0789466953", 'publisher'] = "DK Publishing Inc"

        # ISBN 078946697X is improperly formatted. Fix it
        bookData.loc[bookData.ISBN == "078946697X", 'yearOfPublication'] = 2000
        bookData.loc[
            bookData.ISBN == "078946697X", 'bookTitle'] = "DK Readers: Creating the X-Men, How It All Began (Level 4: Proficient Readers)"
        bookData.loc[bookData.ISBN == "078946697X", 'bookAuthor'] = "Michael Teitelbaum"
        bookData.loc[bookData.ISBN == "078946697X", 'publisher'] = "DK Publishing Inc"

        # ISBN 2070426769 is improperly formatted. Fix it
        bookData.loc[
            bookData.ISBN == '2070426769', "bookTitle"] = "Peuple de ciel, suivi de 'Les Bergers"
        bookData.loc[
            bookData.ISBN == '2070426769', "bookAuthor"] = "Jean-Marie Gustave Le ClÃ?Â©zio"
        bookData.loc[bookData.ISBN == '2070426769', "publisher"] = "Gallimard"
        bookData.loc[bookData.ISBN == '2070426769', "yearOfPublication"] = "2003"

        # Format yearOfPublications to be numeric data
        bookData.yearOfPublication = pandas.to_numeric(bookData.yearOfPublication, errors='coerce')

        # Our dataset is from 2004, so its unlikely that we'll have data from earlier than 2005
        #   Set all publication years that are 0 or later than 2005 as the average year
        # TODO: is setting this stuff as average a good idea?
        bookData.loc[(bookData.yearOfPublication > 2005) | (
                bookData.yearOfPublication == 0), 'yearOfPublication'] = np.NAN
        bookData.yearOfPublication.fillna(round(bookData.yearOfPublication.mean()), inplace=True)

        # set the dtype as int32
        bookData.yearOfPublication = bookData.yearOfPublication.astype(np.int32)

        return bookData

    bookData = bookDataYear(bookData)

    def publisher(bookData):
        """

        :param bookData:
        :return:
        """

        # if there are books with no listed publisher, then set them to be "other"
        bookData.publisher.fillna('other', inplace=True)
        return bookData

    bookData = publisher(bookData)

    def age(userData):
        """

        :param userData:
        :return:
        """

        # its unlikely that people under the age of 12 and people over the age of 90 are leaving
        #   meaningful reviews. We should filter these out
        userData.loc[(userData.userAge > 90) | (userData.userAge < 12), 'age'] = np.nan

        # TODO: should we just remove all reviews from users under the age of 12 and over the age
        #  of 90, or should we replace them with the average age?
        userData.userAge.dropna(inplace=True)
        # userData.userAge.fillna(userData.userAge.mean(), inplace=True)

        # set data type as int
        userData.userAge = userData.userAge.astype(np.int32)

        return userData

    # remove users too young and too old
    userData = age(userData)

    def dataExists(bookRatingData, bookData, userData):
        """

        :param bookRatingData:
        :return:
        """

        # only keep ratings that we have books for
        bookRatingData = bookRatingData[bookRatingData.ISBN.isin(bookData.ISBN)]

        # only keep ratings that we have users for
        bookRatingData = bookRatingData[bookRatingData.userId.isin(userData.userId)]

        return bookRatingData

    bookRatingData = dataExists(bookRatingData, bookData, userData)

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
        ['ISBN', 'bookTitle', 'bookAuthor', 'yearOfPublication', 'publisher', 'User-ID',
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
