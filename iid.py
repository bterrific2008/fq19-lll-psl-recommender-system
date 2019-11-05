#!/usr/bin/env python3

import math
import os

import pandas
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split

k = 10
metric = 'cosine'


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

    def splitImplicitExplicit(bookRatingData):
        """

        :param bookRatingData:
        :return:
        """

        # ratings with a score of 1 are "explicit" ratings (their score is explicitly stated)
        explicitRatings = bookRatingData[bookRatingData.bookRating != 0]

        # ratings with a value of 0 are "implicit" ratings (their score is "implied"
        implicitRatings = bookRatingData[bookRatingData.bookRating == 0]

        return explicitRatings, implicitRatings

    bookRatingData = dataExists(bookRatingData, bookData, userData)
    explicitRatingData, implicitRatingData = splitImplicitExplicit(bookRatingData)

    def bookPopularity(explicitRatingData, bookData):
        """

        :param explicitRatingData:
        :param bookData:
        :return:
        """

        ratingCount = pandas.DataFrame(explicitRatingData.groupby(['ISBN'])['bookRating'].sum())

        top10 = ratingCount.sort_values('bookRating', ascending=False).head(10)
        top10 = top10.merge(bookData, left_index=True, right_on='ISBN')
        print(top10)

        return ratingCount.merge(bookData, left_index=True, right_on='ISBN')

    # TODO find a use for this
    explicitRatingCount = bookPopularity(explicitRatingData, bookData)

    userExplicitRating = userData[userData.userId.isin(explicitRatingData.userId)]
    userImplicitRating = userData[userData.userId.isin(implicitRatingData.userId)]

    # TODO make the values and features work for implicit and explicit data
    values = explicitRatingData
    items = bookData[bookData.ISBN.isin(explicitRatingData.ISBN)]
    users = userExplicitRating

    return items, users, values


def evaluate(predictions, testValues):
    testValues = list(testValues)

    mse = 0.0

    for i in range(len(predictions)):
        # print("Prediction: %.2f vs Actual: %.2f" % (predictions[i], testValues[i]))

        mse += (predictions[i] - testValues[i]) ** 2

    mse /= len(predictions)

    print("MSE: %f" % (mse))


def collaborativeFiltering(items, users, values):
    counts1 = values['userId'].value_counts()
    values = values[values['userId'].isin(counts1[counts1 >= 100].index)]
    counts = values['bookRating'].value_counts()
    values = values[values['bookRating'].isin(counts[counts >= 100].index)]

    ratingMatrix = values.pivot(index='userId', columns='ISBN', values='bookRating')

    ratingMatrix.fillna(0, inplace=True)
    ratingMatrix = ratingMatrix.astype(np.int32)

    def findKSimilarUsers(userId, ratings, metric=metric, k=k):
        similarities = []
        indicies = []
        model_knn = NearestNeighbors(metric=metric, algorithm='brute')
        model_knn.fit(ratings)
        loc = ratings.index.get_loc(userId)
        distances, indicies = model_knn.kneighbors(ratings.iloc[loc, :].values.reshape(1, -1),
                                                   n_neighbors=k + 1)
        similarities = 1 - distances.flatten()

        return similarities, indicies

    def predictUserbased(userId, itemId, ratings, metric=metric, k=k):
        prediction = 0
        user_loc = ratings.index.get_loc(userId)
        item_loc = ratings.columns.get_loc(itemId)
        similarities, indices = findKSimilarUsers(userId, ratings, metric, k)

        mean_rating = ratings.iloc[user_loc, :].mean()
        sum_wt = np.sum(similarities) - 1
        product = 1
        wtd_sum = 0

        for i in range(0, len(indices.flatten())):
            if indices.flatten()[i] == user_loc:
                continue;
            else:
                ratings_diff = ratings.iloc[indices.flatten()[i], item_loc] - np.mean(
                    ratings.iloc[indices.flatten()[i], :])
                product = ratings_diff * (similarities[i])
                wtd_sum = wtd_sum + product

        if prediction <= 0:
            prediction = 1
        elif prediction > 10:
            prediction = 10

        prediction = int(round(mean_rating + (wtd_sum / sum_wt)))
        print("Prediction for user {} -> item {} : {}".format(userId, itemId, prediction))

        return prediction

    predictUserbased(11676, '0001056107', ratingMatrix)


def itemBasedRecommendation(items, users, values):
    counts1 = values['userId'].value_counts()
    values = values[values['userId'].isin(counts1[counts1 >= 100].index)]
    counts = values['bookRating'].value_counts()
    values = values[values['bookRating'].isin(counts[counts >= 100].index)]

    ratingMatrix = values.pivot(index='userId', columns='ISBN', values='bookRating')
    userId = ratingMatrix.index
    ISBN = ratingMatrix.columns
    print(ratingMatrix.shape)

    numUsers = ratingMatrix.shape[0]
    nBooks = ratingMatrix.shape[1]

    ratingMatrix.fillna(0, inplace=True)
    ratingMatrix = ratingMatrix.astype(np.int32)

    def findKSimilarItems(itemId, ratings, metric=metric, k=k):
        similarities = []
        indices = []
        ratings = ratings.T
        loc = ratings.index.get_loc(itemId)
        model_knn = NearestNeighbors(metric=metric, algorithm='brute')
        model_knn.fit(ratings)

        distances, indicies = model_knn.kneighbors(ratings.iloc[loc, :].values.reshape(1, -1),
                                                   n_neighbors=k + 1)
        similarities = 1 - distances.flatten()

        return similarities, indicies

    def predictItemBased(userId, itemId, ratings, metric=metric, k=k):
        prediction = wtd_sum = 0
        user_loc = ratings.index.get_loc(userId)
        item_loc = ratings.columns.get_loc(itemId)

        # similar users based on correlations coefficients
        similarities, indices = findKSimilarItems(itemId, ratings)

        sum_wt = np.sum(similarities) - 1
        product = 1
        for i in range(0, len(indices.flatten())):
            if indices.flatten()[i] == item_loc:
                continue;
            else:
                product = ratings.iloc[user_loc, indices.flatten()[i]] * (similarities[i])
                wtd_sum = wtd_sum + product

        prediction = int(round(wtd_sum/sum_wt))

        if prediction <= 0:
            prediction = 1
        elif prediction > 10:
            prediction = 10

        print("Prediction for user {} -> item {} : {}".format(userId, itemId, prediction))

        return prediction

    predictItemBased(11676, '0001056107', ratingMatrix)



def main():
    items, users, values = getData()

    collaborativeFiltering(items, users, values)
    itemBasedRecommendation(items, users, values)

    """ not sure if we need this...
    # Drop the id columns.
    features = features.drop('ISBN', 1)
    values = values['Book-Rating']"""

    return

    trainFeatures, testFeatures, trainValues, testValues = train_test_split(features, values,
                                                                            random_state=4)

    regressor = RandomForestRegressor(n_estimators=10, random_state=12345, n_jobs=8)
    regressor.fit(trainFeatures, trainValues)

    predictions = regressor.predict(testFeatures)

    evaluate(predictions, testValues)


if (__name__ == '__main__'):
    main()
