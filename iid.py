#!/usr/bin/env python3

import math
import os

# Simple Python Recommendation System Enginer (SurPRISE)
from surprise import KNNBasic
from surprise import Dataset
from surprise import Reader
from surprise import NMF
from surprise import SVD
from surprise import accuracy
from surprise.model_selection import cross_validate
from surprise.model_selection import train_test_split

import pandas
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor


def getData():
    """
    Finds the items, users, and ratings (values) of book recommendations
    Does some cleaning of the data

    Heavily based off of csaluja's Book Recommendation System:
    https://github.com/csaluja/JupyterNotebooks-Medium/blob/master/Book%20Recommendation%20System.ipynb?source=post_page-----5ec959c41847----------------------

    :return: items, users, values
    """

    # TODO put all of these into pickle files so we don't do this step again and again...
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
        Clean errors present in the "yearOfPublication" column for bookData

        There are some parts of the data where the columns have been improperly delimited
        There is an issue where there are dates from 2050, when the dataset was collected in 2004
        This also ensures that the "yearOfPublication" column is saved as an int type

        :param bookData: an unclean version
        :return: bookData: a cleaned version
        """

        # ISBN 0789466953 is improperly formatted. Fix it
        bookData.loc[bookData.ISBN == "0789466953", 'yearOfPublication'] = 2000
        bookData.loc[
            bookData.ISBN == "0789466953", 'bookTitle'] = "DK Readers: Creating the X-Men, How " \
                                                          "Comic Books Come to Life (Level 4: " \
                                                          "Proficient Readers) "
        bookData.loc[bookData.ISBN == "0789466953", 'bookAuthor'] = "James Buckley"
        bookData.loc[bookData.ISBN == "0789466953", 'publisher'] = "DK Publishing Inc"

        # ISBN 078946697X is improperly formatted. Fix it
        bookData.loc[bookData.ISBN == "078946697X", 'yearOfPublication'] = 2000
        bookData.loc[
            bookData.ISBN == "078946697X", 'bookTitle'] = "DK Readers: Creating the X-Men, How It " \
                                                          "All Began (Level 4: Proficient Readers) "
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
        Cleans the "publisher" column of bookData

        Only significant issue here is that there are "NULL" publishers.
        The fix is to mark these publishers as "other" instead

        :param bookData: unclean version
        :return: bookData: clean version
        """

        # if there are books with no listed publisher, then set them to be "other"
        bookData.publisher.fillna('other', inplace=True)
        return bookData

    bookData = publisher(bookData)

    def age(userData):
        """
        Cleans the "age" column of userData

        Only significant issue here is that there are ages below 5, and ages above 90. Making the
        assumption that we only find reviews made by readers at at least an 8th grade level are
        meaningful, and that we shouldn't have readers above the age of 90

        :param userData: unclean version
        :return: userData: clean version
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
        Ensures that we only consider ratings for books that we have book data on, from users
        that we have user data on

        :param bookRatingData: unclean version
        :param bookData: contains the ISBN codes for all the books we have
        :param userData: contains the userId for all the users we have
        :return: bookRatingData: clean version
        """

        # only keep ratings that we have books for
        bookRatingData = bookRatingData[bookRatingData.ISBN.isin(bookData.ISBN)]

        # only keep ratings that we have users for
        bookRatingData = bookRatingData[bookRatingData.userId.isin(userData.userId)]

        return bookRatingData

    def splitImplicitExplicit(bookRatingData):
        """
        Separates implicit ratings (ratings of 0) from explicit ratings (all other ratings)

        :param bookRatingData: unclean version
        :return: explicitRatingData: pandas DataFrame of explicit ratings
        :return: implicitRatings: pandas DataFrame of implicit ratings
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
        Find the overall popularity of a book, based off of a sum of all of its reviews

        :param explicitRatingData: contains ratings with explicit reviews :param bookData:
        contains book data :return: a DataFrame that combines the popularity rating of a book
        with its book data (the DataFrame bookData)
        """

        ratingCount = pandas.DataFrame(explicitRatingData.groupby(['ISBN'])['bookRating'].sum())

        # TODO: rename "bookRating" to "popularityRating"
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


def evaluate(predictions):
    """
    Prints to stdout the Mean Absolute Error, Root Mean Squared Error, and Mean Squared Error
    Also returns the MAE, RMSE, and MSE

    :param predictions: a surprise.prediction_algorithms.predictions object, that contains
    predictions to evaluate
    :return: the MAE, RMSE, and MSE
    """

    mae = accuracy.mae(predictions)
    rmse = accuracy.rmse(predictions)
    mse = accuracy.mse(predictions)

    return mae, rmse, mse


def main():
    item, users, values = getData()

    reader = Reader(rating_scale=(1, 10))
    data = Dataset.load_from_df(values[["userId", "ISBN", "bookRating"]], reader)

    # splits that data into training sets and testing sets
    trainset, testset = train_test_split(data, test_size=.2)

    # tests the model against a Singular Value Decomposition model
    algo = SVD()
    algo.fit(trainset)
    predictions = algo.test(testset)
    print("svd")
    evaluate(predictions)

    # tests the model against a Probabilistic Matrix Factorization Model
    # TODO check to make sure  this actually does what we want it to do.
    #  I took it verbatim from the tutorial
    algo = SVD(biased=False)
    algo.fit(trainset)
    predictions = algo.test(testset)
    print("svd biased false")
    evaluate(predictions)

    # tests the model against a K Nearest Neighbors model
    # TODO getting a timeout error
    algo = KNNBasic(sim_options={'user_based': True})
    algo.fit(trainset)
    predictions = algo.test(trainset)
    print("knn basic")
    evaluate(predictions)


if (__name__ == '__main__'):
    main()
