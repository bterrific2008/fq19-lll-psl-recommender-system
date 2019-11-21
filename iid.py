#!/usr/bin/env python3

import math
import os

import pandas
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def save_classifier(classifier, classifier_fname):
    classifier_file = open(classifier_fname, 'wb')
    pickle.dump(classifier, classifier_file)
    classifier_file.close()


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

        ratingCount = pandas.DataFrame(explicitRatingData.groupby(['ISBN'])['bookRating'].mean())

        # TODO: rename "bookRating" to "popularityRating"
        top10 = ratingCount.sort_values('bookRating', ascending=False).head(10)
        top10 = top10.merge(bookData, left_index=True, right_on='ISBN')
        #print(top10)

        return ratingCount.merge(bookData, left_index=True, right_on='ISBN')

    # TODO find a use for this
    explicitRatingCount = bookPopularity(explicitRatingData, bookData)

    userExplicitRating = userData[userData.userId.isin(explicitRatingData.userId)]
    userImplicitRating = userData[userData.userId.isin(implicitRatingData.userId)]

    # TODO make the values and features work for implicit and explicit data
    values = explicitRatingData
    items = bookData[bookData.ISBN.isin(explicitRatingData.ISBN)]
    users = userExplicitRating

    # Turn strings into ints.
    # for stringColumn in ['ISBN', 'bookAuthor', 'bookTitle', 'publisher']:
    #     encoder = preprocessing.LabelEncoder()
    #     explicitRatingCount[stringColumn] = encoder.fit_transform(explicitRatingCount[stringColumn].astype(str))
    #     print()

    encoderISBN = preprocessing.LabelEncoder()
    explicitRatingCount['ISBN'] = encoderISBN.fit_transform(explicitRatingCount['ISBN'].astype(str))

    encoderBookAuthor = preprocessing.LabelEncoder()
    explicitRatingCount['bookAuthor'] = encoderBookAuthor.fit_transform(explicitRatingCount['bookAuthor'].astype(str))

    encoderBookTitle = preprocessing.LabelEncoder()
    explicitRatingCount['bookTitle'] = encoderBookTitle.fit_transform(explicitRatingCount['bookTitle'].astype(str))

    encoderPublisher = preprocessing.LabelEncoder()
    explicitRatingCount['publisher'] = encoderPublisher.fit_transform(explicitRatingCount['publisher'].astype(str))

    

    def normalizeRatings(data):
        """
        Normalizes the book ratings

        :param data: a DF with a 'bookRating' column
        :return: a DF where the 'bookRating' column is normalized
        """

        minRating = float(data['bookRating'].min())
        maxRating = float(data['bookRating'].max())
        pandas.options.mode.chained_assignment = None
        data['bookRating'] = data['bookRating'].apply(
            lambda rating: (rating - minRating) / (maxRating - minRating)
        )

        return data

    return normalizeRatings(explicitRatingCount)


def evaluate(predictions, testValues):
    testValues = list(testValues)

    mse = 0.0

    for i in range(len(predictions)):
        # print("Prediction: %.2f vs Actual: %.2f" % (predictions[i], testValues[i]))

        mse += (predictions[i] - testValues[i]) ** 2

    mse /= len(predictions)

    print("MSE: %f" % (mse))


def main():
    ratings = getData()

    values = ratings[['bookRating']]
    features = ratings[['bookTitle', 'bookAuthor', 'yearOfPublication', 'publisher']]

    # splits that data into training sets and testing sets
    trainFeatures, testFeatures, trainValues, testValues = train_test_split(features, values, random_state = 4)

    regressor = RandomForestRegressor(n_estimators=10, random_state=12345, n_jobs=8)
    regressor.fit(trainFeatures, trainValues)

    predictions = regressor.predict(testFeatures)

    evaluate(predictions, testValues['bookRating'].to_numpy())


if (__name__ == '__main__'):
    main()
