# reverses the encoding to get the ISBN number for rating_obs.txt

import pickle
import os

import pandas
import iid

minRating = 0 if iid.minRating is None else iid.minRating
maxRating = 0 if iid.maxRating is None else iid.maxRating

def denormalizeRatings(data):
        """
        Reverses normalization back to ratings between 0-10

        :param data: a DF with the normalized book ratings
        :return: a DF where the ratings are de-normalized(?)
        """

        normRating = data[data.columns[1]]
        normRating = normRating.apply(
            lambda ratings: ratings * (maxRating - minRating) + minRating
        )

        return data

def main():
	# load in the encoder for ISBN
	encoderISBN = pickle.load(open('encoderISBN.pickle', 'rb'))

	# load in txt file containing encoded ISBN values and their respective predicted ratings
	path = os.path.join('data', 'rating_obs.txt')
	ISBNRating = pandas.read_csv(path, error_bad_lines=False, sep='\t', encoding='utf-8')
	
	# reverse engineering magic: transforms labels back to original encoding
	ISBNNum = ISBNRating[ISBNRating.columns[0]]
	ISBNRating[ISBNRating.columns[0]] = encoderISBN.inverse_transform(ISBNNum)

	# reverses noramlized ratings
	ISBNRating = denormalizeRatings(ISBNRating)
	
	ISBNRating.to_csv('reversedISBN.txt', sep="\t", header=False, index=False)
	

if (__name__ == '__main__'):
    main()