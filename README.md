# Book Recommender System

## Overview

The code in this repository recommends books based on user information and book ratings. Our book recommender system (hopefully in the future) incorporates the Probabilistic Soft Logic (PSL) framework. 

## Problem (Wait, are we doing a book recommender system or prediciting political preference?)

In this example, we attempt to predict a user's political preference given the social network that they belong to.
This is a synthetic dataset generated using power-law distributions to model typical social networks.

## Dataset

The dataset is synthetic and contains information about a user's inherent bias and six different types of relationships that
they can have with the users around them.

The dataset was taken from http://www2.informatik.uni-freiburg.de/~cziegler/BX/.

## Origin

This example is a simplified version of one of the experiments from Bach et al.'s core PSL paper:
"Hinge-Loss Markov Random Fields and Probabilistic Soft Logic":
```
@article{bach:jmlr17,
  Author = {Bach, Stephen H. and Broecheler, Matthias and Huang, Bert and Getoor, Lise},
  Journal = {Journal of Machine Learning Research (JMLR)},
  Title = {Hinge-Loss {M}arkov Random Fields and Probabilistic Soft Logic},
  Year = {2017}
}
```

## Directions

### Preliminary Instructions

1. Install PSL. https://github.com/linqs/psl
2. Make sure everything runs properly with a basic example.
`java -cp ./target/classes: `cat classpath.out` edu.umd.cs.example.BasicExample`

More information: https://github.com/linqs/psl/wiki/Getting-started

### Instructions (currently under construction)

1. Clone the repository: `git clone https://github.com/bterrific2008/fq19-lll-psl-recommender-system.git`
2. Currently in work because we still have no clue how everything works yet

## Keywords

 - `cli`
 - `groovy`
 - `inference`
 - `synthetic data`
