# AirBnb Listings Price Prediction

## Overview

Since its founding in August 2008, Airbnb, initially known as “AirBedAndBreakfast”, has disrupted the hospitality industry significantly and challenged the traditional hotel industry with its innovative hospitality model. Airbnb’s model is based on being an online marketplace connecting people who want to rent out their residences with people looking for accommodation in specific locations offering their hosts an easy way to earn some income from their properties. The model’s innovation lies in the fact that it doesn’t require the purchase or construction of any lodging properties enlisted on its marketplace; however, it achieves profits by charging a service fee for each booking.

To increase the efficiency of its services, Airbnb must provide effective tools to its different customers: the property hosts and the property renters. At the level of the property hosts, setting the price of their rental unit on Airbnb is a confusing endeavor since it takes into consideration various factors such the availability of customers, seasonality, location, and the amount of desirable profit. On the other hand, renters face the challenge to rent the appropriate property that satisfies their needs and desires at affordable fair prices.

To tackle the problem of the property host’s setting the price of their rental unit, we designed an optimal pricing recommender system. After applying the appropriate preprocessing and feature engineering techniques on both categorical and numerical features, recursive feature elimination to select the most important features was implemented. Afterwards, several regression models were experimented: linear regression, random forest, support vector regression, gradient boosting tree and extreme gradient boosting. After the models were assessed using mean squared error, root mean squared error and mean absolute error, gradient boosting performed the best followed by extreme gradient boosting.

## Data & Methods

In this project, we perform EDA, Feature Engineering, Data Cleaning and use several models to predict the prices of Airbnb listing on the public London Airbnb dataset which can be found [here](https://www.kaggle.com/datasets/labdmitriy/airbnb). The listings.csv file was used for this project.

### Data Splitting

The data was split into train and test sets (80-20%). Then the training set was split into train and validation set through 5-fold cross validation.

### Preprocessing

The pre-processing step consisted of (i) removing irrelevant or uninformative features (columns containing urls) (ii) removing features with a large amount of missing values, (iii) converting some features into floats (e.g. by removing the dollar sign in prices) (iv) removing the features that contained one predominant category (e.g. Real bed constitute 99.21% of the bed type column) (v) reducing the number of features that are highly correlated with other features (vi) one hot encoding the categorical non binary features (vii) ordinal encoding binary features (viii) imputing categorical features with the mode and numerical features with the mean (ix) transforming numerical features so that they follow a gaussian distribution (x) normalizing numerical features. Finally, the labels (price) were log transformed to render the target distribution more gaussian like to avoid biasing our models to the majority labels.

### Feature Selection

Recursive feature elimination was used to find the most important features. The resulting set consisted of 18 features which is 65% less than the number of original features.

### Models

Linear Regression (LR) was taken as the baseline model for evaluating the performance of the other models. After selecting a set of features using recursive feature elimination and Lasso, several machine learning models were considered in order to find the optimal one including random forest regressor (RF), Support Vector Regression (SVR), Gradient Boosting Tree Ensemble (GB) and extreme gradient boosting (XGBoost).

The RandomizedSearchCV function from scikit learn was used to conduct randomized search on hyper parameters with 2-fold cross validation and 5 iterations for each model. The hyperparameters that resulted in minimizing the root mean squared error (RMSE) were selected.
