# Spam-Classification
Spam classification using logistic regression

Preprocessing performed:
a. Standardized the columns so they all had mean 0 and unit variance.
b. Transformed the features using log(xij + 0.1).
c. Binarized the features using I(xij > 0).
For each version of the data, fitted a logistic regression model using gradient descent.
