## Using different machine learning algorithms to predict the type of exercise being done

The R script presented here (and its results PDF document) use several different machine learning algorithms to predict the type of exercise being done based on data from an accelerometer and gyrometer. The data is taken from a study that can be found here [http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har](http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har). 

In this repository a Python script is also present that performs an identical process to the R script. This Python script is included to
demonstrate competency in machine learning modules in Python such Sci-kit learn. In this script a random forest classifier is implemented through the function RandomForestRegressor(). The hyperparameters (the variables chosen at each split) are optimized using GridSearchCV and categorical variables are converted to numeric variables for the random forest to use through one hot encoding using the function 
LabelBinarizer(). Pandas is used to import the data as a data frame, and pre-process and manipulate the data.

The script and report in this repo apply machine learning tenchiqnies to real world problems. It is demonstrated how machine learning problems can be set up, how we can measure what we would expect the accuracy to be in a test case (through cross validation) and the underlying pricinples governing each machine learning technique.


