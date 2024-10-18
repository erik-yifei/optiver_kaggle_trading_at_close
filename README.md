# optiver_trading_at_the_close
Optiver - Trading at the Close Competition

Results - Mean Absolute Error: 5.685042068222932

Description:

The code implements a machine learning pipeline to predict a target variable using a gradient boosting regression model. Here's a step-by-step breakdown:

Data Importation:
Loads the dataset from 'train.csv' into a pandas DataFrame named train_df.

Handling Missing Target Values:
Drops any rows where the target variable 'target' is missing.
Separates the target variable y from the DataFrame.

Feature Selection and Preparation:
Selects relevant features to include in the model and creates a feature matrix X.
Uses LabelEncoder to encode the categorical variable 'imbalance_buy_sell_flag'.

Imputing Missing Values in Features:
Identifies numerical columns in X.
Applies IterativeImputer to fill in missing values for numerical features.
Clips outliers in numerical features to be within the 1st and 99th percentiles.

Feature Engineering:
Creates new features:
'bid_ask_spread': Difference between 'ask_price' and 'bid_price'.
'order_book_imbalance': Calculated as the difference between 'bid_size' and 'ask_size' divided by their sum.
Handles potential division by zero in 'order_book_imbalance' by replacing infinite values with zero.

Data Splitting:
Splits the dataset into training and testing sets using an 80/20 split (train_test_split), ensuring reproducibility with random_state=42.

Model Initialization and Hyperparameter Tuning:
Defines a parameter grid (param_distributions) for hyperparameter tuning of the HistGradientBoostingRegressor.
Sets up time-series cross-validation (TimeSeriesSplit) with 5 splits to maintain the temporal order of the data.
Uses RandomizedSearchCV to perform hyperparameter tuning with 20 iterations, optimizing for negative mean absolute error.
Fits the model to the training data.

Model Evaluation:
Identifies the best model (best_model) from the hyperparameter search.
Makes predictions (y_pred) on the test set using the best model.
Calculates the Mean Absolute Error (MAE) between the predicted and actual target values on the test set.