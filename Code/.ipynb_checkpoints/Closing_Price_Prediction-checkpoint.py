import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # Corrected import
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# 1. Importing DataFrame as train_df
train_df = pd.read_csv('train.csv')

# 2. Handling Missing Values in Target Variable 'y'
# Drop rows where 'target' is NaN
train_df = train_df.dropna(subset=['target'])
y = train_df['target']

# 3. Selecting Features and Creating 'X'
features = ['imbalance_size', 'imbalance_buy_sell_flag', 'reference_price', 'matched_size',
            'far_price', 'near_price', 'bid_price', 'ask_price', 'bid_size', 'ask_size',
            'wap', 'seconds_in_bucket']
X = train_df[features].copy()  # Use .copy() to avoid SettingWithCopyWarning

# 4. Encoding Categorical Variables
label_encoder = LabelEncoder()
X['imbalance_buy_sell_flag'] = label_encoder.fit_transform(X['imbalance_buy_sell_flag'].astype(str))

# 5. Handling Missing Values in Features
# Identify numerical columns
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

# Iterative imputing
iter_imputer = IterativeImputer(random_state=42)
X[num_cols] = iter_imputer.fit_transform(X[num_cols])

# Outliers
for col in num_cols:
    lower_limit = X[col].quantile(0.01)
    upper_limit = X[col].quantile(0.99)
    X[col] = X[col].clip(lower_limit, upper_limit)

# 6. Creating Derived Features
X['bid_ask_spread'] = X['ask_price'] - X['bid_price']
X['order_book_imbalance'] = (X['bid_size'] - X['ask_size']) / (X['bid_size'] + X['ask_size'])

# Handle possible division by zero in 'order_book_imbalance'
X['order_book_imbalance'] = X['order_book_imbalance'].replace([np.inf, -np.inf], np.nan)
X['order_book_imbalance'] = X['order_book_imbalance'].fillna(0)

# 7. Splitting Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 8. Hyperparameter Tuning with HistGradientBoostingRegressor
# Define parameter distributions
param_distributions = {
    'learning_rate': [0.03, 0.05, 0.07],
    'max_iter': [150, 200, 250],
    'max_leaf_nodes': [31, 50],
    'max_depth': [5, 7],
    'min_samples_leaf': [10, 15, 20],
    'l2_regularization': [0.0, 0.05, 0.1],
    'loss': ['squared_error', 'absolute_error'],
}

# Initialize the model
model = HistGradientBoostingRegressor(
    early_stopping=True,
    validation_fraction=0.1,
    random_state=42
)

# Set up cross-validation
tscv = TimeSeriesSplit(n_splits=5)

# Run randomized search
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_distributions,
    n_iter=20,  # Adjust based on computational resources
    cv=tscv,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=1,
    random_state=42
)

# Fit to the training data
random_search.fit(X_train, y_train)

# 9. Evaluating the Best Model
best_model = random_search.best_estimator_

# Make predictions on the test set
y_pred = best_model.predict(X_test)

# Evaluate performance
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')