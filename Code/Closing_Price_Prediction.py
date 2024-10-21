import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.experimental import enable_iterative_imputer  # noqa

def feat_eng(df, fit=True):
    """
    Performs feature engineering on the input DataFrame.

    Parameters:
    - df (pd.DataFrame): The input DataFrame to process.
    - fit (bool): If True, fits the encoders and imputers. If False, uses the already fitted ones.

    Returns:
    - df_processed (pd.DataFrame): The processed DataFrame.
    """
    import numpy as np
    from sklearn.preprocessing import LabelEncoder
    from sklearn.impute import IterativeImputer

    global label_encoder, iter_imputer, clipping_limits  # To reuse in prediction

    df_processed = df.copy()

    # Features to use
    features = [
        'imbalance_size', 'imbalance_buy_sell_flag', 'reference_price', 'matched_size',
        'far_price', 'near_price', 'bid_price', 'ask_price', 'bid_size', 'ask_size',
        'wap', 'seconds_in_bucket'
    ]
    df_processed = df_processed[features]

    # Encoding categorical variables
    if fit:
        label_encoder = LabelEncoder()
        df_processed['imbalance_buy_sell_flag'] = label_encoder.fit_transform(
            df_processed['imbalance_buy_sell_flag'].astype(str)
        )
    else:
        df_processed['imbalance_buy_sell_flag'] = label_encoder.transform(
            df_processed['imbalance_buy_sell_flag'].astype(str)
        )

    # Identifying numerical columns
    num_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()

    # Handling missing values
    if fit:
        iter_imputer = IterativeImputer(random_state=42)
        df_processed[num_cols] = iter_imputer.fit_transform(df_processed[num_cols])
    else:
        df_processed[num_cols] = iter_imputer.transform(df_processed[num_cols])

    # Handling outliers
    if fit:
        clipping_limits = {}
        for col in num_cols:
            lower_limit = df_processed[col].quantile(0.01)
            upper_limit = df_processed[col].quantile(0.99)
            clipping_limits[col] = (lower_limit, upper_limit)
            df_processed[col] = df_processed[col].clip(lower_limit, upper_limit)
    else:
        for col in num_cols:
            lower_limit, upper_limit = clipping_limits[col]
            df_processed[col] = df_processed[col].clip(lower_limit, upper_limit)

    # Creating derived features
    df_processed['bid_ask_spread'] = df_processed['ask_price'] - df_processed['bid_price']
    df_processed['order_book_imbalance'] = (
        df_processed['bid_size'] - df_processed['ask_size']
    ) / (df_processed['bid_size'] + df_processed['ask_size'])

    # Handle possible division by zero
    df_processed['order_book_imbalance'] = df_processed['order_book_imbalance'].replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0)

    # Update the feature list to include derived features
    features.extend(['bid_ask_spread', 'order_book_imbalance'])
    df_processed = df_processed[features]

    return df_processed


# 1. Importing DataFrame as train_df
train_df = pd.read_csv('train.csv')

# 2. Handling Missing Values in Target Variable 'y'
train_df = train_df.dropna(subset=['target'])
y = train_df['target']

# 3. Preprocessing and Feature Engineering
X_processed = feat_eng(train_df, fit=True)

# 4. Splitting Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

# 5. Hyperparameter Tuning with HistGradientBoostingRegressor
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

print('Model customization complete')

# Set up cross-validation
tscv = TimeSeriesSplit(n_splits=5)

# Run randomized search
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_distributions,
    n_iter=20,
    cv=tscv,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=1,
    random_state=42
)

print('Randomized search setup complete, beginning fit')

# Fit to the training data
random_search.fit(X_train, y_train)

# 6. Evaluating the Best Model
best_model = random_search.best_estimator_

# Make predictions on the test set
y_pred = best_model.predict(X_test)

# Evaluate performance
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')