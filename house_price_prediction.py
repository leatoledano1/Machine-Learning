import pandas as pd
from sklearn.model_selection import train_test_split

from linear_regression import LinearRegression
import os
from utils import *


def preprocess_train(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """
    preprocess training data.
    Parameters
    ----------
    X: pd.DataFrame
        the loaded data
    y: pd.Series

    Returns
    -------
    A clean, preprocessed version of the data
    """
    df = X.copy()
    df['price'] = y.values

    df = df.drop(columns=['id', 'date'], errors='ignore')

    df = df.dropna()
    df = df[df['price'] > 0]
    df = df[df['bedrooms'] > 0]
    df = df[df['bathrooms'] > 0]
    df = df[df['sqft_living'] > 0]
    df = df[df['sqft_lot'] > 0]
    df = df[df['floors'] > 0]
    df = df[df['yr_built'] > 0]

    invalid_renovation = (df['yr_renovated'] != 0) & (df['yr_renovated'] < df['yr_built'])
    df = df[~invalid_renovation]

    current_year = 2015
    df['years_since_built'] = current_year - df['yr_built']
    df['recently_renovated'] = (df['yr_renovated'] > 2000).astype(int)

    df = df.drop(columns=['yr_built', 'yr_renovated'], errors='ignore')

    y_clean = df['price']
    X_clean = df.drop(columns=['price'])

    return X_clean, y_clean


def preprocess_test(X: pd.DataFrame) -> pd.DataFrame:
    """
    preprocess test data. You are not allowed to remove rows from X, but only edit its columns.
    Parameters
    ----------
    X: pd.DataFrame
        the loaded data

    Returns
    -------
    A preprocessed version of the test data that matches the coefficients format.
    """
    df = X.copy()

    df = df.drop(columns=['id', 'date'], errors='ignore')

    for col in df.columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    current_year = 2015
    df['years_since_built'] = current_year - df['yr_built']
    df['recently_renovated'] = (df['yr_renovated'] > 2000).astype(int)

    df = df.drop(columns=['yr_built', 'yr_renovated'], errors='ignore')

    return df

def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> None:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    os.makedirs(output_path, exist_ok=True)
    for col in X.columns:
        pearson_corr = np.corrcoef(X[col], y)[0, 1]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=X[col],
            y=y,
            mode='markers',
            marker=dict(size=3, opacity=0.5)))
        fig.update_layout(
            title=f"Feature: {col} | Pearson Correlation: {pearson_corr:.4f}",
            xaxis_title=col,
            yaxis_title="Price")
        fig.write_image(f"{output_path}/{col}.png")

if __name__ == '__main__':
    df = pd.read_csv("house_prices.csv")
    X, y = df.drop("price", axis=1), df.price

    # Question 2
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # Question 3
    X_train, y_train = preprocess_train(X_train_raw, y_train_raw)

    # Question 4
    X_test = preprocess_test(X_test_raw)

    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    # Question 5
    percentages = range(10, 101)
    mean_losses = []
    std_losses = []

    for p in percentages:
        losses = []
        for _ in range(10):
            sample = X_train.sample(frac=p / 100, random_state=None)
            y_sample = y_train.loc[sample.index]

            model = LinearRegression(include_intercept=True)
            model.fit(sample.to_numpy(), y_sample.to_numpy())
            loss = model.loss(X_test.to_numpy(), y_test_raw.to_numpy())
            losses.append(loss)

        mean_losses.append(np.mean(losses))
        std_losses.append(np.std(losses))

    mean_losses = np.array(mean_losses)
    std_losses = np.array(std_losses)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(percentages),
        y=mean_losses,
        mode='lines',
        name='Mean Loss',
        line=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=list(percentages) + list(percentages)[::-1],
        y=list(mean_losses + 2 * std_losses) + list(mean_losses - 2 * std_losses)[::-1],
        fill='toself',
        fillcolor='rgba(0, 100, 255, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Confidence Interval (±2std)'))
    fig.update_layout(
        title="MSE Loss vs Training Set Size",
        xaxis_title="Training Set Percentage (%)",
        yaxis_title="MSE Loss")
    fig.show()


