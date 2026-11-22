import pandas as pd
import plotly.express as px

from polynomial_fitting import PolynomialFitting


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, parse_dates=['Date'])

    df =df[df['Temp'] > -70]

    df['DayOfYear']=df['Date'].dt.dayofyear

    return df


if __name__ == '__main__':
    # Question 2 - Load and preprocessing of city temperature dataset
    df = load_data("city_temperature.csv")

    # Question 3 - Exploring data for specific country
    israel_df = df[df['Country'] == 'Israel']

    fig1 = px.scatter(israel_df, x='DayOfYear', y='Temp', color=israel_df['Year'].astype(str),
                      title="Daily Temperature in Israel as Function of Day of Year",
                      labels={"Temp": "Temperature", "DayOfYear": "Day of Year"})
    fig1.show()

    stds_per_month = israel_df.groupby('Month')['Temp'].agg('std').reset_index()
    fig2 = px.bar(stds_per_month, x='Month', y='Temp',
                  title="Standard Deviation of Daily Temperatures per Month (Israel)",
                  labels={"Temp": "Standard Deviation", "Month": "Month"})
    fig2.show()

    # Question 4 - Fitting model for different values of `k`

    train_df = israel_df.sample(frac=0.75, random_state=42)
    test_df = israel_df.drop(train_df.index)

    X_train, y_train = train_df['DayOfYear'], train_df['Temp']
    X_test, y_test = test_df['DayOfYear'], test_df['Temp']

    # Normalize X
    X_mean = X_train.mean()
    X_std = X_train.std()

    X_train_norm = (X_train - X_mean) / X_std
    X_test_norm = (X_test - X_mean) / X_std

    losses = []

    for k in range(1, 11):
        model = PolynomialFitting(k)
        model.fit(X_train_norm.to_numpy(), y_train.to_numpy())
        loss = round(model.loss(X_test_norm.to_numpy(), y_test.to_numpy()), 2)
        losses.append(loss)
        print(f"Test error for k={k}: {loss}")

    fig3 = px.bar(x=list(range(1, 11)), y=losses,
                  title="Test Error for Different Polynomial Degrees (k)",
                  labels={"x": "Degree k", "y": "MSE Loss"})
    fig3.show()