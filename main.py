import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline 
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


def train_model():
    """Format and train model for prediction
    """

    # Get data and remove any rows with missing values
    atlanta_weather = pd.read_csv("atlanta_weather.csv")
    atlanta_weather = atlanta_weather.dropna()
    # print(atlanta_weather.head())

    # Isolate target variable
    X = atlanta_weather.drop(["TAVG"], axis=1)
    X = X.drop(["STATION"], axis=1)
    X = X.drop(["NAME"], axis=1)
    X = X.drop(["TMAX"], axis=1)
    X = X.drop(["TMIN"], axis=1)
    fix_array(X)

    y = atlanta_weather["TAVG"]

    print(X.head())

    # Create training and target sets
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    # Standarize data to Gaussian distribution
    # scaler = StandardScaler()
    # X_train_scaled = scaler.fit_transform(X_train)

    # tree_model = DecisionTreeRegressor() 
    # tree_model.fit = (X_train, y_train)
 
if __name__ == "__main__": 
    train_model()

