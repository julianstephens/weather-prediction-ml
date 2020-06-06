import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline 
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import joblib 
import datetime


def make_date_str(date):
    """Convert dates from Y-m-d to Ymd
    
    :param date (str): A Y-m-d formatted date 

    Returns
        date_str(str): A Ymd formatted date
    """
    date_arr = date.split('-')
    date_str = ''
 
    first = True
    for itm in date_arr: 
        if first: 
            first = False
        else:
            date_str = date_str + itm

    return date_str

def fix_array(arr):
    """Converts all dates in array to Ymd format

    :param arr (arr): An array of dates to be converted
    """
    new_arr = []

    for date in arr['DATE']:
        new_arr.append(make_date_str(date))
    
    arr['DATE'] = new_arr

def train_model():
    """Format and train model for prediction
    """
    # Get data and remove any rows with missing values
    atlanta_weather = pd.read_csv("atlanta_weather.csv")
    atlanta_weather = atlanta_weather.dropna()

    # Isolate target variable
    X = atlanta_weather.drop(["TAVG"], axis=1)
    X = X.drop(["STATION"], axis=1)
    X = X.drop(["NAME"], axis=1)
    X = X.drop(["TMAX"], axis=1)
    X = X.drop(["TMIN"], axis=1)

    # Convert dates to proper format
    fix_array(X)

    y = atlanta_weather["TAVG"]

    # print(X.head())

    # Create training and target sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    # Standarize data to Gaussian distribution
    # scaler = StandardScaler()
    # X_train_scaled = scaler.fit_transform(X_train)

    # Create decision tree model and fit it
    tree_model = DecisionTreeRegressor() 
    tree_model.fit(X_train, y_train)

    # Save model for faster predictions
    joblib.dump(tree_model, 'weather_predictor.pkl')
    print("-" * 40)
    print("\nDone training!\n")
    print("-" * 40)
    
def predict_weather():
    """Use trained model to predict weather"""
    # Load trained model
    tree_model = joblib.load('weather_predictor.pkl')

    # Get user input for prediction date
    print("-" * 40)
    print("Enter a date below.\n")

    option = input("MM: ")
    month = option
    option = input("DD: ")
    day = option

    date = str(month) + str(day)

    # Predict and report temperature
    predicted_temp = tree_model.predict([[date]])[0]
    print("-" * 40)
    print("\nPredicted temperature: " + str(predicted_temp) + "\n")
    print("-" * 40)

def select_action(option): 
    """Run the function for the user's selected option.

    :param option (int): The action that the user wants to complete
    """
    if option == '1': 
        train_model()
    elif option == '2':
        predict_weather()

def menu(): 
    """List of actions for user to select from"""
    print("*" * 40)
    print("\nWhat would you like to do?\n")
    print("1. Train the model.\n")
    print("2. Predict the weather.\n")
    print("*" * 40)

    option = input("Enter option: ")

    while True: 
        if option == '1' or option == '2' or option == '9':
            break
         
        option = input("Enter option: ")
    
    return option

if __name__ == "__main__": 
    while True: 
        option = menu()

        if option == '9': 
            break
        else:
            select_action(option)
