import pandas as pd
import pickle
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
import sys
import os
# Function to load data from the SQLite database
def load_data(database_filepath):
    """
    Loads data from an SQLite database.

    Parameters:
    database_filepath (str): Path to the SQLite database file.

    Returns:
    X (pd.DataFrame): Features for training.
    y (pd.Series): Target variable (next day's stock price).
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    table_name = f"{os.path.basename(database_filepath).replace('.db', '')}_table"
    print(f"Loading data from table: {table_name}")
    
    df = pd.read_sql_table(table_name, engine)
    df.columns = df.columns.astype(str)
    X = df.drop(columns=['Tomorrow'])
    y = df['Tomorrow']
    return X, y


# Function to build the regression model with GridSearchCV for hyperparameter tuning
def build_model():
    model = RandomForestRegressor()
    parameters = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    cv = GridSearchCV(model, param_grid=parameters, scoring='neg_mean_squared_error', cv=5)
    return cv

# Function to evaluate the model's performance
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
    print("Mean Squared Error:", mean_squared_error(y_test, y_pred))

# Function to save the model to a pickle file
def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)

# Main function to run the whole pipeline
def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        
        print(f'Loading data from database: {database_filepath}')
        X, y = load_data(database_filepath)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test)
        
        print(f'Saving model to {model_filepath}')
        save_model(model, model_filepath)
        
        print('Trained model saved!')

    else:
        print('Please provide the filepath of the database as the first argument and '\
              'the filepath to save the model as the second argument. \n\nExample: python train_classifier.py '\
              'StockData.db stock_model.pkl')

if __name__ == '__main__':
    main()
