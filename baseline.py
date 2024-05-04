import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler



def run_baseline():
    df = pd.read_csv('combined_data.csv')

    df = df.drop(['FL_DATE', 'DATE', 'STATION', 'NAME', 'PGTM', 'TAVG'], axis=1)

    #df['flight cancellation rate'] = df['flight cancellation rate'] * 100

    categorical = ['WT01', 'WT02', 'WT03', 'WT04', 'WT05', 'WT06', 'WT08', 'WT09','WT11', 'WT10', 'WT13', 'WT14', 'WT15', 'WT16', 'WT17', "WT18", "WT19", "WT21", "WT22"]

    num_features = df.drop(categorical + ['FLIGHT_CANCELLATION_AND_DELAY_COMBINED_RATE'], axis=1)

    num_imputer = SimpleImputer(strategy='mean')
    num_features_imputed = num_imputer.fit_transform(num_features)

    num_features_df = pd.DataFrame(num_features_imputed, columns=num_features.columns, index=num_features.index)

    cat_imputer = SimpleImputer(strategy='constant', fill_value=0)
    categorical_features_imputed = cat_imputer.fit_transform(df[categorical])

    categorical_features_df = pd.DataFrame(categorical_features_imputed, columns=categorical, index=df.index)

    X = pd.concat([num_features_df, categorical_features_df], axis=1)

    y = df['FLIGHT_CANCELLATION_AND_DELAY_COMBINED_RATE']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),  
        ('ridge', Ridge(alpha=10.0))  
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    # model = LinearRegression()
    # model.fit(X_train, y_train)
    #
    # y_pred = model.predict(X_test)

    # Output the model's coefficients and performance metrics
    #print('Coefficients:', model.coef_)

    y_train_pred = pipeline.predict(X_train)

    train_error = mean_squared_error(y_train, y_train_pred)

    print('Training error: %.2f' % train_error)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    params = {}

    print('Mean squared error: %.2f' % mse)
    print('Coefficient of determination: %.2f' % r2)

    return r2, mse, params

best_r2, best_mse, best_params = run_baseline()
print(f'Best R^2: {best_r2:.2f}')
print(f'Best MSE: {best_mse:.2f}')
print(f'Best Parameters: {best_params}')