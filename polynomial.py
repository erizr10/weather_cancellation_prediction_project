import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def run_poly():
    cancellation_title = 'FLIGHT_CANCELLATION_AND_DELAY_COMBINED_RATE'
    df = pd.read_csv('combined_data.csv')

    df = df.drop(['FL_DATE', 'DATE', 'STATION', 'NAME', 'PGTM', 'TAVG'], axis=1)

    categorical = ['WT01', 'WT02', 'WT03', 'WT04', 'WT05', 'WT06', 'WT08', 'WT09', 'WT11', 'WT10', 'WT13', 'WT14',
                   'WT15', 'WT16', 'WT17', "WT18", "WT19", "WT21", "WT22"]

    numeric_columns = df.columns.difference(categorical + [cancellation_title])
    num_features = df[numeric_columns]

    categorical_features = df[categorical]

    num_imputer = SimpleImputer(strategy='mean')
    cat_imputer = SimpleImputer(strategy='constant', fill_value=0)

    num_features_imputed = num_imputer.fit_transform(num_features)
    categorical_features_imputed = cat_imputer.fit_transform(categorical_features)

    y = df[cancellation_title]

    best_mse = float('inf')
    best_r2 = -float('inf')
    best_deg = None

    results = []


    for poly_degree in range(2,6):
        poly_features = PolynomialFeatures(degree=poly_degree, include_bias=False)

        num_features_poly = poly_features.fit_transform(num_features_imputed)

        num_features_df = pd.DataFrame(num_features_poly, columns=poly_features.get_feature_names_out(),
                                       index=num_features.index)
        categorical_features_df = pd.DataFrame(categorical_features_imputed, columns=categorical, index=df.index)

        X = pd.concat([categorical_features_df, num_features_df], axis=1)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

        pipeline = Pipeline([
            ('scaler', StandardScaler()), 
            ('ridge', Ridge(alpha=10*10**poly_degree))  
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f'\nPolynomial Degree: {poly_degree}')
        #print('Coefficients:', model.coef_)
        print('Mean squared error: %.2f' % mse)
        print('Coefficient of determination: %.2f' % r2)

        if mse < best_mse:
            best_mse = mse
            best_r2 = r2
            best_deg = poly_degree

        results.append({'Degree': poly_degree, 'MSE': mse, 'R^2': r2})
        
    results_df = pd.DataFrame(results)
    return results_df.to_latex(index=False, caption="Polynomial Regression Metrics", label="tab:poly_metrics")
    # return best_r2, best_mse, best_deg
results_table = run_poly()
print(results_table)
# best_r2, best_mse, best_params = run_poly()
# print(f'Best R^2: {best_r2:.2f}')
# print(f'Best MSE: {best_mse:.2f}')
# print(f'Best Parameters: {best_params}')


