import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor


def run_xgboost():
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

    num_features_df = pd.DataFrame(num_features_imputed, columns=numeric_columns, index=num_features.index)
    categorical_features_df = pd.DataFrame(categorical_features_imputed, columns=categorical, index=df.index)

    X = pd.concat([categorical_features_df, num_features_df], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    best_r2 = float('-inf')
    best_mse = float('inf')
    best_params = {}

    results = []

    learning_rates = [0.01, 0.1, 0.2]
    max_depths = [3, 6, 9, 12]
    reg_lambdas = [0, 0.001, 0.01, 0.1, 1, 10]

    for lr in learning_rates:
        for depth in max_depths:
            for reg_lambda in reg_lambdas:
                model = XGBRegressor(objective='reg:squarederror', learning_rate=lr, max_depth=depth, n_estimators=100,
                                     reg_lambda=reg_lambda)
                model.fit(X_train, y_train)

                y_train_pred = model.predict(X_train)
                train_mse = mean_squared_error(y_train, y_train_pred)

                y_pred = model.predict(X_test)
                test_mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                print(
                    f'Learning Rate: {lr}, Max Depth: {depth}, Reg Lambda: {reg_lambda}, Train MSE: {train_mse:.2f}, Test MSE: {test_mse:.2f}, R2: {r2:.2f}')
                if test_mse < best_mse:
                    best_r2 = r2
                    best_mse = test_mse
                    best_params = {'learning_rate': lr, 'max_depth': depth, 'reg_lambda': reg_lambda}

                results.append({
                    "Learning Rate": lr,
                    "Max Depth": depth,
                    "Reg Lambda": reg_lambda,
                    "MSE": test_mse,
                    "R^2": r2
                })

#     return best_r2, best_mse, best_params
    results_df = pd.DataFrame(results)
    return results_df.to_latex(index=False, caption="XGBoost Regression Metrics")

results_table = run_xgboost()
print(results_table)
# best_r2, best_mse, best_params = run_xgboost()
# print(f'Best R^2: {best_r2:.2f}')
# print(f'Best MSE: {best_mse:.2f}')
# print(f'Best Parameters: {best_params}')