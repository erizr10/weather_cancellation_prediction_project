import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR

def run_svr():
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

    C_values = [0.1, 1, 10, 100]
    kernels = ['linear', 'rbf', 'poly']
    results = []
    mse_values = []

    best_r2 = float('inf')
    best_mse = float('inf')
    best_params = {}
    

    for C in C_values:
        for kernel in kernels:
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('svr', SVR(C=C, kernel=kernel))
            ])

            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            results.append({
                "C": C,
                "Kernel": kernel,
                "MSE": mse,
                "R^2": r2
            })

            mse_values.append(mse)

            if mse < best_mse:
                best_mse = mse
                best_r2 = r2
                best_params = {'C': C, 'kernel': kernel, 'MSE': mse, 'R2': r2}

            print(f'C: {C}, Kernel: {kernel}, MSE: {mse:.2f}, R2: {r2:.2f}')

# return best_r2, best_mse, best_params
    results_df = pd.DataFrame(results)
    return results_df.to_latex(index=False, caption="SVR Regression Metrics")

results_table = run_svr()
print(results_table)
# best_r2, best_mse, best_params = run_svr()
# print(f'Best R^2: {best_r2:.2f}')
# print(f'Best MSE: {best_mse:.2f}')
# print(f'Best Parameters: {best_params}')