import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor

def run_knn():
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

    k_values = range(1, 21)
    best_mse = float('inf')
    best_r2 = -float('inf')
    best_k = None

    k_values = range(1, 21)
    scores = []

    results = []

    for k in k_values:
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsRegressor(n_neighbors=k))
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        scores.append(r2)
        mse = mean_squared_error(y_test, y_pred)

        if mse < best_mse:
            best_mse = mse
            best_r2 = r2
            best_k = k

        print(f'k: {k}, MSE: {mse:.2f}')

        results.append({
            "k": k,
            "MSE": mse,
            "R^2": r2
        })

    results_df = pd.DataFrame(results)
    return results_df.to_latex(index=False, caption="KNN Regression Metrics")

results_table = run_knn()
print(results_table)
# best_r2, best_mse, best_params = run_knn()
# print(f'Best R^2: {best_r2:.2f}')
# print(f'Best MSE: {best_mse:.2f}')
# print(f'Best Parameters: {best_params}')
    # Plotting the results
    # plt.figure(figsize=(10, 5))
    # plt.plot(k_values, scores, marker='o')
    # plt.title('KNN Regressor Performance')
    # plt.xlabel('Number of Neighbors (k)')
    # plt.ylabel('$R^2$ Score')
    # plt.xticks(k_values)
    # plt.grid(True)
    # plt.show()

