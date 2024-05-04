from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import pandas as pd
import matplotlib.pyplot as plt

def run_random_forest():
    cancellation_title = 'FLIGHT_CANCELLATION_AND_DELAY_COMBINED_RATE'
    df = pd.read_csv('combined_data.csv')

    df = df.drop(['FL_DATE', 'DATE', 'STATION', 'NAME', 'PGTM', 'TAVG'], axis=1)

    categorical = ['WT01', 'WT02', 'WT03', 'WT04', 'WT05', 'WT06', 'WT08', 'WT09','WT11', 'WT10', 'WT13', 'WT14', 'WT15', 'WT16', 'WT17', "WT18", "WT19", "WT21", "WT22"]

    num_features = df.drop(categorical + [cancellation_title], axis=1)

    num_imputer = SimpleImputer(strategy='mean')
    num_features_imputed = num_imputer.fit_transform(num_features)
    num_features_df = pd.DataFrame(num_features_imputed, columns=num_features.columns, index=num_features.index)

    cat_imputer = SimpleImputer(strategy='constant', fill_value=0)
    categorical_features_imputed = cat_imputer.fit_transform(df[categorical])
    categorical_features_df = pd.DataFrame(categorical_features_imputed, columns=categorical, index=df.index)

    X = pd.concat([num_features_df, categorical_features_df], axis=1)

    y = df[cancellation_title]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    best_mse = float('inf')
    best_r2 = -float('inf')
    best_settings = {}

    n_estimators_options = [100, 200, 300, 400, 500]
    max_features_options = ['sqrt', 'log2', None] 
    max_depth_options = [None, 10, 20, 30]

    results = []


    for n_estimators in n_estimators_options:
        for max_features in max_features_options:
            for max_depth in max_depth_options:

                model = RandomForestRegressor(n_estimators=n_estimators, max_features=max_features, max_depth=max_depth, random_state=1)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                # results.append((n_estimators, max_features, max_depth, mse, r2))

                print(f'Estimators: {n_estimators}, Max Features: {max_features}, Max Depth: {max_depth}')
                print('Mean squared error: %.2f' % mse)
                print('Coefficient of determination: %.2f' % r2)

                if mse < best_mse:
                    best_mse = mse
                    best_r2 = r2
                    best_settings = {
                        "n_estimators": n_estimators,
                        "max_features": max_features,
                        "max_depth": max_depth,
                    }
                results.append({
                    "n_estimators": n_estimators,
                    "max_features": max_features,
                    "max_depth": max_depth,
                    "MSE": mse,
                    "R^2": r2
                })
    # return best_r2, best_mse, best_settings

    results_df = pd.DataFrame(results)
    return results_df.to_latex(index=False, caption="RForests Regression Metrics", label="tab:poly_metrics")

results_table = run_random_forest()
print(results_table)

# best_r2, best_mse, best_settings = run_random_forest()

# def plot_r2_scores(results):
#     labels = [f"Est: {est}, MF: {mf}, MD: {md}" for est, mf, md, _, _ in results]
#     r2_scores = [r2 for _, _, _, _, r2 in results]

#     plt.figure(figsize=(14, 7))
#     plt.plot(labels, r2_scores, marker='o', linestyle='-', color='b')
#     plt.xticks(rotation=45, ha='right')  # Rotate labels to fit them better
#     plt.xlabel('Model Configuration')
#     plt.ylabel('R^2 Score')
#     plt.title('R^2 Scores for Various Random Forest Configurations')
#     plt.grid(True, which='both', linestyle='--', linewidth=0.5)
#     plt.tight_layout()  # Adjust layout to make room for label rotation
#     plt.show()

# plot_r2_scores(results)
