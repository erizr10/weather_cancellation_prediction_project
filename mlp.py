from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import pandas as pd


def run_mlp():
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

    activation = ['logistic', 'tanh', 'relu']
    solver = ['sgd']
    hidden_layer_sizes = [(50,), (100,), (50, 50), (100, 50), (200,)]
    alphas = [0.001, 0.01, 0.1, 1, 10]
    print()

    best_mse = float('inf')
    best_r2 = -float('inf')
    best_settings = {}
    results = []

    for layer in hidden_layer_sizes:
        for alpha in alphas:
            for a in activation:
                for s in solver:
                    pipeline = Pipeline([
                        ('scaler', StandardScaler()), 
                        ('mlp', MLPRegressor(hidden_layer_sizes=layer,
                                             alpha=alpha,
                                             activation=a,
                                             solver=s,
                                             max_iter=2000,
                                             random_state=1))
                    ])

                    pipeline.fit(X_train, y_train)
                    y_pred = pipeline.predict(X_test)

                    print(f'Layers: {layer}, Alpha: {alpha}, Solver: {s}, Activation: {a}')
                    mse =mean_squared_error(y_test, y_pred)
                    print('Mean squared error: %.2f' % mse)
                    r2 = r2_score(y_test, y_pred)
                    print('Coefficient of determination: %.2f' % r2)

                    if mse < best_mse:
                        best_mse = mse
                        best_r2 = r2
                        best_settings = {
                            "hidden_layer_sizes": layer,
                            "activation": a,
                            "solver": s,
                            "alpha": alpha,
                        }

                    results.append({
                        "Layers": layer,
                        "Alpha": alpha,
                        "Solver": s,
                        "Activation": a,
                        "MSE": mse,
                        "R^2": r2
                    })

    # return best_r2, best_mse, best_settings
    results_df = pd.DataFrame(results)
    return results_df.to_latex(index=False, caption="MLP Regression Metrics")

# best_r2, best_mse, best_params = run_mlp()
# print(f'Best R^2: {best_r2:.2f}')
# print(f'Best MSE: {best_mse:.2f}')
# print(f'Best Parameters: {best_params}')
results_table = run_mlp()
print(results_table)