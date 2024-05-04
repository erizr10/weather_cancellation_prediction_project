import matplotlib.pyplot as plt

from my_xgboost import run_xgboost
from polynomial import run_poly
from rforest import run_random_forest
from mlp import run_mlp
from baseline import run_baseline
from svr import run_svr
from knn_reg import run_knn

models = [
    run_xgboost,
    run_poly,
    run_random_forest,
    run_mlp,
    run_baseline,
    run_svr,
    run_knn
]

model_names = [
    'XGBoost',
    'Polynomial',
    'Random Forest',
    'MLP',
    'Baseline',
    'SVR',
    'KNN'
]

results = {
    'model': [],
    'mse': [],
    'r2': []
}

for model, name in zip(models, model_names):
    result = model() 

    if isinstance(result, tuple):
        result = {'r2': result[0], 'mse': result[1], 'params': result[2]} 

    results['model'].append(name)
    results['mse'].append(result['mse'])
    results['r2'].append(result['r2'])
    print(f"{name} - Best MSE: {result['mse']}, Best R2: {result['r2']}, Best Parameters: {result['params']}")

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Model')
ax1.set_ylabel('MSE', color=color)
ax1.plot(results['model'], results['mse'], color=color, marker='o', label='MSE')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_xticklabels(results['model'], rotation=45, ha="right")

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('R2', color=color)
ax2.plot(results['model'], results['r2'], color=color, marker='x', label='R2')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.title('Performance of Different Models')
plt.show()
