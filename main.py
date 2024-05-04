

from baseline import run_baseline
from polynomial import run_poly
from mlp import run_mlp
from knn_reg import  run_knn
from svr import  run_svr
from my_xgboost import  run_xgboost
from rforest import run_random_forest

import os


if __name__ == '__main__':
    #To run a specific model uncomment the model you want to run
    run_baseline()
    #run_poly()
    #run_mlp()
    #run_knn()
    #run_svr()
    #run_xgboost()
    #run_random_forest
    



