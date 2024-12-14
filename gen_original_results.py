import numpy as np
from susie import susie  # 假设你的包名为susiepy
from elbo import get_objective, estimate_residual_variance, SER_posterior_e_loglik, get_ER2, Eloglik        
from single_effect_regression import single_effect_regression
from test_helper import set_X_attributes
from base import SusieObject
import os
from susie_utils import susie_get_objective, susie_get_posterior_mean
from update_each_effect import update_each_effect
def create_sparsity_mat(sparsity, n, p):
    nonzero = round(n * p * (1 - sparsity))
    nonzero_idx = np.random.choice(n * p, nonzero, replace=False)
    mat = np.zeros(n * p)
    mat[nonzero_idx] = 1
    return mat.reshape((n, p))

# 设置随机种子
np.random.seed(1)

# 基本参数
n = 100
p = 200
beta = np.zeros(p)
beta[0:4] = 10

# 生成数据
X_dense = create_sparsity_mat(0.99, n, p)
y = X_dense @ beta + np.random.normal(size=n)
# 模型参数
L = 10
residual_variance = 0.8
scaled_prior_variance = 0.2

# 初始化模型参数
s = SusieObject()
s.alpha = np.full((L, p), 1/p)
s.mu = np.full((L, p), 2)
s.mu2 = np.full((L, p), 3)
s.Xr = np.full(n, 5)
s.KL = np.full(L, 1.2)
s.sigma2 = residual_variance
s.V = np.repeat(scaled_prior_variance * np.var(y), L)
print("Prior variances (V):", s.V)
s.lbf = np.full(L, np.nan)
s.lbf_variable = np.full((L, p), np.nan)
X = set_X_attributes(X_dense) 
Eb = np.ones(p)
Eb2 = np.ones(p)
s2 = residual_variance
V = scaled_prior_variance



# 确保结果目录存在
results_dir = '/home/jwudt/final_project/results'
os.makedirs(results_dir, exist_ok=True)

# objective_original_res = susie_get_objective(s)
# print(objective_original_res)
# np.save(os.path.join(results_dir, 'objective_original_res.npy'), objective_original_res)

# Eloglik_original_res = Eloglik(X, y, s)
# print(Eloglik_original_res)
# np.save(os.path.join(results_dir, 'Eloglik_original_res.npy'), Eloglik_original_res)

# ER2_original_res = get_ER2(X, y, s)
# print(ER2_original_res)
# np.save(os.path.join(results_dir, 'ER2_original_res.npy'), ER2_original_res)

# SER_original_res = SER_posterior_e_loglik(X, y, s2, Eb, Eb2)
# print(SER_original_res)
# np.save(os.path.join(results_dir, 'SER_original_res.npy'), SER_original_res)

# singleReg_original_res = single_effect_regression(y, X, V)
# print(singleReg_original_res)
# np.save(os.path.join(results_dir, 'singleReg_original_res.npy'), singleReg_original_res)

# vbupdate_original_res = update_each_effect(X, y, s)
# print(vbupdate_original_res)
# np.save(os.path.join(results_dir, 'vbupdate_original_res.npy'), vbupdate_original_res)

# 运行不同配置的susie
# susiefit_original_res = susie(X_dense, y)
# print(susiefit_original_res)
# print("cs", susiefit_original_res.sets)
# np.save(os.path.join(results_dir, 'susiefit_original_res.npy'), susiefit_original_res)

susiefit_original_res2 = susie(X_dense, y, standardize=True, intercept=False)
print(susiefit_original_res2)
print("cs", susiefit_original_res2.sets)
# susiefit_original_res3 = susie(X_dense, y, standardize=False, intercept=True)
# print(susiefit_original_res3)
# susiefit_original_res4 = susie(X_dense, y, standardize=False, intercept=False)
# print(susiefit_original_res4)

np.save(os.path.join(results_dir, 'susiefit_original_res2.npy'), susiefit_original_res2)
# np.save(os.path.join(results_dir, 'susiefit_original_res3.npy'), susiefit_original_res3)
# np.save(os.path.join(results_dir, 'susiefit_original_res4.npy'), susiefit_original_res4)