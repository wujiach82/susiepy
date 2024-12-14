from typing import Optional, Union
import numpy as np
from single_effect_regression import single_effect_regression
from sparse_multiplication import compute_Xb
from elbo import SER_posterior_e_loglik
from susie import SusieObject
from base import ScaledMatrix

def update_each_effect(X: Union[np.ndarray, ScaledMatrix], 
                      y: np.ndarray, 
                      s: SusieObject, 
                      estimate_prior_variance: bool = False, 
                      estimate_prior_method: str = "optim",
                      check_null_threshold: Optional[float] = None) -> SusieObject:
    """更新每个效应一次
    
    Args:
        X: shape (n, p)的回归变量矩阵，可以是np.ndarray或ScaledMatrix
        y: shape (n,)的响应变量向量
        s: 包含当前拟合结果的SusieObject实例
        estimate_prior_variance: 是否估计先验方差
        estimate_prior_method: 估计先验的方法
        check_null_threshold: 用于比较当前估计和空模型似然度的对数尺度阈值
        
    Returns:
        更新后的SusieObject实例
    """
    if not estimate_prior_variance:
        estimate_prior_method = "none"
        
    # 对每个效应进行更新
    L = s.alpha.shape[0]  # 效应数量
    if L > 0:
        for l in range(L):
            # 从拟合值中移除第l个效应
            s.Xr = s.Xr - compute_Xb(X, s.alpha[l] * s.mu[l])
            
            # 计算残差
            R = y - s.Xr
            
            # 更新单个效应
            res = single_effect_regression(R, X, s.V[l], s.sigma2, s.pi,
                                         estimate_prior_method,
                                         check_null_threshold)
            
            # 更新后验均值的变分估计
            s.mu[l] = res['mu']
            s.alpha[l] = res['alpha']
            s.mu2[l] = res['mu2']
            s.V[l] = res['V']
            s.lbf[l] = res['lbf_model']
            s.lbf_variable[l] = res['lbf']
            s.KL[l] = (-res['loglik'] + 
                      SER_posterior_e_loglik(X, R, s.sigma2, 
                                           res['alpha'] * res['mu'],
                                           res['alpha'] * res['mu2']))
            
            # 添加回更新后的效应
            s.Xr = s.Xr + compute_Xb(X, s.alpha[l] * s.mu[l])
            
    return s