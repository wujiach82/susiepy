import numpy as np
from typing import Optional, Union
from base import SusieObject, ScaledMatrix
from sparse_multiplication import compute_Xb

def susie_init_coef(coef_index: np.ndarray, coef_value: np.ndarray, p: int) -> SusieObject:
    """使用回归系数初始化susie对象
    
    Args:
        coef_index: 非零系数的索引数组(L维向量)
        coef_value: 初始系数估计值数组(L维向量)
        p: 变量数量
        
    Returns:
        初始化的SusieObject实例
    """
    L = len(coef_index)
    if L <= 0:
        raise ValueError("至少需要一个非零效应")
    if not np.all(coef_value != 0):
        raise ValueError("输入的coef_value所有元素都必须非零")
    if L != len(coef_value):
        raise ValueError("输入的coef_index和coef_value长度必须相同")
    if np.max(coef_index) > p:
        raise ValueError("输入的coef_index超出了p的范围")
    
    alpha = np.zeros((L, p))
    mu = np.zeros((L, p))
    
    for i in range(L):
        alpha[i, coef_index[i]-1] = 1  # R到Python的索引转换
        mu[i, coef_index[i]-1] = coef_value[i]
    
    s = SusieObject()
    s.alpha = alpha
    s.mu = mu
    s.mu2 = mu * mu
    return s

def init_setup(n: int, p: int, L: int, scaled_prior_variance: float, 
              residual_variance: Optional[float], prior_weights: Optional[np.ndarray],
              null_weight: Optional[float], varY: float, standardize: bool) -> SusieObject:
    """设置susie的默认初始化
    
    Args:
        n: 样本数量
        p: 变量数量
        L: 效应数量
        scaled_prior_variance: 缩放的先验方差
        residual_variance: 残差方差
        prior_weights: 先验权重
        null_weight: 空效应权重
        varY: Y的方差
        standardize: 是否标准化
        
    Returns:
        初始化的SusieObject实例
    """
    if not isinstance(scaled_prior_variance, (int, float)) or scaled_prior_variance < 0:
        raise ValueError("缩放的先验方差应为正数")
    
    if scaled_prior_variance > 1 and standardize:
        raise ValueError("当standardize=True时，缩放的先验方差不应大于1")
    
    if residual_variance is None:
        residual_variance = varY
        
    if prior_weights is None:
        prior_weights = np.repeat(1/p, p)
    else:
        if np.all(prior_weights == 0):
            raise ValueError("至少一个变量的先验权重应大于0")
        prior_weights = prior_weights / np.sum(prior_weights)
    
    if len(prior_weights) != p:
        raise ValueError("先验权重的长度必须为p")
        
    if p < L:
        L = p
        
    s = SusieObject()
    s.alpha = np.full((L, p), 1/p)
    s.mu = np.zeros((L, p))
    s.mu2 = np.zeros((L, p))
    s.Xr = np.zeros(n)
    s.KL = np.full(L, np.nan)
    s.lbf = np.full(L, np.nan)
    s.lbf_variable = np.full((L, p), np.nan)
    s.sigma2 = residual_variance
    s.V = scaled_prior_variance * varY
    s.pi = prior_weights
    s.null_index = 0 if null_weight is None else p
    
    return s

def init_finalize(s: SusieObject, X: Optional[Union[np.ndarray, ScaledMatrix]] = None, 
                 Xr: Optional[np.ndarray] = None) -> SusieObject:
    """更新susie拟合对象以初始化susie模型
    
    Args:
        s: SusieObject实例
        X: 设计矩阵
        Xr: 残差
        
    Returns:
        更新后的SusieObject实例
    """
    if np.isscalar(s.V) or len(np.atleast_1d(s.V)) == 1:
        s.V = np.repeat(s.V, s.alpha.shape[0])

    # 检查sigma2
    if not isinstance(s.sigma2, (int, float)):
        raise ValueError("输入的残差方差sigma2必须为数值型")
    
    # 避免1x1矩阵输入时的维度问题
    s.sigma2 = float(s.sigma2)
    if not np.isscalar(s.sigma2):
        raise ValueError("输入的残差方差sigma2必须为标量")
    if s.sigma2 <= 0:
        raise ValueError("残差方差sigma2必须为正数(您的var(Y)是否为零？)")
    
    # 检查先验方差
    if not isinstance(s.V, (np.ndarray, list)):
        raise ValueError("输入的先验方差必须为数值型")
    if not np.all(s.V >= 0):
        raise ValueError("先验方差必须非负")
    if not np.all(s.mu.shape == s.mu2.shape):
        raise ValueError("输入对象中mu和mu2的维度不匹配")
    if not np.all(s.mu.shape == s.alpha.shape):
        raise ValueError("输入对象中mu和alpha的维度不匹配")
    if s.alpha.shape[0] != len(s.V):
        raise ValueError("输入的先验方差V的长度必须等于alpha的行数")
    
    # 更新Xr
    if Xr is not None:
        s.Xr = Xr
    if X is not None:
        s.Xr = compute_Xb(X, np.sum(s.mu * s.alpha, axis=0))
    
    # 重置KL和lbf
    s.KL = np.full(s.alpha.shape[0], np.nan)
    s.lbf = np.full(s.alpha.shape[0], np.nan)
    
    return s