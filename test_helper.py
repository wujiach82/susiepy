import numpy as np
from scipy import sparse
from susie_utils import compute_colSds
from base import ScaledMatrix

def set_X_attributes(X, center=True, scale=True):
    """为矩阵X设置三个属性
    
    参数:
        X: n x p 数据矩阵（可以是稠密或稀疏矩阵）
        center: 是否按列均值中心化
        scale: 是否按列标准差缩放
    
    返回:
        带有三个属性的ScaledMatrix对象，其中：
        - center: 如果center=True，是X的列均值；否则是0
        - scale: 如果scale=True，是X的列标准差；否则是1
        - d: 标准化后的X的每列平方和
    """
    # 检查是否为稀疏矩阵
    is_sparse = sparse.issparse(X)
    
    # 获取列均值
    if is_sparse:
        cm = np.array(X.mean(axis=0)).flatten()
    else:
        cm = np.mean(X, axis=0)
    
    # 获取列标准差
    csd = compute_colSds(X)
    
    # 标准差为0的列设为1
    csd[csd == 0] = 1
    
    if not center:
        cm = np.zeros_like(cm)
    if not scale:
        csd = np.ones_like(cm)
        
    # 计算d：标准化后的X的每列平方和
    n = X.shape[0]
    # 使用原始的colMeans和colSds计算
    d = n * np.array(X.mean(axis=0)).flatten()**2 + (n-1) * compute_colSds(X)**2
    # 使用设置的center和scale进行调整
    d = (d - n * cm**2) / csd**2
    
    # 创建ScaledMatrix对象
    return ScaledMatrix(
        data=X,
        center=cm,
        scale=csd,
        d=d
    )

def create_sparsity_mat(sparsity, n, p):
    """
    创建稀疏矩阵
    
    参数:
        sparsity: 稀疏度(0到1之间)
        n: 行数
        p: 列数
    """
    nonzero = round(n * p * (1 - sparsity))
    nonzero_idx = np.random.choice(n * p, nonzero, replace=False)
    mat = np.zeros(n * p)
    mat[nonzero_idx] = 1
    mat = mat.reshape((n, p))
    return mat

def simulate(n=100, p=200, is_sparse=False):
    """
    生成模拟数据
    
    参数:
        n: 样本数
        p: 特征数
        is_sparse: 是否生成稀疏矩阵
    """
    np.random.seed(1)
    beta = np.zeros(p)
    beta[:4] = 10
    
    if is_sparse:
        X = create_sparsity_mat(0.99, n, p)
        X_sparse = sparse.csr_matrix(X)
    else:
        X = np.random.normal(3, 4, size=(n, p))
        X_sparse = None
        
    y = X @ beta + np.random.normal(size=n)
    L = 10
    residual_variance = 0.8
    scaled_prior_variance = 0.2
    
    s = {
        'alpha': np.full((L, p), 1/p),
        'mu': np.full((L, p), 2),
        'mu2': np.full((L, p), 3),
        'Xr': np.full(n, 5),
        'KL': np.full(L, 1.2),
        'sigma2': residual_variance,
        'V': scaled_prior_variance * np.var(y),
        'lbf_variable': np.zeros((L, p))
    }
    
    return {
        'X': X,
        'X_sparse': X_sparse,
        's': s,
        'y': y,
        'n': n,
        'p': p,
        'b': beta
    }

def expect_equal_susie_update(new_res, original_res, tolerance=np.sqrt(np.finfo(float).eps)):
    """
    比较两个susie更新结果是否相等
    """
    assert np.allclose(new_res['alpha'], original_res['alpha'], rtol=tolerance)
    assert np.allclose(new_res['mu'], original_res['mu'], rtol=tolerance)
    assert np.allclose(new_res['mu2'], original_res['mu2'], rtol=tolerance)
    assert np.allclose(new_res['Xr'], original_res['Xr'], rtol=tolerance)
    assert np.allclose(new_res['KL'], original_res['KL'], rtol=tolerance)
    assert np.allclose(new_res['sigma2'], original_res['sigma2'], rtol=tolerance)
    assert np.allclose(new_res['V'], original_res['V'], rtol=tolerance)

def expect_equal_SER(new_res, original_res):
    """
    比较两个SER结果是否相等
    
    参数:
        new_res: 新的结果字典
        original_res: 原始结果字典
    """
    assert np.array_equal(new_res['alpha'], original_res['alpha']), "alpha不相等"
    assert np.array_equal(new_res['mu'], original_res['mu']), "mu不相等"
    assert np.array_equal(new_res['mu2'], original_res['mu2']), "mu2不相等"
    assert np.array_equal(new_res['lbf'], original_res['lbf']), "lbf不相等"
    assert np.array_equal(new_res['V'], original_res['V']), "V不相等"
    assert np.array_equal(new_res['loglik'], original_res['loglik']), "loglik不相等"

def expect_equal_susie(new_res, original_res, tolerance=np.sqrt(np.finfo(float).eps)):
    """
    比较两个susie结果是否在允许误差范围内相等
    
    参数:
        new_res: 新的结果字典
        original_res: 原始结果字典
        tolerance: 容差值
    """
    # 首先调用susie_update的比较
    expect_equal_susie_update(new_res, original_res, tolerance=tolerance)
    
    # 比较其他属性
    assert np.allclose(new_res["elbo"], original_res["elbo"], 
                      rtol=tolerance, atol=0), "elbo不相等"
    
    assert new_res["niter"] == original_res["niter"], "niter不相等"
    
    assert np.allclose(new_res["intercept"], original_res["intercept"], 
                      rtol=tolerance, atol=0), "intercept不相等"
    
    assert np.allclose(new_res["fitted"], original_res["fitted"], 
                      rtol=tolerance, atol=0), "fitted不相等"
    
    assert np.allclose(new_res["X_column_scale_factors"], 
                      original_res["X_column_scale_factors"], 
                      rtol=tolerance, atol=0), "X_column_scale_factors不相等"