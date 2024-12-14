import numpy as np
import warnings
from scipy import stats, optimize, linalg, sparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, List, Dict, Optional, Tuple, Any
from base import SusieObject

def set_R_attributes(R: np.ndarray, r_tol: float) -> np.ndarray:
    """设置R矩阵的属性
    
    Args:
        R: np.ndarray
            p x p的LD矩阵
        r_tol: float
            特征值检查的容差水平
            
    Returns:
        res: np.ndarray
            带有附加属性的R矩阵
    """
    # 计算特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eigh(R)
    
    # 将小于容差的特征值设为0
    eigenvalues[np.abs(eigenvalues) < r_tol] = 0
    
    # 检查负特征值
    if np.any(eigenvalues < 0):
        min_lambda = np.min(eigenvalues)
        eigenvalues[eigenvalues < 0] = 0
        warnings.warn(
            f"输入的相关矩阵存在负特征值(最小值为{min_lambda})。"
            "已将这些负特征值调整为0。如果您认为负特征值是由数值舍入误差导致的，"
            "可以忽略此消息。"
        )
    
    # 重构矩阵
    res = eigenvectors @ (eigenvectors.T * eigenvalues)
    
    # 设置属性
    res = res.view(np.ndarray)  # 转换为普通ndarray以添加属性
    res.eigen = {'values': eigenvalues, 'vectors': eigenvectors}
    res.d = np.diag(res)
    res.scaled_scale = np.ones(R.shape[0])
    
    return res

def remove_null_effects(s: SusieObject) -> SusieObject:
    """移除V等于0的效应
    
    Args:
        s: SusieObject
            SuSiE对象实例
            
    Returns:
        s: SusieObject
            移除空效应后的SuSiE对象
    """
    # 创建非零V值的布尔掩码
    null_indices = (s.V == 0)
    
    # 使用布尔索引更新所有数组
    s.alpha = s.alpha[~null_indices]
    s.mu = s.mu[~null_indices]
    s.mu2 = s.mu2[~null_indices]
    s.lbf_variable = s.lbf_variable[~null_indices]
    s.V = s.V[~null_indices]
    
    return s

def add_null_effect(s: SusieObject, V: float) -> SusieObject:
    """向模型添加空效应
    
    Args:
        s: SusieObject
            SuSiE对象实例
        V: float
            要添加到V数组的值
            
    Returns:
        s: SusieObject
            添加空效应后的SuSiE对象
    """
    # 从alpha获取列数
    p = s.alpha.shape[1]
    
    # 为每个数组添加新行
    s.alpha = np.vstack([s.alpha, np.full(p, 1/p)])
    s.mu = np.vstack([s.mu, np.zeros(p)])
    s.mu2 = np.vstack([s.mu2, np.zeros(p)])
    s.lbf_variable = np.vstack([s.lbf_variable, np.zeros(p)])
    s.V = np.append(s.V, V)
    
    return s

def susie_get_objective(res: SusieObject, 
                       last_only: bool = True, 
                       warning_tol: float = 1e-6) -> Union[float, np.ndarray]:
    """获取目标函数值(ELBO)
    
    Args:
        res: SusieObject实例
        last_only: 如果为True,只返回最后一次迭代的ELBO
        warning_tol: ELBO下降的警告阈值
        
    Returns:
        ELBO值或ELBO值数组
    """
    if not hasattr(res, 'elbo') or res.elbo is None:
        return None
        
    if not all(np.diff(res.elbo) >= -warning_tol):
        warning_message("目标函数在下降", "warning")
        
    if last_only:
        return res.elbo[-1]
    return res.elbo

def susie_get_posterior_mean(res: SusieObject, 
                           prior_tol: float = 1e-9) -> np.ndarray:
    """获取回归系数的后验均值"""
    # 过滤掉先验方差为0的单效应
    if isinstance(res.V, (np.ndarray, float)):
        include_idx = np.where(res.V > prior_tol)[0]
    else:
        include_idx = np.arange(res.alpha.shape[0])
        
    if len(include_idx) > 0:
        return np.sum((res.alpha * res.mu)[include_idx], axis=0) / res.X_column_scale_factors
    else:
        return np.zeros(res.mu.shape[1])

def susie_get_posterior_sd(res: SusieObject, 
                         prior_tol: float = 1e-9) -> np.ndarray:
    """获取回归系数的后验标准差"""
    if isinstance(res.V, (np.ndarray, float)):
        include_idx = np.where(res.V > prior_tol)[0]
    else:
        include_idx = np.arange(res.alpha.shape[0])
        
    if len(include_idx) > 0:
        var = np.sum((res.alpha * res.mu2 - 
                     (res.alpha * res.mu)**2)[include_idx], axis=0)
        return np.sqrt(var) / res.X_column_scale_factors
    else:
        return np.zeros(res.mu.shape[1])

def susie_get_niter(res: SusieObject) -> int:
    """获取迭代次数"""
    return res.niter

def susie_get_prior_variance(res: SusieObject) -> Union[float, np.ndarray]:
    """获取先验方差"""
    return res.V

def susie_get_residual_variance(res: SusieObject) -> float:
    """获取残差方差"""
    return res.sigma2

def susie_get_lfsr(res: SusieObject) -> np.ndarray:
    """获取局部虚假信号率(local false sign rate)"""
    pos_prob = stats.norm.cdf(0, loc=res.mu.T, 
                            scale=np.sqrt(res.mu2 - res.mu**2))
    neg_prob = 1 - pos_prob
    return 1 - np.sum(res.alpha * np.maximum(pos_prob.T, neg_prob.T), axis=0)

def susie_get_posterior_samples(res: SusieObject, 
                              num_samples: int) -> Dict[str, np.ndarray]:
    """从后验分布中抽取样本
    
    Args:
        res: SusieObject实例
        num_samples: 需要的样本数量
        
    Returns:
        包含效应大小样本和因果状态样本的字典
    """
    # 移除先验方差为0的效应
    if isinstance(res.V, (np.ndarray, float)):
        include_idx = np.where(res.V > 1e-9)[0]
    else:
        include_idx = np.arange(res.alpha.shape[0])
        
    posterior_mean = res.mu / res.X_column_scale_factors[:, None]
    posterior_sd = np.sqrt(res.mu2 - res.mu**2) / res.X_column_scale_factors[:, None]
    
    pip = res.alpha
    num_snps = pip.shape[1]
    b_samples = np.zeros((num_snps, num_samples))
    gamma_samples = np.zeros((num_snps, num_samples))
    
    for sample_i in range(num_samples):
        b = np.zeros(num_snps)
        if len(include_idx) > 0:
            for l in include_idx:
                gamma_l = np.random.multinomial(1, pip[l])
                nonzero_idx = np.where(gamma_l != 0)[0][0]
                effect_size = np.random.normal(
                    posterior_mean[l, nonzero_idx],
                    posterior_sd[l, nonzero_idx]
                )
                b_l = gamma_l * effect_size
                b += b_l
                
        b_samples[:, sample_i] = b
        gamma_samples[:, sample_i] = (b != 0).astype(float)
        
    return {'b': b_samples, 'gamma': gamma_samples}


def susie_get_cs(res: SusieObject,
                 X: Optional[np.ndarray] = None,
                 Xcorr: Optional[np.ndarray] = None,
                 coverage: float = 0.95,
                 min_abs_corr: float = 0.5,
                 dedup: bool = True,
                 squared: bool = False,
                 check_symmetric: bool = True,
                 n_purity: int = 100) -> Dict[str, Any]:
    """获取可信集(CS)
    
    Args:
        res: SusieObject实例
        X: n x p的原始数据矩阵
        Xcorr: p x p的相关系数矩阵
        coverage: CS的覆盖率
        min_abs_corr: CS纯度的最小阈值
        dedup: 是否去除重复的CS
        squared: 是否报告平方相关系数
        check_symmetric: 是否检查Xcorr的对称性
        n_purity: 计算相关性时使用的最大变量数
        use_rfast: 是否使用快速计算(Python版本不需要)
        
    Returns:
        包含CS信息的字典
    """
    # 检查输入
    if X is not None and Xcorr is not None:
        raise ValueError("只能指定X或Xcorr之一")
        
    # 检查并强制Xcorr对称
    if check_symmetric and Xcorr is not None:
        if not is_symmetric_matrix(Xcorr):
            warning_message("Xcorr不对称；强制将Xcorr设为对称矩阵 "
                          "(Xcorr + Xcorr.T)/2")
            Xcorr = (Xcorr + Xcorr.T) / 2
            
    # 初始化
    null_index = getattr(res, 'null_index', 0)
    include_idx = np.ones(res.alpha.shape[0], dtype=bool)
    
    # 根据V值更新include_idx
    if isinstance(res.V, (np.ndarray, float)):
        include_idx = res.V > 1e-9
        
    # 获取CS位置
    status = in_CS(res.alpha, coverage)
    
    # 获取每个CS的位置列表和覆盖率
    cs = [np.where(row != 0)[0] for row in status]

    claimed_coverage = [np.sum(res.alpha[i][cs[i]]) for i in range(len(cs))]
    
    # 更新include_idx，排除空CS
    include_idx = include_idx & np.array([len(x) > 0 for x in cs])
    
    # 去重
    if dedup:
        cs_tuples = [tuple(sorted(x)) for x in cs]
        include_idx = include_idx & (~pd.Series(cs_tuples).duplicated().values)
        
    # 如果没有有效的CS，返回空结果
    if not np.any(include_idx):
        return {
            'cs': None,
            'coverage': None,
            'purity': None,
            'cs_index': None,
            'requested_coverage': coverage
        }
        
    # 过滤CS和claimed_coverage
    cs = [cs[i] for i in np.where(include_idx)[0]]
    claimed_coverage = np.array(claimed_coverage)[np.where(include_idx)[0]]
    
    # 如果没有X和Xcorr，直接返回结果
    if X is None and Xcorr is None:
        cs_dict = dict(zip([f"L{i}" for i in np.where(include_idx)[0]], cs))
        return {
            'cs': cs_dict,
            'coverage': claimed_coverage,
            'requested_coverage': coverage
        }
        
    # 计算纯度
    purity = []
    for i, cs_i in enumerate(cs):
        if null_index > 0 and null_index in cs_i:
            purity.append([-9, -9, -9])
        else:
            purity.append(get_purity(cs_i, X, Xcorr, squared, n_purity))
    purity = np.array(purity)
    
    # 创建purity DataFrame
    purity_df = pd.DataFrame(
        purity,
        columns=['min.sq.corr', 'mean.sq.corr', 'median.sq.corr'] if squared 
        else ['min.abs.corr', 'mean.abs.corr', 'median.abs.corr']
    )
    
    # 根据纯度过滤
    threshold = min_abs_corr**2 if squared else min_abs_corr
    is_pure = np.where(purity[:, 0] >= threshold)[0]
    
    if len(is_pure) > 0:
        # 过滤并重新排序
        cs_filtered = [cs[i] for i in is_pure]
        purity_filtered = purity_df.iloc[is_pure]
        row_names = [f"L{i}" for i in np.where(include_idx)[0][is_pure]]
        
        # 根据纯度排序
        ordering = np.argsort(-purity_filtered.iloc[:, 0])
        
        # 创建最终的cs字典
        cs_dict = dict(zip(
            [row_names[i] for i in ordering],
            [cs_filtered[i] for i in ordering]
        ))
        
        return {
            'cs': cs_dict,
            'purity': purity_filtered.iloc[ordering],
            'cs_index': np.where(include_idx)[0][is_pure][ordering],
            'coverage': claimed_coverage[is_pure][ordering],
            'requested_coverage': coverage
        }
    else:
        return {
            'cs': None,
            'coverage': None,
            'requested_coverage': coverage
        }

def get_cs_correlation(model: SusieObject,
                      X: Optional[np.ndarray] = None,
                      Xcorr: Optional[np.ndarray] = None,
                      max_only: bool = False) -> np.ndarray:
    """计算CS间相关性
    
    Args:
        model: SusieObject实例
        X: 原始数据矩阵
        Xcorr: 相关系数矩阵
        max_only: 是否只返回最大相关系数
        
    Returns:
        相关系数矩阵或最大相关系数
    """
    if X is not None and Xcorr is not None:
        raise ValueError("只能指定X或Xcorr之一")
        
    # 获取每个CS中PIP最大的变量索引
    if not hasattr(model, 'sets') or not hasattr(model.sets, 'cs'):
        raise ValueError("模型中缺少CS信息")
    
    max_pip_idx = [cs[np.argmax(model.pip[cs])] for cs in model.sets.cs]
    
    # 计算相关性
    if Xcorr is None:
        X_sub = X[:, max_pip_idx]
        cs_corr = np.corrcoef(X_sub.T)
    else:
        cs_corr = Xcorr[np.ix_(max_pip_idx, max_pip_idx)]
        
    if max_only:
        return np.max(np.abs(cs_corr[np.triu_indices_from(cs_corr, k=1)]))
    return cs_corr

def susie_get_pip(res: SusieObject, 
                  prune_by_cs: bool = False,
                  prior_tol: float = 1e-9) -> np.ndarray:
    """计算后验包含概率(PIP)
    
    Args:
        res: SusieObject实例
        prune_by_cs: 是否只考虑CS中的效应计算PIP
        prior_tol: 先验方差的容差水平
        
    Returns:
        每个变量的PIP值数组
    """
    if not isinstance(res, SusieObject):
        raise TypeError("Input must be a SusieObject instance")
        
    alpha = res.alpha
    
    # 移除null weight列
    if hasattr(res, 'null_index') and res.null_index is not None and res.null_index > 0:
        alpha = np.delete(alpha, res.null_index-1, axis=1)
        
    # 过滤先验为0的效应
    if hasattr(res, 'V') and isinstance(res.V, (np.ndarray, float, int)):
        include_idx = np.where(res.V > prior_tol)[0]
    else:
        include_idx = np.arange(alpha.shape[0])
        
    # 只考虑CS中的效应
    if hasattr(res, 'sets') and res.sets.get('cs_index') is not None and prune_by_cs:
        include_idx = np.intersect1d(include_idx, res.sets['cs_index'])
    elif prune_by_cs:
        include_idx = np.array([])
        
    # 提取相关行
    if len(include_idx) > 0:
        alpha = alpha[include_idx]
    else:
        alpha = np.zeros((1, alpha.shape[1]))
    
    return 1 - np.prod(1 - alpha, axis=0)


def n_in_CS_x(x: np.ndarray, coverage: float = 0.9) -> int:
    """计算CS中的变量数量
    
    Args:
        x: 概率向量
        coverage: 覆盖率阈值
        
    Returns:
        CS中的变量数量，如果概率和小于coverage则返回0
    """
    # 如果概率和小于coverage，返回0
    if np.sum(x) <= coverage:
        return 0
    return np.sum(np.cumsum(np.sort(x)[::-1]) < coverage) + 1

def in_CS_x(x: np.ndarray, coverage: float = 0.9) -> np.ndarray:
    """返回指示每个点是否在CS中的二进制向量
    
    Args:
        x: 概率向量
        coverage: 覆盖率阈值
        
    Returns:
        二进制向量，如果概率和小于coverage则返回全0向量
    """
    # 如果概率和小于coverage，返回全0向量
    if np.sum(x) <= coverage:
        return np.zeros_like(x)
        
    n = n_in_CS_x(x, coverage)
    order = np.argsort(x)[::-1]
    result = np.zeros_like(x)
    result[order[:n]] = 1
    return result

def in_CS(res: Union[SusieObject, np.ndarray], coverage: float = 0.9) -> np.ndarray:
    """返回一个l×p二进制矩阵,指示哪些变量在susie credible sets中
    
    Args:
        res: SusieObject实例或alpha矩阵
        coverage: 覆盖率阈值,默认0.9
        
    Returns:
        l×p二进制矩阵
    """
    if isinstance(res, SusieObject):
        alpha = res.alpha
    else:
        alpha = res
    return np.apply_along_axis(lambda x: in_CS_x(x, coverage), 1, alpha)

def n_in_CS(res: Union[SusieObject, np.ndarray], coverage: float = 0.9) -> np.ndarray:
    """返回每个CS中的变量数量
    
    Args:
        res: SusieObject实例或alpha矩阵
        coverage: 覆盖率阈值,默认0.9
        
    Returns:
        每个CS中的变量数量向量
    """
    if isinstance(res, SusieObject):
        alpha = res.alpha
    else:
        alpha = res
    return np.apply_along_axis(lambda x: n_in_CS_x(x, coverage), 1, alpha)

def get_purity(pos: List[int], X: Optional[np.ndarray] = None, 
               Xcorr: Optional[np.ndarray] = None, squared: bool = False, 
               n: int = 100, use_rfast: Optional[bool] = None) -> np.ndarray:
    """计算相关性的最小值、平均值和中位数"""
    
    # 如果只有一个位置，返回全1
    if len(pos) == 1:
        return np.array([1, 1, 1])
    
    # 如果需要，进行子采样
    if len(pos) > n:
        pos = np.random.choice(pos, n, replace=False)
    
    if Xcorr is None:
        # 获取子矩阵
        X_sub = X[:, pos]
        # 计算相关系数矩阵
        corr_matrix = np.corrcoef(X_sub.T)
        # 获取上三角矩阵的值（不包括对角线）
        values = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
    else:
        # 从相关系数矩阵中获取子矩阵
        corr_sub = Xcorr[np.ix_(pos, pos)]
        values = corr_sub[np.triu_indices_from(corr_sub, k=1)]
    
    # 取绝对值
    values = np.abs(values)
    
    if squared:
        values = values ** 2
    
    # 返回最小值、平均值和中位数
    return np.array([
        np.min(values),
        np.sum(values) / len(values),  # 使用sum/length而不是mean
        np.median(values)
    ])

# 计算相关系数矩阵,忽略标准差为零的警告
def muffled_corr(x):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='.*standard deviation is zero.*')
        return np.corrcoef(x, rowvar=False)

# 将协方差矩阵转换为相关系数矩阵,忽略零/NA条目的警告
def muffled_cov2cor(x):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='.*invalid value encountered.*')
        d = np.sqrt(np.diag(x))
        cor = x / np.outer(d, d)
        cor[d == 0] = 0
        return cor

# 检查矩阵是否对称
def is_symmetric_matrix(x):
    return np.allclose(x, x.T)

# 计算回归系数的标准误差
def calc_stderr(X, residuals):
    n = X.shape[0]
    sigma2 = np.sum(residuals**2) / (n - 2)
    return np.sqrt(sigma2 * np.diag(linalg.inv(X.T @ X)))

# 移除susie模型的线性效应后的残差
def get_R(X, Y, s):
    return Y - X @ s.coef_

# 简化susie拟合结果为基本属性
def susie_slim(res):
    return {
        'alpha': res.alpha,
        'niter': res.niter,
        'V': res.V,
        'sigma2': res.sigma2
    }

# 修剪susie模型中的单个效应到指定数量L
def susie_prune_single_effects(s, L=0, V=None):
    num_effects = s.alpha.shape[0]
    if L == 0:
        # 基于s.V中非零元素的过滤
        if s.V is not None:
            L = np.sum(s.V > 0)
        else:
            L = num_effects
    if L == num_effects:
        s.sets = None
        return s
        
    if hasattr(s, 'sets') and hasattr(s.sets, 'cs_index'):
        effects_rank = np.concatenate([
            s.sets.cs_index,
            np.setdiff1d(np.arange(num_effects), s.sets.cs_index)
        ])
    else:
        effects_rank = np.arange(num_effects)
        
    if L > num_effects:
        print(f"指定的效应数L={L}大于输入SuSiE模型中的效应数{num_effects}。SuSiE模型将扩展为{L}个效应。")
        s.alpha = np.vstack([
            s.alpha[effects_rank],
            np.full((L - num_effects, s.alpha.shape[1]), 1/s.alpha.shape[1])
        ])
        
        for n in ['mu', 'mu2', 'lbf_variable']:
            if hasattr(s, n):
                setattr(s, n, np.vstack([
                    getattr(s, n)[effects_rank],
                    np.zeros((L - num_effects, getattr(s, n).shape[1]))
                ]))
                
        for n in ['KL', 'lbf']:
            if hasattr(s, n):
                setattr(s, n, np.concatenate([
                    getattr(s, n)[effects_rank],
                    np.full(L - num_effects, np.nan)
                ]))
                
        if V is not None:
            if isinstance(V, np.ndarray) and len(V) > 1:
                # 如果V是数组且长度大于1
                new_V = np.zeros(L)
                new_V[:num_effects] = s.V[effects_rank]
                V = new_V
            else:
                # 如果V是单个数字
                V = np.repeat(V, L)
        s.V = V
        
    s.sets = None
    return s

def compute_colstats(X: Union[np.ndarray, sparse.spmatrix], 
                    center: bool = True, 
                    scale: bool = True) -> Dict[str, np.ndarray]:
    """
    计算X的列均值、列标准差和Y^2的行和，其中Y是X的中心化和/或标准化版本
    
    Args:
        X: n x p的矩阵(密集或稀疏)
        center: 是否中心化
        scale: 是否标准化
        
    Returns:
        包含以下键的字典:
        - cm: 列均值
        - csd: 列标准差
        - d: Y^2的行和
    """
    n, p = X.shape
    
    # 计算列均值
    if center:
        if sparse.issparse(X):
            cm = np.array(X.mean(axis=0)).flatten()
        else:
            cm = np.mean(X, axis=0)
    else:
        cm = np.zeros(p)
        
    # 计算列标准差
    if scale:
        csd = compute_colSds(X)
        csd[csd == 0] = 1  # 当列方差为0时设置为1
    else:
        csd = np.ones(p)
        
    # 计算d，避免创建中间矩阵
    #   Y = (X - cm)/csd
    #   d = np.sum(Y**2, axis=0)
    if sparse.issparse(X):
        d = n * np.square(np.array(X.mean(axis=0)).flatten()) + \
            (n-1) * np.square(compute_colSds(X))
    else:
        d = n * np.square(np.mean(X, axis=0)) + \
            (n-1) * np.square(compute_colSds(X))
    d = (d - n * np.square(cm)) / np.square(csd)
    
    return {'cm': cm, 'csd': csd, 'd': d}

def compute_colSds(X: Union[np.ndarray, sparse.spmatrix]) -> np.ndarray:
    """
    计算任意类型矩阵的列标准差
    
    Args:
        X: n x p的矩阵(密集或稀疏)
        
    Returns:
        长度为p的列标准差数组
    """
    n = X.shape[0]
    
    if sparse.issparse(X):
        # 完全按照R代码的逻辑：
        # n = nrow(X)
        # Y = apply_nonzeros(X,function (u) u^2)
        # d = colMeans(Y) - colMeans(X)^2
        # return(sqrt(d*n/(n-1)))
        
        Y = X.copy()
        Y.data = Y.data**2  # 等同于 apply_nonzeros(X,function (u) u^2)
        
        # 计算colMeans
        col_means_Y = np.array(Y.mean(axis=0)).flatten()  # colMeans(Y)
        col_means_X = np.array(X.mean(axis=0)).flatten()  # colMeans(X)
        
        # 完全按照R代码的计算顺序
        d = col_means_Y - np.square(col_means_X)  # d = colMeans(Y) - colMeans(X)^2
        return np.sqrt(d * n/(n-1))  # sqrt(d*n/(n-1))
    else:
        return np.std(X, axis=0, ddof=1)

def check_semi_pd(A: np.ndarray, tol: float) -> Dict[str, Any]:
    """
    检查矩阵A是否为半正定矩阵
    
    Args:
        A: p x p的对称矩阵
        tol: 容差
        
    Returns:
        包含以下键的字典:
        - matrix: 带有特征分解的矩阵
        - status: 是否为半正定
        - eigenvalues: 截断后的特征值
    """
    # 计算特征值和特征向量
    eigenvals, eigenvecs = np.linalg.eigh(A)
    
    # 将小于容差的特征值设为0
    eigenvals[np.abs(eigenvals) < tol] = 0
    
    # 将特征分解信息添加到矩阵属性中
    A = A.view(np.ndarray)  # 转换为普通ndarray以添加属性
    A.eigen = {'values': eigenvals, 'vectors': eigenvecs}
    
    return {
        'matrix': A,
        'status': not np.any(eigenvals < 0),
        'eigenvalues': eigenvals
    }

def check_projection(A: np.ndarray, b: np.ndarray) -> Dict:
    """
    检查b是否在A的非零特征向量张成的空间中
    
    参数:
        A: p x p矩阵
        b: 长度为p的向量
        
    返回:
        包含状态和消息的字典
    """
    eigenvals, eigenvecs = np.linalg.eigh(A)
    B = eigenvecs[:, eigenvals > np.finfo(float).eps]
    
    projected_b = B @ (B.T @ b)
    is_equal = np.allclose(projected_b, b)
    
    if is_equal:
        return {'status': True, 'msg': None}
    else:
        return {'status': False, 'msg': f'Difference: {np.max(np.abs(projected_b - b))}'}

def warning_message(message: str, style: str = 'warning') -> None:
    """
    显示带样式的警��消息
    
    参数:
        message: 警告消息
        style: 'warning' 或 'hint'
    """
    if style == 'warning':
        print('\033[1;4;31mWARNING:\033[0m', message)
    else:
        print('\033[1;4;35mHINT:\033[0m', message)