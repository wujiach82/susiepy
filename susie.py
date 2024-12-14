import numpy as np
from scipy import sparse
import warnings
import pandas as pd
from base import SusieObject, ScaledMatrix
from initialize import init_setup, init_finalize
from update_each_effect import update_each_effect
from elbo import get_objective, estimate_residual_variance
from susie_utils import susie_slim, susie_prune_single_effects, susie_get_cs, susie_get_pip, compute_colstats
from univariate_regression import calc_z

def update_susie_obj(s: SusieObject, s_init: SusieObject) -> SusieObject:
    """用s_init的属性更新s的属性
    
    Args:
        s: 原始SusieObject实例
        s_init: 包含新值的SusieObject实例
    
    Returns:
        更新后的SusieObject实例
    """
    # 获取s_init的所有非私有属性
    for attr in vars(s_init):
        if not attr.startswith('__'):
            setattr(s, attr, getattr(s_init, attr))
    return s


def susie(X, y, L=None, scaled_prior_variance=0.2, residual_variance=None,
          prior_weights=None, null_weight=0, standardize=True, intercept=True,
          is_estimate_residual_variance=True, is_estimate_prior_variance=True,
          estimate_prior_method="EM", check_null_threshold=0,
          prior_tol=1e-9, residual_variance_upperbound=float('inf'),
          s_init=None, coverage=0.95, min_abs_corr=0.5,
          compute_univariate_zscore=False, na_rm=False,
          max_iter=100, tol=1e-3, verbose=False, track_fit=False,
          residual_variance_lowerbound=None, refine=False, n_purity=100):
    """
    Sum of Single Effects (SuSiE) Regression
    
    执行稀疏贝叶斯多元线性回归,使用"单效应之和"模型。
    """
    # 设置默认L值
    if L is None:
        L = min(10, X.shape[1])
        
    # 处理estimate_prior_method输入
    if estimate_prior_method not in ["optim", "EM", "simple"]:
        raise ValueError("estimate_prior_method must be one of: optim, EM, simple")
    
    # 检查输入X
    if not isinstance(X, (np.ndarray, sparse.csr_matrix)) or X.dtype != np.float64:
        raise ValueError("Input X must be a double-precision matrix or a sparse matrix")
        
    # 处理null_weight
    if isinstance(null_weight, (int, float)) and null_weight == 0:
        null_weight = None
    
    if null_weight is not None:
        if not isinstance(null_weight, (int, float)):
            raise ValueError("Null weight must be numeric")
        if null_weight < 0 or null_weight >= 1:
            raise ValueError("Null weight must be between 0 and 1")
            
        p = X.shape[1]
        if prior_weights is None:
            prior_weights = np.concatenate([
                np.repeat(1/p * (1 - null_weight), p),
                [null_weight]
            ])
        else:
            prior_weights = np.concatenate([
                prior_weights * (1-null_weight),
                [null_weight]
            ])
        X = np.hstack([X, np.zeros((X.shape[0], 1))])

    # 处理缺失值
    if np.any(np.isnan(X)):
        raise ValueError("Input X must not contain missing values")
    if np.any(np.isnan(y)):
        if na_rm:
            samples_kept = ~np.isnan(y)
            y = y[samples_kept]
            X = X[samples_kept]
        else:
            raise ValueError("Input y must not contain missing values")
            
    p = X.shape[1]
    n = X.shape[0]
    mean_y = np.mean(y)

    # 中心化输入
    if intercept:
        y = y - mean_y

    # 计算列统计量并设置X的属性
    out = compute_colstats(X, center=intercept, scale=standardize)
    X = ScaledMatrix(data=X, center=out['cm'], scale=out['csd'], d=out['d'])

    # 初始化susie拟合
    if residual_variance_lowerbound is None:
        residual_variance_lowerbound = np.var(y) / 1e4
        
    s = init_setup(n, p, L, scaled_prior_variance, residual_variance,
                  prior_weights, null_weight, float(np.var(y)), standardize)

    # 处理s_init
    if s_init is not None:
        if not hasattr(s_init, '__susie__'):
            raise ValueError("s_init should be a susie object")
        if np.max(s_init.alpha) > 1 or np.min(s_init.alpha) < 0:
            raise ValueError("s_init$alpha has invalid values outside range [0,1]")
            
        s_init = susie_prune_single_effects(s_init)
        num_effects = s_init.alpha.shape[0]
        
        if L is None:
            L = num_effects
        elif min(p, L) < num_effects:
            warnings.warn(f"Specified number of effects L = {min(p,L)} is smaller than the number of effects {num_effects}")
            L = num_effects
            
        s_init = susie_prune_single_effects(s_init, min(p, L), s.V)
        s = update_susie_obj(s, s_init)
        s = init_finalize(s, X=X)
    else:
        s = init_finalize(s)

    # 初始化ELBO
    elbo = np.full(max_iter + 1, np.nan)
    elbo[0] = float('-inf')
    tracking = []

    # 主循环
    for i in range(max_iter):
        if track_fit:
            tracking.append(susie_slim(s))
            
        s = update_each_effect(X, y, s, is_estimate_prior_variance,
                             estimate_prior_method, check_null_threshold)
        
        if verbose:
            print(f"objective: {get_objective(X,y,s)}")

        elbo[i+1] = get_objective(X, y, s)
        if (elbo[i+1] - elbo[i]) < tol:
            s.converged = True
            break

        if is_estimate_residual_variance:
            s.sigma2 = max(residual_variance_lowerbound,
                          estimate_residual_variance(X, y, s))
            if s.sigma2 > residual_variance_upperbound:
                s.sigma2 = residual_variance_upperbound
            if verbose:
                print(f"objective: {get_objective(X,y,s)}")

    # 清理和最终处理
    elbo = elbo[1:i+2]
    s.elbo = elbo
    s.niter = i

    if not hasattr(s, 'converged'):
        warnings.warn(f"IBSS algorithm did not converge in {max_iter} iterations!")
        s.converged = False

    if intercept:
        s.intercept = mean_y - np.sum(getattr(X, 'center') * 
                                     (np.sum(s.alpha * s.mu, axis=0)/getattr(X, 'scale')))
        s.fitted = s.Xr + mean_y
    else:
        s.intercept = 0
        s.fitted = s.Xr

    s.fitted = s.fitted.flatten()

    if track_fit:
        s.trace = tracking

    # SuSiE CS and PIP
    if coverage is not None and min_abs_corr is not None:
        s.sets = susie_get_cs(s, coverage=coverage, X=X,
                             min_abs_corr=min_abs_corr, n_purity=n_purity)
        s.pip = susie_get_pip(s, prune_by_cs=False, prior_tol=prior_tol)

    # 处理变量名
    if hasattr(X, 'columns'):
        variable_names = X.columns
        if null_weight is not None:
            variable_names[-1] = "null"
            s.pip.index = variable_names[:-1]
        else:
            s.pip.index = variable_names
            
        s.alpha = pd.DataFrame(s.alpha, columns=variable_names)
        s.mu = pd.DataFrame(s.mu, columns=variable_names)
        s.mu2 = pd.DataFrame(s.mu2, columns=variable_names)
        s.lbf_variable = pd.DataFrame(s.lbf_variable, columns=variable_names)

    # 计算单变量z分数
    if compute_univariate_zscore:
        if not isinstance(X, np.ndarray):
            warnings.warn("Calculation of univariate regression z-scores may be slow for non-matrix inputs")
        if null_weight is not None and null_weight != 0:
            X = X[:, :-1]
        s.z = calc_z(X, y, center=intercept, scale=standardize)

    s.X_column_scale_factors = getattr(X, 'scale')

    # Refinement部分
    if refine:
        if s_init is not None:
            warnings.warn("The given s_init is not used in refinement")
            
        if null_weight is not None and null_weight != 0:
            pw_s = s.pi[:-1] / (1 - null_weight)
            if not compute_univariate_zscore:
                X = X[:, :-1]
        else:
            pw_s = s.pi
            
        conti = True
        while conti and len(s.sets.cs) > 0:
            m = []
            for cs in range(len(s.sets.cs)):
                pw_cs = pw_s.copy()
                pw_cs[s.sets.cs[cs]] = 0
                
                if np.all(pw_cs == 0):
                    break
                    
                s2 = susie(X, y, L=L, scaled_prior_variance=scaled_prior_variance,
                          residual_variance=residual_variance,
                          prior_weights=pw_cs, s_init=None, null_weight=null_weight,
                          standardize=standardize, intercept=intercept,
                          is_estimate_residual_variance=is_estimate_residual_variance,
                          is_estimate_prior_variance=is_estimate_prior_variance,
                          estimate_prior_method=estimate_prior_method,
                          check_null_threshold=check_null_threshold,
                          prior_tol=prior_tol, coverage=coverage,
                          residual_variance_upperbound=residual_variance_upperbound,
                          min_abs_corr=min_abs_corr,
                          compute_univariate_zscore=compute_univariate_zscore,
                          na_rm=na_rm, max_iter=max_iter, tol=tol, verbose=False,
                          track_fit=False,
                          residual_variance_lowerbound=np.var(y.flatten())/1e4,
                          refine=False)
                          
                sinit2 = SusieObject()  # 假设我们有一个SusieObject类
                sinit2.alpha = s2.alpha
                sinit2.mu = s2.mu
                sinit2.mu2 = s2.mu2
                sinit2.__susie__ = True
                
                s3 = susie(X, y, L=L, scaled_prior_variance=scaled_prior_variance,
                          residual_variance=residual_variance,
                          prior_weights=pw_s, s_init=sinit2, null_weight=null_weight,
                          standardize=standardize, intercept=intercept,
                          is_estimate_residual_variance=is_estimate_residual_variance,
                          is_estimate_prior_variance=is_estimate_prior_variance,
                          estimate_prior_method=estimate_prior_method,
                          check_null_threshold=check_null_threshold,
                          prior_tol=prior_tol, coverage=coverage,
                          residual_variance_upperbound=residual_variance_upperbound,
                          min_abs_corr=min_abs_corr,
                          compute_univariate_zscore=compute_univariate_zscore,
                          na_rm=na_rm, max_iter=max_iter, tol=tol, verbose=False,
                          track_fit=False,
                          residual_variance_lowerbound=np.var(y.flatten())/1e4,
                          refine=False)
                m.append(s3)
                
            if len(m) == 0:
                conti = False
            else:
                elbo = np.array([get_objective(X,y,x) for x in m])
                if (np.max(elbo) - get_objective(X,y,s)) <= 0:
                    conti = False
                else:
                    s = m[np.argmax(elbo)]

    return s