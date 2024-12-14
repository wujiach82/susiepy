import numpy as np
import statsmodels.api as sm
from base import ScaledMatrix
def univariate_regression(X, y, Z=None, center=True, scale=False, return_residuals=False):
    """
    Perform univariate linear regression separately for each column of X
    
    Parameters:
        X: ndarray, shape (n, p) - matrix of regressors
        y: ndarray, shape (n,) - response variable
        Z: ndarray, shape (n, k), optional - matrix of covariates
        center: bool - whether to center variables
        scale: bool - whether to scale variables
        return_residuals: bool - whether to return residuals when Z is not None
        
    Returns:
        dict: containing regression coefficients (betahat) and standard errors (sebetahat)
    """
    # Convert inputs to numpy arrays if they aren't already
    X = np.asarray(X)
    y = np.asarray(y)
    
    # Handle missing values
    valid_idx = ~np.isnan(y)
    X = X[valid_idx]
    y = y[valid_idx]
    
    # Center and scale
    if center:
        y = y - np.mean(y)
        if scale:
            X = (X - np.mean(X, axis=0)) / np.std(X, axis=0, ddof=1)
        else:
            X = X - np.mean(X, axis=0)
    elif scale:
        X = X / np.std(X, axis=0, ddof=1)
    
    # Handle potential NaN values
    X = np.nan_to_num(X)
    
    # Handle covariates
    if Z is not None:
        Z = np.asarray(Z)
        if center:
            if scale:
                Z = (Z - np.mean(Z, axis=0)) / np.std(Z, axis=0, ddof=1)
            else:
                Z = Z - np.mean(Z, axis=0)
        # Remove effects of covariates
        Z = sm.add_constant(Z)
        model_z = sm.OLS(y, Z).fit()
        y = model_z.resid
    
    n, p = X.shape
    betahat = np.zeros(p)
    sebetahat = np.zeros(p)
    
    # Perform univariate regression for each column
    for i in range(p):
        X_i = sm.add_constant(X[:, i])
        try:
            model = sm.OLS(y, X_i).fit()
            betahat[i] = model.params[1]    # Skip intercept
            sebetahat[i] = model.bse[1]     # Standard error
        except:
            betahat[i] = 0
            sebetahat[i] = 0
    
    result = {'betahat': betahat, 'sebetahat': sebetahat}
    if return_residuals and Z is not None:
        result['residuals'] = y
    
    return result

def calc_z(X, Y, center=False, scale=False):
    """
    Compute z-scores (t-statistics) for association between Y and each column of X
    
    Parameters:
        X: ndarray - matrix of predictors
        Y: ndarray - response variable(s)
        center: bool - whether to center variables
        scale: bool - whether to scale variables
    
    Returns:
        ndarray - z-scores (betahat/sebetahat)
    """
    X = np.asarray(X)
    Y = np.asarray(Y)
    
    if Y.ndim == 1:
        result = univariate_regression(X, Y, center=center, scale=scale)
        return result['betahat'] / result['sebetahat']
    else:
        z_scores = np.zeros((X.shape[1], Y.shape[1]))
        for i in range(Y.shape[1]):
            result = univariate_regression(X, Y[:, i], center=center, scale=scale)
            z_scores[:, i] = result['betahat'] / result['sebetahat']
        return z_scores


# # 创建测试数据
# np.random.seed(42)
# raw_X = np.array([[1], [2], [3], [4], [5]])
# noise = np.random.normal(0, 0.5, 5)
# y = 2 * raw_X.flatten() + noise

# # 创建 ScaledMatrix 实例
# # 假设数据已经被中心化和缩放
# X_centered = raw_X - np.mean(raw_X)
# X_scaled = X_centered / np.std(X_centered, ddof=1)
# X = ScaledMatrix(
#     data=X_scaled,
#     center=np.mean(raw_X),
#     scale=np.std(X_centered, ddof=1)
# )

# # 测试1：使用已缩放的ScaledMatrix
# print("测试1: 使用已缩放的ScaledMatrix")
# result1 = univariate_regression(X, y, center=False, scale=False)  # 因为数据已经缩放过
# print("回归系数:", result1['betahat'][0])
# print("标准误差:", result1['sebetahat'][0])

# # 测试2：使用原始数据直接计算
# print("\n测试2: 使用原始数据")
# result2 = univariate_regression(raw_X, y, center=True, scale=True)
# print("回归系数:", result2['betahat'][0])
# print("标准误差:", result2['sebetahat'][0])

# # 使用statsmodels验证
# X_sm = sm.add_constant(raw_X)
# model = sm.OLS(y, X_sm).fit()
# print("\nstatsmodels结果:")
# print("回归系数:", model.params[1])
# print("标准误差:", model.bse[1])