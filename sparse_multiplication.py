import numpy as np

def compute_Xb(X, b):
    """计算标准化的X与b的矩阵乘积
    
    参数:
    X : ScaledMatrix对象, shape (n, p)
        未标准化的矩阵，带有center和scale属性
    b : ndarray, shape (p,)
        向量
        
    返回:
    ndarray, shape (n,)
        标准化后的矩阵乘积
    """
    cm = X.center
    csd = X.scale
    
    # 计算标准化的Xb
    scaled_Xb = X.data @ (b / csd)
    
    # 中心化
    Xb = scaled_Xb - np.sum(cm * b / csd)
    return Xb.flatten()

def compute_Xty(X, y):
    cm = X.center
    csd = X.scale
    
    # crossprod(y,X) in R is equivalent to t(y) %*% X
    y = np.asarray(y).reshape(-1, 1)
    ytX = (y.T @ X.data).flatten()  # 相当于R中的crossprod(y,X)
    
    # Scale Xty
    scaled_Xty = ytX / csd
    
    # Center Xty
    centered_scaled_Xty = scaled_Xty - (cm / csd) * np.sum(y)
    return centered_scaled_Xty

def compute_MXt(M, X):
    """计算M与标准化X转置的矩阵乘积
    
    参数:
    M : ndarray, shape (L, p)
        输入矩阵
    X : ScaledMatrix对象, shape (n, p)
        未标准化的矩阵，带有center和scale属性
        
    返回:
    ndarray, shape (L, n)
        标准化后的矩阵乘积
    """
    cm = X.center
    csd = X.scale
    
    # 计算 t(X %*% (t(M)/csd))
    # 注意：需要调整除法的维度
    scaled_Mt = M.T / csd[:, np.newaxis]  # 确保每列除以对应的csd
    result = (X.data @ scaled_Mt).T
    
    # 减去中心化项
    correction = M @ (cm/csd)
    return result - correction[:, np.newaxis]