import numpy as np
from scipy import stats
from scipy.optimize import root_scalar, minimize, minimize_scalar
from sparse_multiplication import compute_Xty


def single_effect_regression(y, X, V, residual_variance=1, prior_weights=None,
                           optimize_V="none", check_null_threshold=0):
    """Bayesian single-effect linear regression"""
    
    optimize_V = optimize_V if optimize_V in ["none", "optim", "uniroot", "EM", "simple"] else "none"
    
    Xty = compute_Xty(X, y)  # Using existing function
    betahat = (1/X.d) * Xty  
    shat2 = residual_variance / X.d
    
    if prior_weights is None:
        prior_weights = np.ones(X.shape[1]) / X.shape[1]
        
    if optimize_V not in ["EM", "none"]:
        V = optimize_prior_variance(optimize_V, betahat, shat2, prior_weights,
                                  alpha=None, post_mean2=None, V_init=V,
                                  check_null_threshold=check_null_threshold)
    
    # log(po) = log(BF * prior) for each SNP
    lbf = (stats.norm.logpdf(betahat, 0, np.sqrt(V + shat2)) - 
           stats.norm.logpdf(betahat, 0, np.sqrt(shat2)))
    lpo = lbf + np.log(prior_weights + np.finfo(float).eps)
    
    # Deal with special case of infinite shat2
    lbf[np.isinf(shat2)] = 0
    lpo[np.isinf(shat2)] = 0
    maxlpo = np.max(lpo)
    
    w_weighted = np.exp(lpo - maxlpo)
    weighted_sum_w = np.sum(w_weighted)
    alpha = w_weighted / weighted_sum_w
    
    post_var = (1/V + X.d/residual_variance)**(-1)
    post_mean = (1/residual_variance) * post_var * Xty
    post_mean2 = post_var + post_mean**2
    
    lbf_model = maxlpo + np.log(weighted_sum_w)
    loglik = lbf_model + np.sum(stats.norm.logpdf(y, 0, np.sqrt(residual_variance)))
    
    if optimize_V == "EM":
        V = optimize_prior_variance(optimize_V, betahat, shat2, prior_weights,
                                  alpha=alpha, post_mean2=post_mean2,
                                  check_null_threshold=check_null_threshold)
    
    return {
        'alpha': alpha,
        'mu': post_mean,
        'mu2': post_mean2,
        'lbf': lbf,
        'lbf_model': lbf_model,
        'V': V,
        'loglik': loglik
    }

def est_V_uniroot(betahat, shat2, prior_weights):
    result = root_scalar(negloglik_grad_logscale, 
                        args=(betahat, shat2, prior_weights),
                        bracket=[-10, 10])
    return np.exp(result.root)

def optimize_prior_variance(optimize_V, betahat, shat2, prior_weights,
                          alpha=None, post_mean2=None, V_init=None, 
                          check_null_threshold=0):
    V = V_init
    if optimize_V != "simple":
        if optimize_V == "optim":
            # 使用与R代码相同的初始值策略
            init_val = np.log(max(np.maximum(betahat**2 - shat2, 1)))
            
            # 使用与R代码相同的边界
            result = minimize_scalar(
                neg_loglik_logscale,
                args=(betahat, shat2, prior_weights),
                method='bounded',
                bounds=(-30, 15),
            )
            
            lV = result.x
            # 检查新估计是否比当前值更好
            if (neg_loglik_logscale(lV, betahat, shat2, prior_weights) >
                neg_loglik_logscale(np.log(V), betahat, shat2, prior_weights)):
                lV = np.log(V)
            V = np.exp(lV)
            
        elif optimize_V == "uniroot":
            # 使用root_scalar替代R中的uniroot
            result = root_scalar(
                negloglik_grad_logscale,
                args=(betahat, shat2, prior_weights),
                bracket=(-10, 10),
                method='brentq'
            )
            V = np.exp(result.root)
            
        elif optimize_V == "EM":
            V = np.sum(alpha * post_mean2)
        else:
            raise ValueError("Invalid option for optimize_V method")

    # # 检查是否应该将V设为0
    # if (loglik(0, betahat, shat2, prior_weights) + check_null_threshold >=
    #     loglik(V, betahat, shat2, prior_weights)):
    #     V = 0
        
    return V
def loglik(V, betahat, shat2, prior_weights):
    lbf = (stats.norm.logpdf(betahat, 0, np.sqrt(V + shat2)) - 
           stats.norm.logpdf(betahat, 0, np.sqrt(shat2)))
    lpo = lbf + np.log(prior_weights + np.finfo(float).eps)
    
    lbf[np.isinf(shat2)] = 0
    lpo[np.isinf(shat2)] = 0
    
    maxlpo = np.max(lpo)
    w_weighted = np.exp(lpo - maxlpo)
    weighted_sum_w = np.sum(w_weighted)
    return np.log(weighted_sum_w) + maxlpo

def neg_loglik_logscale(lV, betahat, shat2, prior_weights):
    return -loglik(np.exp(lV), betahat, shat2, prior_weights)

def loglik_grad(V, betahat, shat2, prior_weights):
    lbf = (stats.norm.logpdf(betahat, 0, np.sqrt(V + shat2)) - 
           stats.norm.logpdf(betahat, 0, np.sqrt(shat2)))
    lpo = lbf + np.log(prior_weights + np.finfo(float).eps)
    
    lbf[np.isinf(shat2)] = 0
    lpo[np.isinf(shat2)] = 0
    
    maxlpo = np.max(lpo)
    w_weighted = np.exp(lpo - maxlpo)
    weighted_sum_w = np.sum(w_weighted)
    alpha = w_weighted / weighted_sum_w
    return np.sum(alpha * lbf_grad(V, shat2, betahat**2/shat2))

def negloglik_grad_logscale(lV, betahat, shat2, prior_weights):
    return -np.exp(lV) * loglik_grad(np.exp(lV), betahat, shat2, prior_weights)

def lbf_grad(V, shat2, T2):
    l = 0.5 * (1/(V + shat2)) * ((shat2/(V + shat2))*T2 - 1)
    l[np.isnan(l)] = 0
    return l

def lbf(V, shat2, T2):
    l = 0.5*np.log(shat2/(V + shat2)) + 0.5*T2*(V/(V + shat2))
    l[np.isnan(l)] = 0
    return l