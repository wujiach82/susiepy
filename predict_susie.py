import numpy as np
from typing import Optional, Union, Literal
import warnings

def coef_susie(object) -> np.ndarray:
    """
    Extract regression coefficients from susie fit

    Parameters:
        object: A susie fit.

    Returns:
        A p+1 vector, the first element being an intercept, and the
        remaining p elements being estimated regression coefficients.
    """
    s = object
    return np.concatenate([[s.intercept], np.sum(s.alpha * s.mu, axis=0) / s.X_column_scale_factors])

def predict_susie(
    object,
    newx: Optional[np.ndarray] = None,
    type: Literal["response", "coefficients"] = "response"
) -> np.ndarray:
    """
    Predict outcomes or extract coefficients from susie fit.

    Parameters:
        object: A susie fit.
        newx: A new value for X at which to do predictions.
        type: The type of output. For type="response",
            predicted or fitted outcomes are returned; for type="coefficients",
            the estimated coefficients are returned.

    Returns:
        For type="response", predicted or fitted outcomes are returned;
        for type="coefficients", the estimated coefficients are returned.
        If the susie fit has intercept=NA (which is common when using susie_suff_stat)
        then predictions are computed using an intercept of 0, and a warning is emitted.
    """
    s = object
    if type == "coefficients":
        if newx is not None:
            raise ValueError("Do not supply newx when predicting coefficients")
        return coef_susie(s)
    
    if newx is None:
        return s.fitted
        
    if np.isnan(s.intercept):
        warnings.warn("The prediction assumes intercept = 0")
        return newx @ coef_susie(s)[1:]
    else:
        return s.intercept + newx @ coef_susie(s)[1:]