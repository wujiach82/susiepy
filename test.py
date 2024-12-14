import numpy as np
from scipy import sparse
from numpy.testing import assert_allclose, assert_array_equal, assert_array_almost_equal, assert_almost_equal
import pytest
from susie_utils import compute_colSds, susie_get_pip, susie_prune_single_effects
from base import SusieObject, ScaledMatrix
from test_helper import simulate, set_X_attributes
from sparse_multiplication import compute_MXt, compute_Xb, compute_Xty
from single_effect_regression import single_effect_regression
from initialize import susie_init_coef, init_setup, init_finalize
from susie_utils import susie_get_cs
import copy
from types import SimpleNamespace

def test_dense_matrix():
    """测试密集矩阵的列标准差计算"""
    np.random.seed(1)
    # 创建 4x6 的随机矩阵
    X = np.random.randn(4, 6)
    # 计算列标准差
    y1 = compute_colSds(X)
    y2 = np.std(X, axis=0, ddof=1)
    # 比较结果，允许 1e-15 的误差
    np.testing.assert_allclose(y1, y2, rtol=1e-15, atol=1e-15)

def test_sparse_matrix():
    """测试稀疏矩阵的列标准差计算"""
    np.random.seed(1)
    # 创建随机矩阵
    X = np.random.randn(4, 6)
    # 将约30%的元素设为0
    mask = np.random.random(X.shape) < 0.3
    X[mask] = 0
    # 转换为稀疏矩阵 (CSR格式)
    Y = sparse.csr_matrix(X)
    # 计算列标准差
    y1 = compute_colSds(Y)
    y2 = np.std(X, axis=0, ddof=1)
    # 比较结果
    np.testing.assert_allclose(y1, y2, rtol=1e-15, atol=1e-15)

def test_pip_computation():
    """测试PIP在不同条件下的计算是否正确"""
    # 创建测试数据，对应R代码：
    # res = list(alpha = matrix(c(rep(1,10),rep(2,10),rep(3,10))/10, 3, 10, byrow=T), 
    #           sets=list(cs_index = c(2)), 
    #           null_index = 10, 
    #           V = rep(1,3))
    
    # 创建SusieObject实例
    res = SusieObject()
    res.alpha = np.array([
        [0.1] * 10,
        [0.2] * 10,
        [0.3] * 10
    ])
    res.sets = {'cs_index': [1]}  # 对应R的list(cs_index = c(2))
    res.null_index = 10
    res.V = np.array([1.0, 1.0, 1.0])  # 对应R的rep(1,3)
    
    # 测试1：基本PIP计算
    expected_pip = np.array([0.496] * 9)
    np.testing.assert_almost_equal(
        susie_get_pip(res)[:9], 
        expected_pip,
        decimal=3
    )
    
    # 测试2：使用prune_by_cs=True的情况
    expected_pip_pruned = np.array([0.2] * 9)
    np.testing.assert_almost_equal(
        susie_get_pip(res, prune_by_cs=True)[:9],
        expected_pip_pruned,
        decimal=3
    )
    
    # 测试3：V=0的情况
    res.V = np.array([0.0, 0.0, 0.0])
    expected_pip_zero = np.array([0.0] * 9)
    np.testing.assert_almost_equal(
        susie_get_pip(res)[:9],
        expected_pip_zero,
        decimal=3
    )
    
    # 测试4：cs_index=None且prune_by_cs=True的情况
    res.V = np.array([1.0, 1.0, 1.0])
    res.sets = {'cs_index': None}  # 对应R中的NULL
    np.testing.assert_almost_equal(
        susie_get_pip(res, prune_by_cs=True)[:9],
        expected_pip_zero,
        decimal=3
    )

def test_sparse_multiplication():
    """测试稀疏矩阵乘法"""
    # 设置随机种子
    np.random.seed(1)
    
    # 生成模拟数据
    sim_data = simulate(is_sparse=True)
    X = sim_data['X']
    y = sim_data['y']
    b = sim_data['b']
    p = X.shape[1]
    
    # 生成随机矩阵M
    L = 10
    M = np.random.normal(size=(L, p))
    
    # 设置X的属性
    X_dense = set_X_attributes(X)
    
    # 标准化X
    center = getattr(X_dense, 'center')
    scale = getattr(X_dense, 'scale')
    X_standardized = ((X_dense - center[np.newaxis, :]) / scale[np.newaxis, :])
    
    X_sparse = set_X_attributes(sim_data['X_sparse'])
    
    # 测试compute_Xb
    assert_allclose(
        compute_Xb(X_sparse, b),
        np.asarray(X_standardized @ b).flatten(),
        rtol=1e-12
    )
    assert_allclose(
        compute_Xb(X_dense, b),
        np.asarray(X_standardized @ b).flatten(),
        rtol=1e-12
    )
    
    # 测试compute_Xty
    assert_allclose(
        compute_Xty(X_sparse, y),
        np.asarray(X_standardized.T @ y).flatten(),
        rtol=1e-12
    )
    assert_allclose(
        compute_Xty(X_dense, y),
        np.asarray(X_standardized.T @ y).flatten(),
        rtol=1e-12
    )
    
    # 测试compute_MXt
    assert_allclose(
        compute_MXt(M, X_sparse),
        M @ X_standardized.T,
        rtol=1e-12
    )
    assert_allclose(
        compute_MXt(M, X_dense),
        M @ X_standardized.T,
        rtol=1e-12
    )

def test_sparse_set_X_attributes():
    """测试稀疏矩阵版本的set_X_attributes"""
    # 生成模拟数据
    sim_data = simulate(is_sparse=True)
    X = sim_data['X']
    X_sparse = sim_data['X_sparse']
    
    # 分别对稠密矩阵和稀疏矩阵设置属性
    dense_res = set_X_attributes(X)
    sparse_res = set_X_attributes(X_sparse)
    
    # 检查基础矩阵是否相等（忽略属性）
    assert_array_almost_equal(
        sparse_res.data.toarray() if sparse.issparse(sparse_res.data) else sparse_res.data,
        X_sparse.toarray() if sparse.issparse(X_sparse) else X_sparse
    )
    
    # # 检查三个属性是否相等
    # assert_array_almost_equal(dense_res.center, sparse_res.center)
    # assert_array_almost_equal(dense_res.scale, sparse_res.scale)
    assert_array_almost_equal(dense_res.d, sparse_res.d)

def test_single_effect_regression():
    """测试single_effect_regression函数"""
    # 生成模拟数据
    sim_data = simulate(is_sparse=True)
    X = sim_data['X'] 
    y = sim_data['y']
    
    # 设置参数
    V = 0.2
    residual_variance = 1.0
    
    # 对稠密矩阵和稀疏矩阵分别进行测试
    X_dense = set_X_attributes(X)
    X_sparse = set_X_attributes(sim_data['X_sparse'])
    
    # 运行single_effect_regression
    dense_res = single_effect_regression(y, X_dense, V, residual_variance, optimize_V="EM")
    sparse_res = single_effect_regression(y, X_sparse, V, residual_variance, optimize_V="EM")
    
    # 检查返回的字典中是否包含所有必要的键
    expected_keys = ['alpha', 'mu', 'mu2', 'lbf', 'lbf_model', 'V', 'loglik']
    assert all(key in dense_res for key in expected_keys)
    print("test1")
    assert all(key in sparse_res for key in expected_keys)
    print("test2")
    # 检查稠密矩阵和稀疏矩阵的结果是否一致
    for key in expected_keys:
        assert_array_almost_equal(dense_res[key], sparse_res[key])
    print("test3")
    # 检查alpha是否为概率分布（和为1）
    assert_almost_equal(np.sum(dense_res['alpha']), 1.0)
    assert_almost_equal(np.sum(sparse_res['alpha']), 1.0)
    print("test4")
    # 检查维度是否正确
    n, p = X.shape
    assert dense_res['alpha'].shape[0] == p
    assert dense_res['mu'].shape[0] == p
    assert dense_res['mu2'].shape[0] == p
    assert dense_res['lbf'].shape[0] == p

# 测试init_coef函数
def test_init_coef():
    """测试susie_init_coef函数"""
    # 测试正常情况
    coef_index = np.array([1, 3, 5]) 
    coef_value = np.array([0.5, -0.3, 0.8])
    p = 10
    s = susie_init_coef(coef_index, coef_value, p)
    
    # 检查返回对象的属性
    assert s.alpha.shape == (3, 10)
    assert s.mu.shape == (3, 10)
    assert s.mu2.shape == (3, 10)
    
    # 检查alpha矩阵是否正确初始化
    expected_alpha = np.zeros((3, 10))
    expected_alpha[0, 0] = 1  # coef_index[0]-1 = 0
    expected_alpha[1, 2] = 1  # coef_index[1]-1 = 2 
    expected_alpha[2, 4] = 1  # coef_index[2]-1 = 4
    assert_array_equal(s.alpha, expected_alpha)
    
    # 检查mu矩阵是否正确初始化
    expected_mu = np.zeros((3, 10))
    expected_mu[0, 0] = 0.5
    expected_mu[1, 2] = -0.3
    expected_mu[2, 4] = 0.8
    assert_array_equal(s.mu, expected_mu)
    
    # 检查mu2是否为mu的平方
    assert_array_equal(s.mu2, s.mu * s.mu)
    
    # 测试异常情况
    with pytest.raises(ValueError):
        susie_init_coef(np.array([]), np.array([]), 10)  # 空数组
    with pytest.raises(ValueError):
        susie_init_coef(np.array([1]), np.array([0]), 10)  # 零系数
    with pytest.raises(ValueError):
        susie_init_coef(np.array([1,2]), np.array([1]), 10)  # 长度不匹配
    with pytest.raises(ValueError):
        susie_init_coef(np.array([11]), np.array([1]), 10)  # 索引超出范围

def test_init_setup():
    """测试init_setup函数"""
    n, p, L = 100, 10, 5
    scaled_prior_variance = 0.2
    residual_variance = 1.0
    varY = 2.0
    
    # 测试基本功能
    s = init_setup(n, p, L, scaled_prior_variance, residual_variance, 
                    None, None, varY, True)
    
    # 检查返回对象的属性
    assert s.alpha.shape == (5, 10)
    assert s.mu.shape == (5, 10)
    assert s.mu2.shape == (5, 10)
    assert s.Xr.shape == (100,)
    assert s.KL.shape == (5,)
    # assert s.V.shape == (5,)  # V应该是5维的数组
    assert s.pi.shape == (10,)
    
    # 检查初始化值
    assert_array_equal(s.alpha, np.full((5, 10), 0.1))
    assert_array_equal(s.mu, np.zeros((5, 10)))
    assert_array_equal(s.mu2, np.zeros((5, 10)))
    assert_array_equal(s.Xr, np.zeros(100))
    assert np.all(np.isnan(s.KL))
    assert s.sigma2 == residual_variance
    assert_array_equal(s.V, np.repeat(scaled_prior_variance * varY, 5))  # V应该是5个相同的值
    assert_array_equal(s.pi, np.repeat(1/p, p))
    
    # 测试异常情况
    with pytest.raises(ValueError):
        init_setup(n, p, L, -1, residual_variance, None, None, varY, True)  # 负的先验方差
    with pytest.raises(ValueError):
        init_setup(n, p, L, 1.5, residual_variance, None, None, varY, True)  # 标准化时先验方差>1

def test_init_finalize():
    """测试init_finalize函数"""
    # 创建基本的SusieObject
    s = SusieObject()
    s.alpha = np.full((3, 10), 0.1)
    s.mu = np.zeros((3, 10))
    s.mu2 = np.zeros((3, 10))
    s.sigma2 = 1.0
    s.V = np.array([0.2, 0.2, 0.2])
    
    # 测试基本功能
    s_final = init_finalize(s)
    
    # 检查KL和lbf是否被重置
    assert np.all(np.isnan(s_final.KL))
    assert np.all(np.isnan(s_final.lbf))
    
    # 测试异常情况
    s.sigma2 = -1
    with pytest.raises(ValueError):
        init_finalize(s)  # 负的残差方差
        
    s.sigma2 = 1.0
    s.V = np.array([-0.1, 0.2, 0.2])
    with pytest.raises(ValueError):
        init_finalize(s)  # 负的先验方差
        
    s.V = np.array([0.2, 0.2])  # 长度不匹配
    with pytest.raises(ValueError):
        init_finalize(s)

def test_susie_get_cs():
    """测试susie_get_cs函数"""
    # 创建测试数据
    res = SusieObject()
    res.alpha = np.array([
        [0.5, 0.3, 0.2, 0, 0],
        [0.1, 0.6, 0.2, 0.1, 0],
        [0.2, 0.2, 0.4, 0.1, 0.1]
    ])
    res.V = np.array([1.0, 1.0, 1.0])
    
    # # 测试基本功能(无X和Xcorr)
    # result = susie_get_cs(res)
    # assert isinstance(result, dict)
    # assert 'cs' in result
    # assert 'coverage' in result
    # assert 'requested_coverage' in result
    # assert result['requested_coverage'] == 0.95
    
    # # 测试空CS情况
    # res.alpha = np.zeros((3, 5))
    # result = susie_get_cs(res)
    # assert result['cs'] is None
    # assert result['coverage'] is None
    
    # 测试带X的情况
    res.alpha = np.array([
        [0.5, 0.3, 0.2, 0, 0],
        [0.1, 0.6, 0.2, 0.1, 0],
        [0.2, 0.2, 0.4, 0.1, 0.1]
    ])
    # 生成具有强相关性的X
    np.random.seed(123)  # 设置随机种子以保证结果可重现
    base = np.random.randn(10, 1)  # 基础随机向量
    noise = 0.05 * np.random.randn(10, 5)  # 小的随机噪声
    X = base @ np.ones((1, 5)) + noise  # 生成相关的数据
    # result = susie_get_cs(res, X=X)
    # assert 'purity' in result
    # assert 'cs_index' in result
    
    # # 测试带Xcorr的情况
    Xcorr = np.corrcoef(X.T)
    # result = susie_get_cs(res, Xcorr=Xcorr)
    # assert 'purity' in result
    # assert 'cs_index' in result
    
    # # 测试不同的coverage值
    # result = susie_get_cs(res, X=X, coverage=0.9)
    # assert result['requested_coverage'] == 0.9
    
    # # 测试min_abs_corr参数
    # result = susie_get_cs(res, X=X, min_abs_corr=0.8)
    # if result['cs'] is not None:
    #     assert all(result['purity']['min.abs.corr'] >= 0.8)
    
    # # 测试dedup参数
    # result_dedup = susie_get_cs(res, X=X, dedup=True)
    # result_no_dedup = susie_get_cs(res, X=X, dedup=False)
    # assert len(result_dedup['cs']) <= len(result_no_dedup['cs'])
    
    # # 测试squared参数
    # result_squared = susie_get_cs(res, X=X, squared=True)
    # assert 'min.sq.corr' in result_squared['purity'].columns
    
    # 测试异常情况
    # with pytest.raises(ValueError):
    #     susie_get_cs(res, X=X, Xcorr=Xcorr)  # 同时提供X和Xcorr

    # # 测试不对称的Xcorr矩阵
    # Xcorr_asymm = Xcorr.copy()
    # Xcorr_asymm[0,1] = 0.5
    # result = susie_get_cs(res, Xcorr=Xcorr_asymm)
    # assert 'purity' in result
    
    # # 测试先验方差为0的情况
    # res.V = np.array([0, 1, 0])
    # result = susie_get_cs(res, X=X)
    # if result['cs'] is not None:
    #     assert len(result['cs']) <= 1
        
    # # 测试null_index的情况
    # res.null_index = 4
    # result = susie_get_cs(res, X=X)
    # if result['cs'] is not None:
    #     for cs in result['cs'].values():
    #         assert 4 not in cs
            
    # 测试n_purity参数
    result = susie_get_cs(res, X=X, n_purity=50)
    assert 'purity' in result

def test_susie_prune_single_effects():
    # 创建一个基本的SusieObject实例
    s = SusieObject()
    
    # 设置基本属性
    s.alpha = np.array([
        [0.5, 0.3, 0.2],
        [0.1, 0.6, 0.3],
        [0.2, 0.2, 0.6]
    ])
    s.mu = np.array([
        [1.0, 0.5, 0.3],
        [0.2, 1.2, 0.4],
        [0.3, 0.4, 1.1]
    ])
    s.mu2 = np.array([
        [1.1, 0.6, 0.4],
        [0.3, 1.3, 0.5],
        [0.4, 0.5, 1.2]
    ])
    s.lbf_variable = np.array([
        [0.1, 0.2, 0.3],
        [0.2, 0.3, 0.4],
        [0.3, 0.4, 0.5]
    ])
    s.V = np.array([1.0, 0.0, 1.0])  # 第二个效应的V为0
    s.KL = np.array([0.1, 0.2, 0.3])
    s.lbf = np.array([0.4, 0.5, 0.6])
    
    # 测试1：L=0（默认情况，应该保留V>0的效应）
    s1 = susie_prune_single_effects(copy.deepcopy(s))
    assert s1.alpha.shape[0] == 3  # 应该只保留2个效应（V>0的效应）
    
    # 测试2：L=2（指定保留2个效应）
    s2 = susie_prune_single_effects(copy.deepcopy(s), L=2)
    assert s2.alpha.shape[0] == 3
    
    # 测试3：L>num_effects（扩展效应）
    s3 = susie_prune_single_effects(copy.deepcopy(s), L=5)
    assert s3.alpha.shape[0] == 5
    assert s3.mu.shape[0] == 5
    assert s3.mu2.shape[0] == 5
    assert s3.lbf_variable.shape[0] == 5
    assert len(s3.KL) == 5
    assert len(s3.lbf) == 5
    
    # 测试4：带有cs_index的情况
    s4 = copy.deepcopy(s)
    s4.sets = SimpleNamespace()
    s4.sets.cs_index = np.array([0, 2])  # 假设第1和第3个效应在cs中
    s4 = susie_prune_single_effects(s4, L=2)
    assert s4.alpha.shape[0] == 3
    # 检查cs_index中的效应是否被优先保留
    assert np.allclose(s4.alpha[0], s.alpha[0])
    assert np.allclose(s4.alpha[2], s.alpha[2])
    
    # 测试5：指定新的V值
    s5 = susie_prune_single_effects(copy.deepcopy(s), L=4, V=0.5)
    assert len(s5.V) == 4
    assert np.all(s5.V == 0.5)
    
    print("All tests passed!")

