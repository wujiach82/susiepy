class ScaledMatrix:
    """带有缩放属性的矩阵类"""
    def __init__(self, data, center=None, scale=None, d=None):
        self.data = data
        self.center = center
        self.scale = scale
        self.d = d
        
    def __array__(self):
        """允许像NumPy数组一样使用"""
        return self.data
    
    def __matmul__(self, other):
        """实现矩阵乘法 @"""
        return self.data @ other
        
    def __rmatmul__(self, other):
        """实现右矩阵乘法"""
        return other @ self.data
    
    def __getattr__(self, name):
        """委托未知属性到data"""
        return getattr(self.data, name)
    
    def __getitem__(self, key):
        """支持切片操作"""
        return self.data[key]
    
    @property
    def T(self):
        """转置"""
        return self.data.T
    
    @property
    def shape(self):
        """形状"""
        return self.data.shape
        
    def copy(self):
        """复制"""
        return ScaledMatrix(
            self.data.copy(),
            self.center.copy() if self.center is not None else None,
            self.scale.copy() if self.scale is not None else None,
            self.d.copy() if self.d is not None else None
        )


class SusieObject:
    """
    SuSiE (Sum of Single Effects) 对象类
    """
    
    def __init__(self):
        """初始化SusieObject实例"""
        self.__susie__ = True  # 标识这是一个susie对象
        
        # 初始化基本属性
        self.alpha = None
        self.mu = None
        self.mu2 = None
        self.V = None
        self.sigma2 = None
        self.Xr = None
        self.pi = None
        self.lbf_variable = None
        self.null_index = None 
        self.lbf = None
        # 拟合相关属性
        self.elbo = None
        self.niter = 0
        self.converged = False
        self.fitted = None
        self.intercept = 0
        self.KL = None
        # 统计和分析相关属性
        self.sets = {
            'cs': None,               # 可信集列表，每个元素是一个包含变量索引的数组
            'purity': None,           # 可信集的纯度DataFrame，包含min/mean/median相关性
            'cs_index': None,         # 每个可信集对应的效应索引
            'coverage': None,         # 每个可信集的实际覆盖率
            'requested_coverage': None # 请求的覆盖率阈值
        }
        self.pip = None
        self.z = None
        self.trace = None
        self.X_column_scale_factors = None

    def __str__(self):
        """返回对象的字符串表示"""
        cs_count = len(self.sets['cs']) if self.sets['cs'] is not None else 0
        return f"SuSiE对象 (迭代次数: {self.niter}, 收敛: {self.converged}, 可信集数量: {cs_count})"
    
    def __repr__(self):
        """返回对象的详细表示"""
        return self.__str__()