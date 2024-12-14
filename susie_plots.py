import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import norm
from susie_utils import in_CS, n_in_CS

# def susie_plot(model, y, add_bar=False, pos=None, b=None, max_cs=400, add_legend=None, **kwargs):
#     """
#     SuSiE plot function.
    
#     Parameters
#     ----------
#     model : object
#         A SuSiE fit or vector of z-scores/PIPs
#     y : str
#         Type of plot: "z_original", "z", "PIP", or "log10PIP"
#     add_bar : bool
#         Whether to add vertical bars
#     pos : array-like or dict
#         Position indices or dict with attr, start, end
#     b : array-like
#         True effects to highlight in red
#     max_cs : float
#         Maximum credible set size/purity to display
#     add_legend : bool or str
#         Whether and where to add legend
#     """
#     # 检查是否是susie对象
#     is_susie = hasattr(model, 'alpha') and hasattr(model, 'pip')
#     ylab = y
    
#     # 定义颜色列表（与R代码完全相同）
#     color = [
#         "#1E90FF",  # dodgerblue
#         "#228B22",  # forestgreen
#         "#6A3D9A",  # purple
#         "#FF7F00",  # orange
#         "#FFD700",  # gold
#         "#87CEEB",  # skyblue
#         "#FB9A99",  # light pink
#         "#98FB98",  # palegreen
#         "#CAB2D6",  # light purple
#         "#FDBF6F",  # light orange
#         "#B8B8B8",  # gray
#         "#F0E68C",  # khaki
#         "#800000",  # maroon
#         "#FF69B4",  # hotpink
#         "#FF1493",  # deeppink
#         "#0000FF",  # blue
#         "#4682B4",  # steelblue
#         "#00CED1",  # darkturquoise
#         "#00FF00",  # lime
#         "#8B8B00",  # yellow4
#         "#CDC900",  # yellow3
#         "#8B4500",  # darkorange4
#         "#A52A2A"   # brown
#     ]
    
#     # 处理y参数
#     if y == "z":
#         if is_susie:
#             if not hasattr(model, 'z'):
#                 raise ValueError("z-scores are not available from SuSiE fit")
#             zneg = -np.abs(model.z)
#         else:
#             zneg = -np.abs(model)
#         p = -np.log10(2 * norm.cdf(zneg))
#         ylab = "-log10(p)"
#     elif y == "z_original":
#         if is_susie:
#             if not hasattr(model, 'z'):
#                 raise ValueError("z-scores are not available from SuSiE fit")
#             p = model.z
#         else:
#             p = model
#         ylab = "z score"
#     elif y == "PIP":
#         p = model.pip if is_susie else model
#     elif y == "log10PIP":
#         p = np.log10(model.pip if is_susie else model)
#         ylab = "log10(PIP)"
#     else:
#         if is_susie:
#             raise ValueError("Need to specify z_original, z, PIP or log10PIP for SuSiE fits")
#         p = model
    
#     # 初始化位置和效应
#     if b is None:
#         b = np.zeros(len(p))
#     if pos is None:
#         pos = pos = np.arange(len(p))  # 1-based indexing like R
    
#     # 处理pos参数
#     start = 0
#     if isinstance(pos, dict):
#         if not all(k in pos for k in ['attr', 'start', 'end']):
#             raise ValueError("pos argument should be a dict with 'attr', 'start', 'end' keys")
#         if not hasattr(model, pos['attr']):
#             raise ValueError(f"Cannot find attribute {pos['attr']} in input model object")
#         if pos['start'] >= pos['end']:
#             raise ValueError("Position start should be smaller than end")
        
#         # 处理位置调整
#         attr_values = getattr(model, pos['attr'])
#         start = min(min(attr_values), pos['start'])
#         end = max(max(attr_values), pos['end'])
#         pos_with_value = attr_values - start + 1
        
#         # 调整alpha和p
#         new_p = np.full(end - start + 1, min(p))
#         new_p[pos_with_value - 1] = p
#         p = new_p
        
#         # 调整cs
#         if hasattr(model, 'sets') and model.sets is not None and 'cs' in model.sets:
#             for cs_key in model.sets['cs']:
#                 model.sets['cs'][cs_key] = pos_with_value[model.sets['cs'][cs_key] - 1]
        
#         # 调整pos范围
#         start_adj = -min(min(attr_values) - pos['start'], 0)
#         end_adj = max(max(attr_values) - pos['end'], 0)
#         pos = np.arange(1 + start_adj, len(p) - end_adj + 1)
    
#     # 创建图形
#     plt.figure(figsize=(8, 6.5))
    
#     # 绘制基础点
#     plt.scatter(pos + start, p[pos], c='black', s=20)
    
#     # 处理credible sets
#     legend_text = {'col': [], 'purity': [], 'size': []}
#     if is_susie and hasattr(model, 'sets') and model.sets is not None and model.sets.get('cs') is not None:
#         for i in reversed(range(model.alpha.shape[0])):
#             if model.sets.get('cs_index') is not None and i not in model.sets.get('cs_index'):
#                 continue
#             x0 = 0
#             y1 = 0
#             if model.sets.get('cs_index') is not None:
#                 cs_idx = np.where(model.sets.get('cs_index') == i)[0][0]
#                 purity = model.sets['purity'].iloc[cs_idx]['min.abs.corr']
                

                
#                 # 使用条件判断
#                 if (model.sets.get('purity') is not None and 
#                     max_cs < 1 and purity >= max_cs):
#                     # 第一种情况：基于purity的筛选
#                     cs_key = f'L{i}'
#                     if cs_key in model.sets['cs']:
#                         x0 = np.intersect1d(pos, model.sets['cs'][cs_key])
#                         if len(x0) > 0:
#                             y1 = p[x0-1]
#                 elif n_in_CS(model, model.sets['requested_coverage'])[i] < max_cs:
#                     # 第二种情况：基于credible set大小的筛选
#                     in_cs = in_CS(model, model.sets['requested_coverage'])[i, :] > 0
#                     x0 = np.intersect1d(pos, np.where(in_cs)[0] + 1)  # 1-based indexing
#                     if len(x0) > 0:
#                         y1 = p[x0-1]
#                 else:
#                     x0 = 0
#                     y1 = 0
                
#                 if add_bar:
#                     for x, y in zip(x0, y1):
#                         plt.plot([x + start, x + start], [0, y], 
#                                 color='gray', linewidth=1.5)
                
#                 # 正确的画空心圈的方式
#                 plt.plot(x0 + start, y1, 
#                         color=color[0],           # 边框颜色
#                         marker='o',               # 圆形
#                         markersize=8,            # 对应R中的cex=1.5
#                         markerfacecolor='none',   # 设置为空心
#                         markeredgewidth=2.5,      # 对应R中的lwd=2.5
#                         linestyle='none')         # 只画点，不画线
            
#             # # 更新图例信息
#             # legend_text['col'].append(color[0])
#             # legend_text['purity'].append(round(purity, 4))
#             # legend_text['size'].append(len(x0))
            
#             # 循环颜色
#             color = color[1:] + [color[0]]
        
#         # 添加图例
#         if len(legend_text['col']) > 0 and add_legend is not None and add_legend is not False:
#             text = []
#             for i in range(len(legend_text['col'])):
#                 if legend_text['size'][i] == 1:
#                     text.append(f"L{i}: C=1")
#                 else:
#                     text.append(f"L{i}: C={legend_text['size'][i]}/R={legend_text['purity'][i]}")
            
#             if not isinstance(add_legend, str) or add_legend not in [
#                 'bottomright', 'bottom', 'bottomleft', 'left',
#                 'topleft', 'top', 'topright', 'right', 'center'
#             ]:
#                 add_legend = 'upper right'
            
#             plt.legend(text, loc=add_legend, frameon=False, fontsize=8)
    
#     # 标注真实效应
#     if b is not None:
#         nonzero_idx = np.where((b != 0) & (~np.isnan(b)))[0]
#         if len(nonzero_idx) > 0:
#             plt.scatter(pos[nonzero_idx] + start, p[nonzero_idx], 
#                        c='red', s=20)
    
#     # 设置轴标签
#     plt.xlabel(kwargs.get('xlab', 'variable'))
#     plt.ylabel(kwargs.get('ylab', ylab))
    
#     plt.tight_layout()
#     # plt.show()
    
#     return plt.gcf()
def susie_plot(model, y, add_bar=False, pos=None, b=None, max_cs=400, add_legend=None, **kwargs):
    """
    SuSiE plot function.
    
    Parameters
    ----------
    model : object
        A SuSiE fit or vector of z-scores/PIPs
    y : str
        Type of plot: "z_original", "z", "PIP", or "log10PIP"
    add_bar : bool
        Whether to add vertical bars
    pos : array-like or dict
        Position indices or dict with attr, start, end
    b : array-like
        True effects to highlight in red
    max_cs : float
        Maximum credible set size/purity to display
    add_legend : bool or str
        Whether and where to add legend
    """
    # 检查是否是susie对象
    is_susie = hasattr(model, 'alpha') and hasattr(model, 'pip')
    ylab = y
    
    # 定义颜色列表（与R代码完全相同）
    color = [
        "#1E90FF",  # dodgerblue
        "#228B22",  # forestgreen
        "#6A3D9A",  # purple
        "#FF7F00",  # orange
        "#FFD700",  # gold
        "#87CEEB",  # skyblue
        "#FB9A99",  # light pink
        "#98FB98",  # palegreen
        "#CAB2D6",  # light purple
        "#FDBF6F",  # light orange
        "#B8B8B8",  # gray
        "#F0E68C",  # khaki
        "#800000",  # maroon
        "#FF69B4",  # hotpink
        "#FF1493",  # deeppink
        "#0000FF",  # blue
        "#4682B4",  # steelblue
        "#00CED1",  # darkturquoise
        "#00FF00",  # lime
        "#8B8B00",  # yellow4
        "#CDC900",  # yellow3
        "#8B4500",  # darkorange4
        "#A52A2A"   # brown
    ]
    
    # 处理y参数
    if y == "z":
        if is_susie:
            if not hasattr(model, 'z'):
                raise ValueError("z-scores are not available from SuSiE fit")
            zneg = -np.abs(model.z)
        else:
            zneg = -np.abs(model)
        p = -np.log10(2 * norm.cdf(zneg))
        ylab = "-log10(p)"
    elif y == "z_original":
        if is_susie:
            if not hasattr(model, 'z'):
                raise ValueError("z-scores are not available from SuSiE fit")
            p = model.z
        else:
            p = model
        ylab = "z score"
    elif y == "PIP":
        p = model.pip if is_susie else model
    elif y == "log10PIP":
        p = np.log10(model.pip if is_susie else model)
        ylab = "log10(PIP)"
    else:
        if is_susie:
            raise ValueError("Need to specify z_original, z, PIP or log10PIP for SuSiE fits")
        p = model
    
    # 初始化位置和效应
    if b is None:
        b = np.zeros(len(p))
    if pos is None:
        pos = pos = np.arange(len(p))  # 1-based indexing like R
    
    # 处理pos参数
    start = 0
    if isinstance(pos, dict):
        if not all(k in pos for k in ['attr', 'start', 'end']):
            raise ValueError("pos argument should be a dict with 'attr', 'start', 'end' keys")
        if not hasattr(model, pos['attr']):
            raise ValueError(f"Cannot find attribute {pos['attr']} in input model object")
        if pos['start'] >= pos['end']:
            raise ValueError("Position start should be smaller than end")
        
        # 处理位置调整
        attr_values = getattr(model, pos['attr'])
        start = min(min(attr_values), pos['start'])
        end = max(max(attr_values), pos['end'])
        pos_with_value = attr_values - start + 1
        
        # 调整alpha和p
        new_p = np.full(end - start + 1, min(p))
        new_p[pos_with_value - 1] = p
        p = new_p
        
        # 调整cs
        if hasattr(model, 'sets') and model.sets is not None and 'cs' in model.sets:
            for cs_key in model.sets['cs']:
                model.sets['cs'][cs_key] = pos_with_value[model.sets['cs'][cs_key] - 1]
        
        # 调整pos范围
        start_adj = -min(min(attr_values) - pos['start'], 0)
        end_adj = max(max(attr_values) - pos['end'], 0)
        pos = np.arange(1 + start_adj, len(p) - end_adj + 1)
    
    # 创建图形
    plt.figure(figsize=(8, 6.5))
    
   # 分开处理真实效应点和非真实效应点
    if b is not None:
        # 找出真实效应和非真实效应的位置
        true_effects_mask = b[pos] != 0
        true_effects_pos = pos[true_effects_mask]
        non_effects_pos = pos[~true_effects_mask]
        
        # 先画非真实效应的点（黑色小点）
        plt.scatter(non_effects_pos + start, p[non_effects_pos], 
                   c='black', s=20, alpha=0.6)
        
        # 画真实效应的点（红色大点）
        if len(true_effects_pos) > 0:
            plt.scatter(true_effects_pos + start, p[true_effects_pos], 
                       c='darkred', s=20, alpha=0.6)
    else:
        # 如果没有真实效应信息，画所有点为黑色
        plt.scatter(pos + start, p[pos], c='black', s=20, alpha=0.6)
    
    # 处理credible sets
    legend_text = {'col': [], 'purity': [], 'size': []}
    if is_susie and hasattr(model, 'sets') and model.sets is not None and model.sets.get('cs') is not None:
        for i, cs_key in enumerate(model.sets['cs'].keys()):
            cs = model.sets['cs'][cs_key]
            if cs_key.startswith('L'):  # 确保是L1, L2这样的键
                cs_idx = int(cs_key[1:]) - 1  # 从L1转换为索引0
                
                # 获取该可信集的纯度
                purity = model.sets['purity'].iloc[cs_idx]['min.abs.corr']
                
                # 根据max_cs参数过滤可信集
                if (max_cs < 1 and purity >= max_cs) or (max_cs >= 1 and len(cs) <= max_cs):
                    cs_pos = np.intersect1d(pos, cs)
                    if len(cs_pos) > 0:
                        if add_bar:
                            plt.vlines(cs_pos + start, 0, p[cs_pos], 
                                     colors='gray', linewidth=1.5, alpha=0.3)


                # 正确的画空心圈的方式
                plt.plot(cs_pos + start, p[cs_pos], 
                        color=color[0],           # 边框颜色
                        marker='o',               # 圆形
                        markersize=8,            # 对应R中的cex=1.5
                        markerfacecolor='none',   # 设置为空心
                        markeredgewidth=2.5,      # 对应R中的lwd=2.5
                        linestyle='none')         # 只画点，不画线
            
            # # 更新图例信息
            # legend_text['col'].append(color[0])
            # legend_text['purity'].append(round(purity, 4))
            # legend_text['size'].append(len(x0))
            
            # 循环颜色
            color = color[1:] + [color[0]]
        
        # 添加图例
        if len(legend_text['col']) > 0 and add_legend is not None and add_legend is not False:
            text = []
            for i in range(len(legend_text['col'])):
                if legend_text['size'][i] == 1:
                    text.append(f"L{i}: C=1")
                else:
                    text.append(f"L{i}: C={legend_text['size'][i]}/R={legend_text['purity'][i]}")
            
            if not isinstance(add_legend, str) or add_legend not in [
                'bottomright', 'bottom', 'bottomleft', 'left',
                'topleft', 'top', 'topright', 'right', 'center'
            ]:
                add_legend = 'upper right'
            
            plt.legend(text, loc=add_legend, frameon=False, fontsize=8)
    
    # 标注真实效应
    if b is not None:
        nonzero_idx = np.where((b != 0) & (~np.isnan(b)))[0]
        if len(nonzero_idx) > 0:
            plt.scatter(pos[nonzero_idx] + start, p[nonzero_idx], 
                       c='red', s=20)
    
    # 设置轴标签
    plt.xlabel(kwargs.get('xlab', 'variable'))
    plt.ylabel(kwargs.get('ylab', ylab))
    
    plt.tight_layout()
    # plt.show()
    
    return plt.gcf()
def susie_plot_iteration(model, L, file_prefix=None, pos=None):
    def get_layer(obj, k, idx, vars):
        alpha = pd.DataFrame(obj['alpha'][:k, vars], columns=['L', 'variables', 'alpha'])
        alpha['L'] = alpha['L'].astype('category')
        sns.barplot(data=alpha, x='variables', y='alpha', hue='L')
        plt.title(f"Iteration {idx}")
        plt.show()

    k = min(model['alpha'].shape[0], L)
    vars = np.arange(model['alpha'].shape[1]) if pos is None else pos
    
    if file_prefix is None:
        file_prefix = "susie_plot"
    
    if 'trace' not in model:
        get_layer(model, k, model['niter'], vars)
    else:
        for i in range(1, len(model['trace'])):
            get_layer(model['trace'][i], k, i, vars)