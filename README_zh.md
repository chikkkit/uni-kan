# Uni-KAN

这是一个基于PyTorch的Python库，用于通用地构建KAN型网络。通用KAN型网络被命名为
Uni-KAN。

## 安装

```bash
pip install unikan
```

## 特性

- 通用的构建KAN型网络的框架(uni-kan)
- 提供SKAN(单参数KAN)的快速实现
- 包含一些在SKAN相关文章中预定义好的基函数
- Pytoch兼容（GPU加速等）

## 快速开始

### Uni-KAN示例

```python
from unikan import UniversalKAN
import unikan.basis as basis

# 定义每层的节点配置
function_lists = [
    [
        {'function': basis.lshifted_softplus, 'param_num': 1, 
         'node_type': 'add', 'node_num': 90, 'use_bias': True},
        {'function': basis.lshifted_softplus, 'param_num': 1, 
         'node_type': 'mul', 'node_num': 10, 'use_bias': True}
    ],
    [
        {'function': basis.lshifted_softplus, 'param_num': 1, 
         'node_type': 'add', 'node_num': 10, 'use_bias': True}
    ]
]

# 创建通用KAN网络
net = UniversalKAN([784, 100, 10],  # 层大小
                   function_lists=function_lists,  # 节点配置
                   device=device)  # 设备选择
```

### SKAN示例

```python
import torch
from unikan import SKAN_pure
import unikan.basis as basis

# 创建SKAN网络
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = SKAN_pure([784, 100, 10],  # 层大小：输入层784，隐藏层100，输出层10
                basis_function=basis.lshifted_softplus,  # 基函数
                device=device)  # 设备选择
```

## 许可

MIT License

## 引用

如果您在研究中使用了本项目，请引用：

```bibtex
@software{unikan,
  title = {Uni-KAN: Universal KAN-type Networks Builder},
  author = {Chen, Zhijie},
  year = {2024},
  url = {https://github.com/chikkkit/uni-kan}
}
```

## 联系方式

- 邮箱: zhijiechencs@gmail.com