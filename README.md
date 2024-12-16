# Uni-KAN

<p align="center"><b>English</b> / <a href="https://github.com/chikkkit/uni-kan/blob/main/README_zh.md">简体中文</a></p>

This is a Python library based on PyTorch for universally building KAN-type networks. Universal KAN-type networks is named Uni-KAN.

## Installation

```bash
pip install unikan
```

## Features

- Framework for building Universal KAN type network (uni-kan)
- SKAN (Single-Parameterized KAN) implementation 
- Contain pre defined basis functions in SKAN
- Pytorch compatible (GPU acceleration, etc.)

## Quick Start

### Universal KAN Example

```python
from unikan import UniversalKAN
import unikan.basis as basis

# Define node configurations for each layer
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

# Create universal KAN network
net = UniversalKAN([784, 100, 10],  # layer sizes
                   function_lists=function_lists,  # node configs
                   device=device)  # device selection
```

### SKAN Example

```python
import torch
from unikan import SKAN_pure
import unikan.basis as basis

# Create SKAN network
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = SKAN_pure([784, 100, 10],  # layer sizes: input 784, hidden 100, output 10
                basis_function=basis.lshifted_softplus,  # basis function
                device=device)  # device selection
```

## License

MIT License

## Citation

If you use this project in your research, please cite:

```bibtex
@misc{chen2024lssskanefficientkolmogorovarnoldnetworks,
      title={LSS-SKAN: Efficient Kolmogorov-Arnold Networks based on Single-Parameterized Function}, 
      author={Zhijie Chen and Xinglin Zhang},
      year={2024},
      eprint={2410.14951},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2410.14951}, 
}
```

## Contact

- Email: zhijiechencs@gmail.com
