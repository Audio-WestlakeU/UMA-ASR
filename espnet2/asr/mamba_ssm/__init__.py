'''
Author: FnoY fangying@westlake.edu.cn
LastEditors: FnoY0723 fangying@westlake.edu.cn
LastEditTime: 2024-03-15 20:55:13
FilePath: /espnet/espnet2/asr/mamba_ssm/__init__.py
'''
__version__ = "1.2.0.post1"

from espnet2.asr.mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn
from espnet2.asr.mamba_ssm.modules.mamba_simple import Mamba

