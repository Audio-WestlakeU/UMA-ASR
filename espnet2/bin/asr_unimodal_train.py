'''
Author: FnoY fangying@westlake.edu.cn
LastEditTime: 2023-09-15 14:22:03
FilePath: /espnet/espnet2/bin/asr_unimodal_train.py
'''
#!/usr/bin/env python3
from espnet2.tasks.asr_unimodal import ASRTask


def get_parser():
    parser = ASRTask.get_parser()
    return parser


def main(cmd=None):
    r"""ASR training.

    Example:

        % python asr_train.py asr --print_config --optim adadelta \
                > conf/train_asr.yaml
        % python asr_train.py --config conf/train_asr.yaml
    """
    ASRTask.main(cmd=cmd)


if __name__ == "__main__":
    main()
