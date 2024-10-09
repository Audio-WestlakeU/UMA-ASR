<!--
 * @Author: FnoY fangying@westlake.edu.cn
 * @LastEditors: fnoy 1084585914@qq.com
 * @LastEditTime: 2024-10-09 15:04:18
 * @FilePath: \UMA-ASR\README.md
-->
# UMA-ASR
This repository is the official implementation of unimodal aggregation (UMA) for automaticspeech recognition (ASR).

It consists of two works:
1. for non-autoregressive offline ASR: ["Unimodal Aggregation for CTC-based Speech Recognition" (ICASSP 2024)](https://ieeexplore.ieee.org/abstract/document/10448248)
2. for streaming ASR: ["Mamba for Streaming ASR Combined with Unimodal Aggregation" (submitted to ICASSP 2025)](https://arxiv.org/abs/2410.00070)
 
<div>
    </p>
    <a href="https://github.com/Audio-WestlakeU/UMA-ASR/"><img src="https://img.shields.io/badge/Platform-linux-lightgrey" alt="version"></a>
    <a href="https://github.com/Audio-WestlakeU/UMA-ASR/"><img src="https://img.shields.io/badge/Python-3.9-orange" alt="version"></a>
    <a href="https://github.com/Audio-WestlakeU/UMA-ASR/"><img src="https://img.shields.io/badge/PyTorch-1.13-brightgreen" alt="python"></a>
</div>

[Poster :star_struck:](https://sigport.org/sites/default/files/docs/fangying_UMA_poster4.0.pdf) **|** [Issues :sweat_smile:](https://github.com/Audio-WestlakeU/UMA-ASR/issues)
 **|** [Lab :hear_no_evil:](https://github.com/Audio-WestlakeU) **|** [Contact :kissing_heart:](fangying@westlake.edu.cn)

## Introduction

### For Non-autoregressive Offline ASR
A unimodal aggregation (UMA) is proposed to segment and integrate the feature frames that belong to the same text token, and thus to learn better feature representations for text tokens. The frame-wise features and weights are both derived from an encoder. Then, the feature frames with unimodal weights are integrated and further processed by a decoder. Connectionist temporal classification (CTC) loss is applied for training. Moreover, by integrating self-conditioned CTC into the proposed framework, the performance can be further noticeably improved.

<div align="center">
<image src="./uma.png"  width="800" alt="The proposed UMA model" />
</div>

### For Streaming ASR
Mamba, a recently proposed state space model, has demonstrated the ability to match or surpass Transformers in various tasks while benefiting from a linear complexity advantage. We explore the efficiency of Mamba encoder for streaming ASR and propose an associated lookahead mechanism for leveraging controllable future information. Additionally, a streaming-style unimodal aggregation (UMA) method is
implemented, which automatically detects token activity and streamingly triggers token output, and meanwhile aggregates feature frames for better learning token representation. Based on UMA, an early termination (ET) method is proposed to further reduce recognition latency.

<div align="center">
<image src="./mamba_uma.png"  width="500" alt="The proposed Mamba-UMA model" />
</div>


## Get started
1. The proposed method is implemented using [ESPnet2](https://github.com/espnet/espnet). So please make sure you have [installed ESPnet](https://espnet.github.io/espnet/installation.html#) successfully. 
2. Roll back [espnet](https://github.com/espnet/espnet/tree/v.202304) to the specified version as follows：
    ```
    git checkout v.202304
    ```
3. Clone the UMA-ASR codes by:
   ```
   git clone https://github.com/Audio-WestlakeU/UMA-ASR
   ```
4. Copy the configurations of the recipes in the [egs2](https://github.com/Audio-WestlakeU/UMA-ASR/tree/main/egs2) folder to the corresponding directory in "espnet/egs2/". At present, experiments have only been conducted on AISHELL-1, AISHELL-2, HKUST dataset. If you want to experiment on other Chinese datasets, you can refer to these configurations.  
5. Copy the files in the [espnet2](https://github.com/Audio-WestlakeU/UMA-ASR/tree/main/espnet2) folder to the corresponding folder in "espnet/espnet2", and check that the comment path in the file header matches your path.
6. To experiment, follow the [ESPnet's steps](https://espnet.github.io/espnet/espnet2_tutorial.html#recipes-using-espnet2). You can implement UMA method by simply changing **run.sh** from the command line to our **run_unimodal.sh**.  For example:
    ```
    ./run_unimodal.sh --stage 10 --stop_stage 13
    ```
    Be careful to change the permissions of the bash files to executable.
    ```
    chmod -x asr_unimodal.sh
    chmod -x run_unimodal.sh
    ```

## Citation
You can cite this paper like:

```
@inproceedings{fang2024unimodal,
  title={Unimodal aggregation for CTC-based speech recognition},
  author={Fang, Ying and Li, Xiaofei},
  booktitle={ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={10591--10595},
  year={2024},
  organization={IEEE}
}

@article{fang2024mambauma,
    title={Mamba for Streaming ASR Combined with Unimodal Aggregation},
    author={Ying Fang and Xiaofei Li},
    journal={arXiv preprint arXiv:2410.00070},
    year={2023}
}
```
