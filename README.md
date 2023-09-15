<!--
 * @Author: FnoY fangying@westlake.edu.cn
 * @LastEditors: fnoy 1084585914@qq.com
 * @LastEditTime: 2023-09-15 16:40:23
 * @FilePath: \UMA-ASR\README.md
-->
# UMA-ASR
This repository is the official implementation of  "Unimodal Aggregation for CTC-based Speech Recognition".

This work is submitted to ICASSP 2024.
<div>
    </p>
    <a href="https://github.com/Audio-WestlakeU/UMA-ASR/"><img src="https://img.shields.io/badge/Platform-linux-lightgrey" alt="version"></a>
    <a href="https://github.com/Audio-WestlakeU/UMA-ASR/"><img src="https://img.shields.io/badge/Python-3.9-orange" alt="version"></a>
    <a href="https://github.com/Audio-WestlakeU/UMA-ASR/"><img src="https://img.shields.io/badge/PyTorch-1.13-brightgreen" alt="python"></a>
</div>

[Paper :star_struck:](TBC) **|** [Issues :sweat_smile:](https://github.com/Audio-WestlakeU/UMA-ASR/issues)
 **|** [Lab :hear_no_evil:](https://github.com/Audio-WestlakeU) **|** [Contact :kissing_heart:](fangying@westlake.edu.cn)

## Introduction

This project works on non-autoregressive automatic speech recognition. A unimodal aggregation (UMA) is proposed to segment and integrate the feature frames that belong to the same text token, and thus to learn better feature representations for text tokens. The frame-wise features and weights are both derived from an encoder. Then, the feature frames with unimodal weights are integrated and further processed by a decoder. Connectionist temporal classification (CTC) loss is applied for training. Moreover, by integrating self-conditioned CTC into the proposed framework, the performance can be further noticeably improved.

<div align="center">
<image src="./uma.png"  width="800" alt="The proposed UMA model" />
</div>

## Get started
1. The proposed method is implemented using [ESPnet2](https://github.com/espnet/espnet). So please make sure you have [installed ESPnet](https://espnet.github.io/espnet/installation.html#) successfully. 
2. Clone the UMA-ASR codes by:
   ```
   git clone https://github.com/Audio-WestlakeU/UMA-ASR
   ```
3. Copy the configurations of the recipes in the [egs2](https://github.com/Audio-WestlakeU/UMA-ASR/tree/main/egs2) folder to the corresponding directory in "espnet/egs2/". At present, experiments have only been conducted on AISHELL-1, AISHELL-2, HKUST dataset. If you want to experiment on other Chinese datasets, you can refer to these configurations.  
4. Copy the files in the [espnet2](https://github.com/Audio-WestlakeU/UMA-ASR/tree/main/espnet2) folder to the corresponding folder in "espnet/espnet2", and check that the comment path in the file header matches your path.
5. To experiment, follow the [ESPnet's steps](https://espnet.github.io/espnet/espnet2_tutorial.html#recipes-using-espnet2). You can implement UMA method by simply changing **run.sh** from the command line to our **run_unimodal.sh**.  For example:
    ```
    ./run_unimodal.sh --stage 10 --stop_stage 13
    ```
    Be careful to change the permissions of the bash files to executable.
    ```
    chmod -x asr_unimodal.sh
    chmod -x run_unimodal.sh
    ```

