#!/usr/bin/env bash
###
 # @Author: FnoY fangying@westlake.edu.cn
 # @LastEditTime: 2023-09-15 13:43:56
 # @FilePath: /espnet/egs2/hkust/asr1/run_unimodal.sh
### 
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
# CUDA_VISIBLE_DEVICES=7
set -e
set -u
set -o pipefail

train_set=train_nodup
valid_set=train_dev
test_sets="dev"


asr_config=umaconf/train_asr_uma_conformer.yaml
inference_config=umaconf/decode_asr_uma.yaml

lm_config=conf/tuning/train_lm_transformer.yaml
use_lm=false
expdir=exp_uma_conformer
inference_asr_model=valid.cer.ave_10best.pth

# speed perturbation related
# (train_set will be "${train_set}_sp" if speed_perturb_factors is specified)
speed_perturb_factors="0.9 1.0 1.1"

./asr_unimodal.sh                                               \
    --nj 64 \
    --inference_nj 1  \
    --ngpu 1 \
    --lang zh                                          \
    --audio_format flac                                \
    --feats_type raw                                   \
    --token_type char                                  \
    --nlsyms_txt data/nlsyms.txt \
    --use_lm ${use_lm}                                 \
    --expdir ${expdir}                                 \
    --inference_asr_model ${inference_asr_model}       \
    --lm_config "${lm_config}"                         \
    --asr_config "${asr_config}"                       \
    --inference_config "${inference_config}"           \
    --train_set "${train_set}"                         \
    --valid_set "${valid_set}"                         \
    --test_sets "${test_sets}"                         \
    --speed_perturb_factors "${speed_perturb_factors}" \
    --lm_train_text "data/${train_set}/text" "$@"