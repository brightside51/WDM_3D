#!/bin/bash
#
#SBATCH --partition=gpu_min80gb                                   # Partition (check with "$sinfo")
#SBATCH --output=../MetaBreast/logs/wdm3d/V3/output.out           # Filename with STDOUT. You can use special flags, such as %N and %j.
#SBATCH --error=../MetaBreast/logs/wdm3d/V3/error.err             # (Optional) Filename with STDERR. If ommited, use STDOUT.
#SBATCH --job-name=wdm3d                                        # (Optional) Job name
#SBATCH --time=14-00:00                                             # (Optional) Time limit (D: days, HH: hours, MM: minutes)
#SBATCH --qos=gpu_min80GB                                           # (Optional) 01.ctm-deep-05

# general settings
GPU=0;                    # gpu to use
SEED=42;                  # randomness seed for sampling
CHANNELS=64;              # number of model base channels (we use 64 for all experiments)
MODE='train';             # train vs sample
DATASET='metabreast';          # brats or lidc-idri
MODEL='ours_wnet_128';    # 'ours_unet_256', 'ours_wnet_128', 'ours_wnet_256'

# settings for sampling/inference
ITERATIONS=0;             # training iteration (as a multiple of 1k) checkpoint to use for sampling
SAMPLING_STEPS=0;         # number of steps for accelerated sampling, 0 for the default 1000
RUN_DIR="";               # tensorboard dir to be set for the evaluation

# detailed settings (no need to change for reproducing)
if [[ $MODEL == 'ours_unet_128' ]]; then
  echo "MODEL: WDM (U-Net) 128 x 128 x 128";~
  #CHANNEL_MULT=1,2,2,4,4;
  CHANNEL_MULT=1,2,4;
  IMAGE_SIZE=128;
  ADDITIVE_SKIP=True;
  USE_FREQ=False;
  BATCH_SIZE=1;
elif [[ $MODEL == 'ours_unet_256' ]]; then
  echo "MODEL: WDM (U-Net) 256 x 256 x 256";
  CHANNEL_MULT=1,2,2,4,4,4;
  IMAGE_SIZE=256;
  ADDITIVE_SKIP=True;
  USE_FREQ=False;
  BATCH_SIZE=1;
elif [[ $MODEL == 'ours_wnet_128' ]]; then
  echo "MODEL: WDM (WavU-Net) 128 x 128 x 128";
  #CHANNEL_MULT=1,2,2,4,4;
  CHANNEL_MULT=1,2,2,4;
  IMAGE_SIZE=128;
  ADDITIVE_SKIP=False;
  USE_FREQ=True;
  BATCH_SIZE=1;
elif [[ $MODEL == 'ours_wnet_256' ]]; then
  echo "MODEL: WDM (WavU-Net) 256 x 256 x 256";
  CHANNEL_MULT=1,2,2,4,4,4;
  IMAGE_SIZE=256;
  ADDITIVE_SKIP=False;
  USE_FREQ=True;
  BATCH_SIZE=1;
else
  echo "MODEL TYPE NOT FOUND -> Check the supported configurations again";
fi

# some information and overwriting batch size for sampling
# (overwrite in case you want to sample with a higher batch size)
# no need to change for reproducing
if [[ $MODE == 'sample' ]]; then
  echo "MODE: sample"
  BATCH_SIZE=1;
elif [[ $MODE == 'train' ]]; then
  if [[ $DATASET == 'brats' ]]; then
    echo "MODE: training";
    echo "DATASET: BRATS";
    DATA_DIR=~/wdm-3d/data/BRATS/;
  elif [[ $DATASET == 'lidc-idri' ]]; then
    echo "MODE: training";
    echo "Dataset: LIDC-IDRI";
    DATA_DIR=~/wdm-3d/data/LIDC-IDRI/;
  elif [[ $DATASET == 'metabreast' ]]; then
    echo "MODE: training";
    echo "Dataset: METABREAST";
  else
    echo "DATASET NOT FOUND -> Check the supported datasets again";
  fi
fi

COMMON="
--dataset=${DATASET}
--num_channels=${CHANNELS}
--class_cond=False
--num_res_blocks=3
--num_heads=1
--learn_sigma=False
--use_scale_shift_norm=False
--attention_resolutions=
--channel_mult=${CHANNEL_MULT}
--diffusion_steps=1000
--noise_schedule=linear
--rescale_learned_sigmas=False
--rescale_timesteps=True
--dims=3
--batch_size=${BATCH_SIZE}
--num_groups=32
--in_channels=8
--out_channels=8
--bottleneck_attention=False
--resample_2d=True
--renormalize=True
--additive_skips=${ADDITIVE_SKIP}
--use_freq=${USE_FREQ}
--predict_xstart=True
"
TRAIN="
--data_dir=${DATA_DIR}
--resume_checkpoint=
--resume_step=0
--image_size=${IMAGE_SIZE}
--use_fp16=False
--lr=1e-7
--save_interval=500000
--num_workers=12
--devices=${CPU}
"
SAMPLE="
--data_dir=${DATA_DIR}
--data_mode=${DATA_MODE}
--seed=${SEED}
--image_size=${IMAGE_SIZE}
--use_fp16=False
--model_path=./${RUN_DIR}/checkpoints/${DATASET}_${ITERATIONS}000.pt
--devices=${GPU}
--output_dir=./results/${RUN_DIR}/${DATASET}_${MODEL}_${ITERATIONS}000/
--num_samples=1000
--use_ddim=False
--sampling_steps=${SAMPLING_STEPS}
--clip_denoised=True
"

#set CUDA_LAUNCH_BLOCKING = 1
#SET TORCH_USE_CUDA_DSA = 1

# run the python scripts
if [[ $MODE == 'train' ]]; then
  set TORCH_USE_CUDA_DSA=1
  CUDA_LAUNCH_BLOCKING=1 python scripts/generation_train.py $TRAIN $COMMON;
else
  CUDA_LAUNCH_BLOCKING=1 python scripts/generation_sample.py $SAMPLE $COMMON;
fi
