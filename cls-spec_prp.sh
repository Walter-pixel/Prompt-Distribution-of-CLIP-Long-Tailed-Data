#!/bin/bash


source /home/walterl/ml20_scratch/walterl/miniconda/bin/activate torch1.12_scratch
cd /home/walterl/ml20_scratch/walterl/vision_language/clip_imb_prompt_generator_ImageNet


python3 Imagenet_main_cls-spec_prp.py \
--workers 20 \
--combine_mode avg \
--n_sample_trn 1 \
--n_sample_tst 10 \
--dataset ImageNet_LT \
--num_classes 1000 \
--dtype fp32 \
--batch_size 500 \
--resolution 224 \
--clip_config_path config/Imagenet_LT_nctx15.yaml \
--im_enc_type clip_rn50 \
--lr_type coslr \
--epochs 120 \
--opt_type adam \
--lr_latent 1e-3 \
--lr_gen 1e-3 \
--latent_dim 10 \
--loss_type logit_adj \
--stoch_tau_range 1.2 \
--kl_coef 1e-15 \
--infer_at_epoch 5 \
--tensorboard ./log_imagenet/log_cls-spec_prp/trn1-tst10/tau1.2_nctx15_lat10_coslr1e-3_kl1e-15_adam











