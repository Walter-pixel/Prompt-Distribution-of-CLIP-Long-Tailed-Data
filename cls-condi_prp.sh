#!/bin/bash


source /home/walterl/ml20_scratch/walterl/miniconda/bin/activate torch1.12_scratch
cd /home/walterl/ml20_scratch/walterl/vision_language/clip_imb_prompt_generator_ImageNet



python3 Imagenet_main_condi_prp.py \
--workers 30 \
--combine_mode avg \
--n_sample_trn 1 \
--n_sample_tst 10 \
--dataset ImageNet_LT \
--num_classes 1000 \
--dtype fp32 \
--batch_size 20 \
--resolution 224 \
--clip_config_path config/Imagenet_LT_nctx36.yaml \
--im_enc_type clip_rn50 \
--lr_type multistep \
--lr_ratio 0.1 \
--list_steplr 85 105 \
--epochs 120 \
--opt_type adam \
--lr_latent 1e-3 \
--lr_gen 1e-3 \
--latent_dim 10 \
--loss_type logit_adj \
--stoch_tau_range 1.15 \
--kl_coef 1e-15 \
--infer_at_epoch 10 \
--tensorboard .debug/log_imagenet/log_cls-condi_prp/trn1-tst10_bz350e120/tau1.15_nctx36_lat10_lr1e-3_kl1e-15_adam_stp85.105






python3 Imagenet_main_condi_prp.py \
--workers 30 \
--combine_mode avg \
--n_sample_trn 1 \
--n_sample_tst 10 \
--dataset ImageNet_LT \
--num_classes 1000 \
--dtype fp32 \
--batch_size 120 \
--resolution 224 \
--clip_config_path config/Imagenet_LT_nctx36.yaml \
--im_enc_type clip_rn50 \
--lr_type multistep \
--lr_ratio 0.1 \
--list_steplr 85 105 \
--epochs 120 \
--opt_type adam \
--lr_latent 1e-3 \
--lr_gen 1e-3 \
--latent_dim 10 \
--loss_type logit_adj \
--stoch_tau_range 1.25 \
--kl_coef 1e-15 \
--infer_at_epoch 10 \
--tensorboard .debug/log_imagenet/log_cls-condi_prp/trn1-tst10_bz350e120/tau1.25_nctx36_lat10_lr1e-3_kl1e-15_adam_stp85.105









