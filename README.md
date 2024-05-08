# Prompt-Distribution-Learning-for-CLIP


## Setup
1. Download the CLIP model pretrained weights from [(Here)](https://drive.google.com/drive/folders/1Jw1u5xkyeY7hkmsyV6nqAKsXL1OMGCg6?usp=sharing). Create a folder ''ckpt_clip_github'' and then put the pre-trained weights in this folder as: 
    ```
    Prompt-Distribution-Laerning-for-CLIP
    |-- ckpt_clip_github
        |-- RN50.pt
    ```

## Run Experiments
1. To train class-agnostic model on ImageNet-LT dataset and update the prompts only, run:
    ```
    sh cls-agn_prp.sh
    ```
2. To train class-specific model on ImageNet-LT dataset and update the prompts only, run:
    ```
    sh cls-spec_prp.sh
    ```
3. To train class-condition model on ImageNet-LT dataset and update the prompts only, run:
    ```
    sh cls-condi_prp.sh
    ```