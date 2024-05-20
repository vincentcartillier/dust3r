#!/bin/bash

cd /srv/essa-lab/flash3/vcartillier3/egoexo-view-synthesis/dependencies/dust3r/

source ~/.bashrc

OUTPUT_DIR="$1"

echo "OUTPUT_DIR: $OUTPUT_DIR"

conda activate dust3r


#export LOCAL_RANK=$SLURM_LOCALID 
#export WORLD_SIZE=$SLURM_NTASKS 
#export RANK=$SLURM_PROCID
#export MASTER_ADDR=$SLURM_LAUNCH_NODE_IPADDR 
#export MASTER_PORT=6000


torchrun --nproc_per_node=$SLURM_GPUS_ON_NODE\
         --node_rank=$SLURM_NODEID\
         --master_addr=$SLURM_LAUNCH_NODE_IPADDR --master_port=6000 \
         --nnodes=$SLURM_NNODES train.py\
         --train_dataset "1000 @ Co3d(split='train', ROOT='data/co3d_subset_processed', aug_crop=16, mask_bg='rand', resolution=224, transform=ColorJitter)" \
         --test_dataset "100 @ Co3d(split='test', ROOT='data/co3d_subset_processed', resolution=224, seed=777)" \
         --model "AsymmetricCroCo3DStereo(pos_embed='RoPE100', img_size=(224, 224), head_type='linear', output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12)" \
         --train_criterion "ConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2)" \
         --test_criterion "Regr3D_ScaleShiftInv(L21, gt_scale=True)" \
         --pretrained "checkpoints/CroCo_V2_ViTLarge_BaseDecoder.pth" \
         --lr 0.0001 --min_lr 1e-06 --warmup_epochs 1 --epochs 10 --batch_size 16 --accum_iter 1 \
         --save_freq 1 --keep_freq 5 --eval_freq 1 \
         --output_dir "$OUTPUT_DIR"














