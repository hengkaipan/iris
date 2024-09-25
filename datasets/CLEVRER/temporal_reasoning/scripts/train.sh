CUDA_VISIBLE_DEVICES=2 python train.py \
--gen_valid_idx 1 \
--data_dir /home/kathy/dev/proj/data/video_frames \
--label_dir /home/kathy/dev/proj/data/processed_proposals \
--resume_epoch 0 \
--resume_iter 0
