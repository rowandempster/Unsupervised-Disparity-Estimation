python3 train.py --exp mix_half \
                 --l1_weight 0.4995 \
                 --smooth_weight 0.001 \
                 --per_weight 0.4995 \
                 --per_layers 1 \
                 --agg prod \
                 --gpu 1 \
                 --batch_size 2 \
                 --epochs 4\
                 --num_workers 8

python3 train.py --exp mix_l1 \
                 --l1_weight 0.8991 \
                 --smooth_weight 0.001 \
                 --per_weight 0.0999 \
                 --per_layers 1 \
                 --agg prod \
                 --gpu 1 \
                 --batch_size 2 \
                 --epochs 4\
                 --num_workers 8

python3 train.py --exp mix_per \
                 --l1_weight 0.0999 \
                 --smooth_weight 0.001 \
                 --per_weight 0.8991 \
                 --per_layers 1 \
                 --agg prod \
                 --gpu 1 \
                 --batch_size 2 \
                 --epochs 4\
                 --num_workers 8