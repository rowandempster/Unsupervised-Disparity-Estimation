python3 train.py --exp test_exp \
                 --l1_weight 0.999 \
                 --smooth_weight 0.001 \
                 --per_weight 0 \
                 --per_layers 1 2 \
                 --agg sum \
                 --gpu 0 \
                 --batch_size 4 \
                 --epochs 5 \
                 --num_workers 1