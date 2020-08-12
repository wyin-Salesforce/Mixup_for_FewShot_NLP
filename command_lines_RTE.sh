export BETATIMES=15
export SHOT=3



CUDA_VISIBLE_DEVICES=0 python -u train_RTE.py \
    --task_name rte \
    --do_train \
    --do_lower_case \
    --num_train_epochs 100 \
    --data_dir '' \
    --output_dir '' \
    --train_batch_size 3 \
    --eval_batch_size 32 \
    --learning_rate 5e-6 \
    --max_seq_length 128 \
    --seed 42 \
    --kshot $SHOT \
    --beta_sampling_times 1 > log.seed.42.txt 2>&1 &

CUDA_VISIBLE_DEVICES=1 python -u train_RTE.py \
    --task_name rte \
    --do_train \
    --do_lower_case \
    --num_train_epochs 100 \
    --data_dir '' \
    --output_dir '' \
    --train_batch_size 3 \
    --eval_batch_size 32 \
    --learning_rate 5e-6 \
    --max_seq_length 128 \
    --seed 16 \
    --kshot $SHOT \
    --beta_sampling_times 1 > log.seed.16.txt 2>&1 &

CUDA_VISIBLE_DEVICES=2 python -u train_RTE.py \
    --task_name rte \
    --do_train \
    --do_lower_case \
    --num_train_epochs 100 \
    --data_dir '' \
    --output_dir '' \
    --train_batch_size 3 \
    --eval_batch_size 32 \
    --learning_rate 5e-6 \
    --max_seq_length 128 \
    --seed 32 \
    --kshot $SHOT \
    --beta_sampling_times 1 > log.seed.32.txt 2>&1 &

CUDA_VISIBLE_DEVICES=3 python -u train_RTE.py \
    --task_name rte \
    --do_train \
    --do_lower_case \
    --num_train_epochs 100 \
    --data_dir '' \
    --output_dir '' \
    --train_batch_size 3 \
    --eval_batch_size 32 \
    --learning_rate 5e-6 \
    --max_seq_length 128 \
    --seed 64 \
    --kshot $SHOT \
    --beta_sampling_times 1 > log.seed.64.txt 2>&1 &


CUDA_VISIBLE_DEVICES=4 python -u train_RTE.py \
    --task_name rte \
    --do_train \
    --do_lower_case \
    --num_train_epochs 100 \
    --data_dir '' \
    --output_dir '' \
    --train_batch_size 3 \
    --eval_batch_size 32 \
    --learning_rate 5e-6 \
    --max_seq_length 128 \
    --seed 128 \
    --kshot $SHOT \
    --beta_sampling_times 1 > log.seed.128.txt 2>&1 &