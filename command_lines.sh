export GLUE_DIR=datapath
export TASK_NAME=SST-2

CUDA_VISIBLE_DEVICES=0 python -u train_CLINC150.py \
    --task_name rte \
    --do_train \
    --do_lower_case \
    --num_train_epochs 50 \
    --data_dir '' \
    --output_dir '' \
    --train_batch_size 5 \
    --eval_batch_size 32 \
    --learning_rate 5e-6 \
    --max_seq_length 20 \
    --seed 42 \
    --DomainName 'banking' \
    --kshot 3 \
    --beta_sampling_times 1 > log.seed.42.txt 2>&1

CUDA_VISIBLE_DEVICES=1 python -u train_CLINC150.py \
    --task_name rte \
    --do_train \
    --do_lower_case \
    --num_train_epochs 50 \
    --data_dir '' \
    --output_dir '' \
    --train_batch_size 5 \
    --eval_batch_size 32 \
    --learning_rate 5e-6 \
    --max_seq_length 20 \
    --seed 16 \
    --DomainName 'banking' \
    --kshot 3 \
    --beta_sampling_times 1 > log.seed.16.txt 2>&1

CUDA_VISIBLE_DEVICES=2 python -u train_CLINC150.py \
    --task_name rte \
    --do_train \
    --do_lower_case \
    --num_train_epochs 50 \
    --data_dir '' \
    --output_dir '' \
    --train_batch_size 5 \
    --eval_batch_size 32 \
    --learning_rate 5e-6 \
    --max_seq_length 20 \
    --seed 32 \
    --DomainName 'banking' \
    --kshot 3 \
    --beta_sampling_times 1 > log.seed.32.txt 2>&1

CUDA_VISIBLE_DEVICES=3 python -u train_CLINC150.py \
    --task_name rte \
    --do_train \
    --do_lower_case \
    --num_train_epochs 50 \
    --data_dir '' \
    --output_dir '' \
    --train_batch_size 5 \
    --eval_batch_size 32 \
    --learning_rate 5e-6 \
    --max_seq_length 20 \
    --seed 64 \
    --DomainName 'banking' \
    --kshot 3 \
    --beta_sampling_times 1 > log.seed.64.txt 2>&1

CUDA_VISIBLE_DEVICES=4 python -u train_CLINC150.py \
    --task_name rte \
    --do_train \
    --do_lower_case \
    --num_train_epochs 50 \
    --data_dir '' \
    --output_dir '' \
    --train_batch_size 5 \
    --eval_batch_size 32 \
    --learning_rate 5e-6 \
    --max_seq_length 20 \
    --seed 128 \
    --DomainName 'banking' \
    --kshot 3 \
    --beta_sampling_times 1 > log.seed.128.txt 2>&1