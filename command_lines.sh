export DOMAIN="work"
export BETATIMES=15
export SHOT=5



# CUDA_VISIBLE_DEVICES=0 python -u train_CLINC150.py \
#     --task_name rte \
#     --do_train \
#     --do_lower_case \
#     --num_train_epochs 50 \
#     --data_dir '' \
#     --output_dir '' \
#     --train_batch_size 5 \
#     --eval_batch_size 32 \
#     --learning_rate 5e-6 \
#     --max_seq_length 20 \
#     --seed 42 \
#     --DomainName "$DOMAIN" \
#     --kshot $SHOT \
#     --beta_sampling_times 1 > log.seed.42.txt 2>&1 &
#
# CUDA_VISIBLE_DEVICES=1 python -u train_CLINC150.py \
#     --task_name rte \
#     --do_train \
#     --do_lower_case \
#     --num_train_epochs 50 \
#     --data_dir '' \
#     --output_dir '' \
#     --train_batch_size 5 \
#     --eval_batch_size 32 \
#     --learning_rate 5e-6 \
#     --max_seq_length 20 \
#     --seed 16 \
#     --DomainName "$DOMAIN" \
#     --kshot $SHOT \
#     --beta_sampling_times 1 > log.seed.16.txt 2>&1 &
#
# CUDA_VISIBLE_DEVICES=2 python -u train_CLINC150.py \
#     --task_name rte \
#     --do_train \
#     --do_lower_case \
#     --num_train_epochs 50 \
#     --data_dir '' \
#     --output_dir '' \
#     --train_batch_size 5 \
#     --eval_batch_size 32 \
#     --learning_rate 5e-6 \
#     --max_seq_length 20 \
#     --seed 32 \
#     --DomainName "$DOMAIN" \
#     --kshot $SHOT \
#     --beta_sampling_times 1 > log.seed.32.txt 2>&1 &
#
# CUDA_VISIBLE_DEVICES=3 python -u train_CLINC150.py \
#     --task_name rte \
#     --do_train \
#     --do_lower_case \
#     --num_train_epochs 50 \
#     --data_dir '' \
#     --output_dir '' \
#     --train_batch_size 5 \
#     --eval_batch_size 32 \
#     --learning_rate 5e-6 \
#     --max_seq_length 20 \
#     --seed 64 \
#     --DomainName "$DOMAIN" \
#     --kshot $SHOT \
#     --beta_sampling_times 1 > log.seed.64.txt 2>&1 &
#
# CUDA_VISIBLE_DEVICES=4 python -u train_CLINC150.py \
#     --task_name rte \
#     --do_train \
#     --do_lower_case \
#     --num_train_epochs 50 \
#     --data_dir '' \
#     --output_dir '' \
#     --train_batch_size 5 \
#     --eval_batch_size 32 \
#     --learning_rate 5e-6 \
#     --max_seq_length 20 \
#     --seed 128 \
#     --DomainName "$DOMAIN" \
#     --kshot $SHOT \
#     --beta_sampling_times 1 > log.seed.128.txt 2>&1 &

#w/ mixup
CUDA_VISIBLE_DEVICES=0 python -u train_CLINC150.py \
    --task_name rte \
    --do_train \
    --do_lower_case \
    --num_train_epochs 50 \
    --data_dir '' \
    --output_dir '' \
    --train_batch_size 5 \
    --eval_batch_size 32 \
    --learning_rate 1e-6 \
    --max_seq_length 20 \
    --seed 42 \
    --DomainName "$DOMAIN" \
    --kshot $SHOT \
    --use_mixup\
    --beta_sampling_times $BETATIMES > log."$DOMAIN".mixup.42.txt 2>&1 &

CUDA_VISIBLE_DEVICES=1 python -u train_CLINC150.py \
    --task_name rte \
    --do_train \
    --do_lower_case \
    --num_train_epochs 50 \
    --data_dir '' \
    --output_dir '' \
    --train_batch_size 5 \
    --eval_batch_size 32 \
    --learning_rate 1e-6 \
    --max_seq_length 20 \
    --seed 16 \
    --DomainName "$DOMAIN" \
    --kshot $SHOT \
    --use_mixup\
    --beta_sampling_times $BETATIMES > log."$DOMAIN".mixup.16.txt 2>&1 &

CUDA_VISIBLE_DEVICES=2 python -u train_CLINC150.py \
    --task_name rte \
    --do_train \
    --do_lower_case \
    --num_train_epochs 50 \
    --data_dir '' \
    --output_dir '' \
    --train_batch_size 5 \
    --eval_batch_size 32 \
    --learning_rate 1e-6 \
    --max_seq_length 20 \
    --seed 32 \
    --DomainName "$DOMAIN" \
    --kshot $SHOT \
    --use_mixup\
    --beta_sampling_times $BETATIMES > log."$DOMAIN".mixup.32.txt 2>&1 &

CUDA_VISIBLE_DEVICES=3 python -u train_CLINC150.py \
    --task_name rte \
    --do_train \
    --do_lower_case \
    --num_train_epochs 50 \
    --data_dir '' \
    --output_dir '' \
    --train_batch_size 5 \
    --eval_batch_size 32 \
    --learning_rate 1e-6 \
    --max_seq_length 20 \
    --seed 64 \
    --DomainName "$DOMAIN" \
    --kshot $SHOT \
    --use_mixup\
    --beta_sampling_times $BETATIMES > log."$DOMAIN".mixup.64.txt 2>&1 &

CUDA_VISIBLE_DEVICES=4 python -u train_CLINC150.py \
    --task_name rte \
    --do_train \
    --do_lower_case \
    --num_train_epochs 50 \
    --data_dir '' \
    --output_dir '' \
    --train_batch_size 5 \
    --eval_batch_size 32 \
    --learning_rate 1e-6 \
    --max_seq_length 20 \
    --seed 128 \
    --DomainName "$DOMAIN" \
    --kshot $SHOT \
    --use_mixup\
    --beta_sampling_times $BETATIMES > log."$DOMAIN".mixup.128.txt 2>&1 &
