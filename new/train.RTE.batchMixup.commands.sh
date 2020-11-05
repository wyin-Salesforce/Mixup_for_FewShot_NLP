export BETATIMES=15
export BATCHMIXTIMES=400
export SHOT=0
export BATCHSIZE=5
export EPOCHSIZE=40 #need only 10 for fine-tune

CUDA_VISIBLE_DEVICES=0 python -u train.RTE.batchMixup.py \
    --task_name rte \
    --do_train \
    --do_lower_case \
    --num_train_epochs $EPOCHSIZE \
    --train_batch_size $BATCHSIZE \
    --eval_batch_size 32 \
    --learning_rate 1e-6 \
    --max_seq_length 128 \
    --seed 42 \
    --kshot $SHOT \
    --batch_mix_times $BATCHMIXTIMES \
    --beta_sampling_times $BETATIMES > log.RTE.batchMixup.42.txt 2>&1 &

CUDA_VISIBLE_DEVICES=1 python -u train.RTE.batchMixup.py \
    --task_name rte \
    --do_train \
    --do_lower_case \
    --num_train_epochs $EPOCHSIZE \
    --train_batch_size $BATCHSIZE \
    --eval_batch_size 32 \
    --learning_rate 1e-6 \
    --max_seq_length 128 \
    --seed 16 \
    --kshot $SHOT \
    --batch_mix_times $BATCHMIXTIMES \
    --beta_sampling_times $BETATIMES > log.RTE.batchMixup.16.txt 2>&1 &

CUDA_VISIBLE_DEVICES=2 python -u train.RTE.batchMixup.py \
    --task_name rte \
    --do_train \
    --do_lower_case \
    --num_train_epochs $EPOCHSIZE \
    --train_batch_size $BATCHSIZE \
    --eval_batch_size 32 \
    --learning_rate 1e-6 \
    --max_seq_length 128 \
    --seed 32 \
    --kshot $SHOT \
    --batch_mix_times $BATCHMIXTIMES \
    --beta_sampling_times $BETATIMES > log.RTE.batchMixup.32.txt 2>&1 &

CUDA_VISIBLE_DEVICES=3 python -u train.RTE.batchMixup.py \
    --task_name rte \
    --do_train \
    --do_lower_case \
    --num_train_epochs $EPOCHSIZE \
    --train_batch_size $BATCHSIZE \
    --eval_batch_size 32 \
    --learning_rate 1e-6 \
    --max_seq_length 128 \
    --seed 64 \
    --kshot $SHOT \
    --batch_mix_times $BATCHMIXTIMES \
    --beta_sampling_times $BETATIMES > log.RTE.batchMixup.64.txt 2>&1 &

CUDA_VISIBLE_DEVICES=4 python -u train.RTE.batchMixup.py \
    --task_name rte \
    --do_train \
    --do_lower_case \
    --num_train_epochs $EPOCHSIZE \
    --train_batch_size $BATCHSIZE \
    --eval_batch_size 32 \
    --learning_rate 1e-6 \
    --max_seq_length 128 \
    --seed 128 \
    --kshot $SHOT \
    --batch_mix_times $BATCHMIXTIMES \
    --beta_sampling_times $BETATIMES > log.RTE.batchMixup.128.txt 2>&1 &
