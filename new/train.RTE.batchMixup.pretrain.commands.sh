export BATCHMIXTIMES=400
export SHOT=0.50
export BATCHSIZE=5
export EPOCHSIZE=40 #20 batchmixup, 20 for standard mixup

CUDA_VISIBLE_DEVICES=0 python -u train.RTE.batchMixup.pretrain.py \
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
    --use_mixup\
    --batch_mix_times $BATCHMIXTIMES > log.RTE.batchMixup.pretrain.$SHOT.shot.42.txt 2>&1 &

CUDA_VISIBLE_DEVICES=1 python -u train.RTE.batchMixup.pretrain.py \
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
    --use_mixup\
    --batch_mix_times $BATCHMIXTIMES > log.RTE.batchMixup.pretrain.$SHOT.shot.16.txt 2>&1 &

CUDA_VISIBLE_DEVICES=2 python -u train.RTE.batchMixup.pretrain.py \
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
    --use_mixup\
    --batch_mix_times $BATCHMIXTIMES > log.RTE.batchMixup.pretrain.$SHOT.shot.32.txt 2>&1 &

CUDA_VISIBLE_DEVICES=3 python -u train.RTE.batchMixup.pretrain.py \
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
    --use_mixup\
    --batch_mix_times $BATCHMIXTIMES > log.RTE.batchMixup.pretrain.$SHOT.shot.64.txt 2>&1 &

# CUDA_VISIBLE_DEVICES=4 python -u train.RTE.batchMixup.pretrain.py \
#     --task_name rte \
#     --do_train \
#     --do_lower_case \
#     --num_train_epochs $EPOCHSIZE \
#     --train_batch_size $BATCHSIZE \
#     --eval_batch_size 32 \
#     --learning_rate 1e-6 \
#     --max_seq_length 128 \
#     --seed 128 \
    # --kshot $SHOT \
    # --use_mixup\
    # --batch_mix_times $BATCHMIXTIMES > log.RTE.batchMixup.128.txt 2>&1 &
