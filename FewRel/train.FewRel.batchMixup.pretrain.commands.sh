export SHOT=0.75
export BATCHSIZE=32
export EPOCHSIZE=8
export LEARNINGRATE=1e-5
export MAXLEN=64
export BATCHMIXTIMES=2400


CUDA_VISIBLE_DEVICES=4 python -u train.FewRel.batchMixup.pretrain.py \
    --task_name rte \
    --do_train \
    --do_lower_case \
    --num_train_epochs $EPOCHSIZE \
    --train_batch_size $BATCHSIZE \
    --eval_batch_size 128 \
    --learning_rate $LEARNINGRATE \
    --max_seq_length $MAXLEN \
    --seed 42 \
    --kshot $SHOT \
    --use_mixup\
    --batch_mix_times $BATCHMIXTIMES  > log.FewRel.batchMixup.pretrain.$SHOT.shot.seed.42.txt 2>&1 &

CUDA_VISIBLE_DEVICES=5 python -u train.FewRel.batchMixup.pretrain.py \
    --task_name rte \
    --do_train \
    --do_lower_case \
    --num_train_epochs $EPOCHSIZE \
    --train_batch_size $BATCHSIZE \
    --eval_batch_size 128 \
    --learning_rate $LEARNINGRATE \
    --max_seq_length $MAXLEN \
    --seed 16 \
    --kshot $SHOT \
    --use_mixup\
    --batch_mix_times $BATCHMIXTIMES > log.FewRel.batchMixup.pretrain.$SHOT.shot.seed.16.txt 2>&1 &

CUDA_VISIBLE_DEVICES=6 python -u train.FewRel.batchMixup.pretrain.py \
    --task_name rte \
    --do_train \
    --do_lower_case \
    --num_train_epochs $EPOCHSIZE \
    --train_batch_size $BATCHSIZE \
    --eval_batch_size 128 \
    --learning_rate $LEARNINGRATE \
    --max_seq_length $MAXLEN \
    --seed 32 \
    --kshot $SHOT \
    --use_mixup\
    --batch_mix_times $BATCHMIXTIMES > log.FewRel.batchMixup.pretrain.$SHOT.shot.seed.32.txt 2>&1 &

CUDA_VISIBLE_DEVICES=7 python -u train.FewRel.batchMixup.pretrain.py \
    --task_name rte \
    --do_train \
    --do_lower_case \
    --num_train_epochs $EPOCHSIZE \
    --train_batch_size $BATCHSIZE \
    --eval_batch_size 128 \
    --learning_rate $LEARNINGRATE \
    --max_seq_length $MAXLEN \
    --seed 64 \
    --kshot $SHOT \
    --use_mixup\
    --batch_mix_times $BATCHMIXTIMES > log.FewRel.batchMixup.pretrain.$SHOT.shot.seed.64.txt 2>&1 &
