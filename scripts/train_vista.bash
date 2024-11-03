export data_path=./IRef-VLA # PATH TO THE DATASET

export proj_name=vista_train # PROJECT NAME FOR LOGGING
export run_name=exp-01 # NAME OF CURRENT RUN
export model_ckpt= # PATH TO THE MODEL CHECKPOINT

export train_split='train' # CHANGE THIS TO THE DESIRED TRAINING SPLIT
export test_split='test' # CHANGE THIS TO THE DESIRED TEST SPLIT

export LC_ALL=C.UTF-8
export LANG=C.UTF-8

conda activate 3dvista
python -m vista.train_3dvista \
    --data_path $data_path \
    --run_name $run_name \
    --proj_name $proj_name \
    --train_split $train_split \
    --test_split $test_split \
    --resume_path $model_ckpt \
    --use_context \
    --context_size 100 \
    --batch_size 32 \
    --val_freq 1 \
    # --use_sr3d
