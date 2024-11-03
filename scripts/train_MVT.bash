export data_path=./IRef-VLA # PATH TO THE DATASET

export proj_name=mvt_train # PROJECT NAME FOR LOGGING

export train_split='train' # CHANGE THIS TO THE DESIRED TRAINING SPLIT
export test_split='test' # CHANGE THIS TO THE DESIRED TEST SPLIT

export LC_ALL=C.UTF-8
export LANG=C.UTF-8

export PYTHONPATH=..

cd MVT
conda activate irefvla
python train_MVT.py  \
    --batch_size 24 \
    --data_path $data_path \
    --train_split $train_split \
    --test_split $test_split \
    --proj_name $proj_name \
    --log_dir 'logs/MVT_sr3d' \
    --save_freq 5 \
    --val_freq 100 \
    --log_freq 10 \
    --use_sr3d \
    --dataset 'vla'  # CHANGE THIS TO THE DESIRED DATASET, either 'r3d', 'vla', or 'both'