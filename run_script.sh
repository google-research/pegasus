# This is for evaluation - change with different models
nohup python3 pegasus/bin/evaluate.py --params=cnn_dailymail_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,batch_size=1,beam_size=8,beam_alpha=0.8 --model_dir=ckpt/pegasus_ckpt/cnn_dailymail/model.ckpt-210000 --evaluate_test > trial_2.txt &

# This is for fine tuning - change with different models
python3 pegasus/bin/train.py --params=wikihow_all_transformer \
--param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,train_pattern=tfds:wikihow/all-train-take_1000 \
--train_init_checkpoint=ckpt/pegasus_ckpt/model.ckpt-1500000 \
--model_dir=ckpt/pegasus_ckpt/wikihow

nohup python3 pegasus/bin/train.py --params=cnn_dailymail_transformer \
--param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,train_pattern=tfds:cnn_dailymail/plain_text-train-take_1000 \
--train_init_checkpoint=ckpt/pegasus_ckpt/model.ckpt-1500000 \
--model_dir=ckpt/pegasus_ckpt/cnn_dailymail > trial_4.txt &

# Fine-tuning from Readme
python3 pegasus/bin/train.py --params=aeslc_transformer \
--param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model \
--train_init_checkpoint=ckpt/pegasus_ckpt/model.ckpt-1500000 \
--model_dir=ckpt/pegasus_ckpt/aeslc

# Should be able to modify this by including
# train_pattern=tfds:wikihow/all-train-take_1000 \
# Refer to pegasus/datasets.py and pegasus/public_params.py