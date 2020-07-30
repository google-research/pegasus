#!/bin/bash
# Fine-tune on 1000 examples with original loss
nohup python3 pegasus/bin/train.py --params=aeslc_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,train_pattern=tfds:aeslc-train-take_1000,batch_size=1 --train_init_checkpoint=ckpt/pegasus_ckpt/model.ckpt-1500000 --model_dir=ckpt/pegasus_ckpt/aeslc > aeslc_trial_3.txt
# Evaluate on 1000 examples with original loss
nohup python3 pegasus/bin/evaluate.py --params=aeslc_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,batch_size=1,beam_size=8,beam_alpha=0.8 --model_dir=ckpt/pegasus_ckpt/aeslc/model.ckpt-210000 --evaluate_test > aeslc_eval_3.txt
# Copy results to new folder
cp ckpt/pegasus_ckpt/aeslc ckpt/pegasus_ckpt/aeslc_old
rm -rf ckpt/pegasus_ckpt/aeslc
# Load new aeslc
gsutil -m cp -r gs://pegasus_ckpt/aeslc/ ckpt/pegasus_ckpt/
# Fine-tune on 1000 examples with new loss
nohup python3 pegasus/bin/train.py --params=aeslc_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,train_pattern=tfds:aeslc-train-take_1000,batch_size=1 --train_init_checkpoint=ckpt/pegasus_ckpt/model.ckpt-1500000 --model_dir=ckpt/pegasus_ckpt/aeslc > aeslc_trial_4.txt
# Evaluate on 1000 examples with new loss
nohup python3 pegasus/bin/evaluate.py --params=aeslc_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,batch_size=1,beam_size=8,beam_alpha=0.8 --model_dir=ckpt/pegasus_ckpt/aeslc/model.ckpt-210000 --evaluate_test > aeslc_eval_4.txt
