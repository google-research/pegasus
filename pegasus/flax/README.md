# PEGASUS and PEGASUS-X in JAX/Flax

## Overview

This folder contains code for the Flax implementation of the PEGASUS model, as well as the PEGASUS-X model that is adapted for long input summarization.

The PEGASUS-X model is described in [Investigating Efficiently Extending Transformers for Long Input Summarization](https://arxiv.org/abs/2208.04347).

## Preparation

### Model Conversion from TF Checkpoints

To convert PEGASUS TensorFlow checkpoints for use with the Flax code, use the script 
[convert_from_pegasus_to_flax](checkpoint_conversion/convert_from_pegasus_to_flax.py).

### Model Conversion between encoder architectures

### Tokenizer

You will also need to download the tokenizer file from [here](https://storage.googleapis.com/pegasus_ckpt/c4.unigram.newline.10pct.96000.model).

### Checkpoints

The full set of available checkpoints can be found in the [PEGASUS GCS Bucket](https://console.cloud.google.com/storage/browser/pegasus_ckpt/).

We highlight the notable PEGASUS-X-specific checkpoints here:

* PEGASUS-X-base (272M):
  * [PEGASUS-X-base (adapted but not fine-tuned)](https://storage.googleapis.com/pegasus_ckpt/px/untuned/base/checkpoint_1800000)
  * [Fine-tuned on arXiv](https://storage.googleapis.com/pegasus_ckpt/px/tuned/base/arxiv_beam2_alpha1.ckpt)
  * [Fine-tuned on Big Patent](https://storage.googleapis.com/pegasus_ckpt/px/tuned/base/bigpatent.ckpt)
  * [Fine-tuned on PubMed](https://storage.googleapis.com/pegasus_ckpt/px/tuned/base/pubmed.ckpt)
  * [Fine-tuned on GovReport (SCROLLS)](https://storage.googleapis.com/pegasus_ckpt/px/tuned/base/scrolls_govreport.ckpt)
  * [Fine-tuned on QMSum (SCROLLS)](https://storage.googleapis.com/pegasus_ckpt/px/tuned/base/scrolls_qmsum.ckpt)
  * [Fine-tuned on SummScreen/FD (SCROLLS)](https://storage.googleapis.com/pegasus_ckpt/px/tuned/base/scrolls_summscreen.ckpt)
* PEGASUS-X (568M):
  * [PEGASUS-X (adapted but not fine-tuned)](https://storage.googleapis.com/pegasus_ckpt/px/untuned/large/checkpoint_1800000)
  * [Fine-tuned on arXiv](https://storage.googleapis.com/pegasus_ckpt/px/tuned/large/arxiv_beam2_alpha1.ckpt)
  * [Fine-tuned on Big Patent](https://storage.googleapis.com/pegasus_ckpt/px/tuned/large/bigpatent.ckpt)
  * [Fine-tuned on PubMed](https://storage.googleapis.com/pegasus_ckpt/px/tuned/large/pubmed.ckpt)
  * [Fine-tuned on GovReport (SCROLLS)](https://storage.googleapis.com/pegasus_ckpt/px/tuned/large/scrolls_govreport.ckpt)
  * [Fine-tuned on QMSum (SCROLLS)](https://storage.googleapis.com/pegasus_ckpt/px/tuned/large/scrolls_qmsum.ckpt)
  * [Fine-tuned on SummScreen/FD (SCROLLS)](https://storage.googleapis.com/pegasus_ckpt/px/tuned/large/scrolls_summscreen.ckpt)

## Fine-tuning

### Data

First, we need to prepare the data for fine-tuning using TFDS. 

For example, this is the command for downloading and preparing the SCROLLS SummScreen/FD dataset. 

```bash
python -m tensorflow_datasets.scripts.download_and_prepare --datasets=scrolls/summ_screen_fd
```

### Fine-tuning a model

To fine-tune the model, you will need modify a config file, and then supply it to the run command.
A sample is provided [here](configs/examples/summscreen_eval.py).

Take note of the following arguments:

* `run_mode`: Set to `train`
* `tokenizer_path`: Path to the tokenizer model file.
* `dataset_name`: Set to the dataset you want to evaluate on.
* `checkpoint_dir`: Folder pointing to where pretrained checkpoints are stored
* `load_checkpoint_step`: Checkpoint step to load on (Optional, default=`-1`). If you are evaluating a downloaded checkpoint, you can point directly to it with the default=`-1` and it will load correctly.
* `overwrite_train_steps`: Set starting training step to 0
* `learning_rate`: Learning rate
* `per_device_batch_size`: Batch size per device
* `checkpoint_every_steps`: How often to save a checkpoint
* `max_input_length`: Maximum input context length for the model
* `max_target_length`: Maximum decode length for the model

Here is a sample command (remember to modify the config file):

```bash
python pegasus/flax/main.py \
    --config pegasus/flax/configs/examples/summscreen_eval.py \
    --workdir path/to/fine_tuning_output
```

### Evaluating the model

To evaluate a model, like with fine-tuning, you will need modify a config file accordingly.
A sample is provided [here](configs/examples/summscreen_eval.py).

Take note of the following arguments:

* `run_mode`: Set to `eval_only`
* `tokenizer_path`: Path to the tokenizer model file.
* `dataset_name`: Set to the dataset you want to evaluate on.
* `checkpoint_dir`: Folder pointing to where fine-tuned checkpoints are stored
* `eval_step`: Checkpoint step to evaluate on. If you are evaluating a downloaded checkpoint, you can rename it to `checkpoint_0` and set this to `0`
* `per_device_batch_size`: Batch size per device
* `beam_size`: Beam search size
* `beam_alpha`: Beam search alpha
* `max_input_length`: Maximum input context length for the model
* `max_target_length`: Maximum decode length for the model

Here is a sample command (remember to modify the config file):

```bash
python pegasus/flax/main.py \
    --config pegasus/flax/configs/examples/summscreen_eval.py \
    --workdir path/to/eval_output
```

## Citation

```
@misc{phang2022pegasusx,
    title={Investigating Efficiently Extending Transformers for Long Input Summarization},
    author={Jason Phang and Yao Zhao and Peter J. Liu},
    year={2022},
    eprint={2208.04347},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
