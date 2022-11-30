# PEGASUS library

Pre-training with Extracted Gap-sentences for Abstractive SUmmarization
Sequence-to-sequence models, or PEGASUS, uses self-supervised objective Gap
Sentences Generation (GSG) to train a transformer encoder-decoder model. The
paper can be found on [arXiv](https://arxiv.org/abs/1912.08777). ICML 2020 accepted.

If you use this code or these models, please cite the following paper:
```
@misc{zhang2019pegasus,
    title={PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization},
    author={Jingqing Zhang and Yao Zhao and Mohammad Saleh and Peter J. Liu},
    year={2019},
    eprint={1912.08777},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

# PEGASUS-X / Flax Implementation

**Update (2022/08)**: Go to [pegasus/flax](pegasus/flax) for PEGASUS-X models

# Results update

We train a pegasus model with sampled gap sentence ratios on both C4 and HugeNews, and stochastically sample important sentences. The updated the results are reported in this table.

| dataset | C4 | HugeNews | Mixed & Stochastic|
| ---- | ---- | ---- | ----|
| xsum | 45.20/22.06/36.99 | 47.21/24.56/39.25 | 47.60/24.83/39.64|
| cnn_dailymail | 43.90/21.20/40.76 | 44.17/21.47/41.11 | 44.16/21.56/41.30|
| newsroom | 45.07/33.39/41.28 | 45.15/33.51/41.33 | 45.98/34.20/42.18|
| multi_news | 46.74/17.95/24.26 | 47.52/18.72/24.91 | 47.65/18.75/24.95|
| gigaword | 38.75/19.96/36.14 | 39.12/19.86/36.24 | 39.65/20.47/36.76|
| wikihow | 43.07/19.70/34.79 | 41.35/18.51/33.42 | 46.39/22.12/38.41 *|
| reddit_tifu | 26.54/8.94/21.64 | 26.63/9.01/21.60 | 27.99/9.81/22.94|
| big_patent | 53.63/33.16/42.25 | 53.41/32.89/42.07 | 52.29/33.08/41.66 *|
| arxiv | 44.70/17.27/25.80 | 44.67/17.18/25.73 | 44.21/16.95/25.67|
| pubmed | 45.49/19.90/27.69 | 45.09/19.56/27.42 | 45.97/20.15/28.25|
| aeslc | 37.69/21.85/36.84 | 37.40/21.22/36.45 | 37.68/21.25/36.51|
| billsum | 57.20/39.56/45.80 | 57.31/40.19/45.82 | 59.67/41.58/47.59|

The "Mixed & Stochastic" model has the following changes:
- trained on both C4 and HugeNews (dataset mixture is weighted by their number of examples). 
- trained for 1.5M instead of 500k (we observe slower convergence on pretraining perplexity).
- the model uniformly sample a gap sentence ratio between 15% and 45%.
- importance sentences are sampled using a 20% uniform noise to importance scores.
- the sentencepiece tokenizer is updated to be able to encode newline character.


(*) the numbers of wikihow and big_patent datasets are not comparable because of change in tokenization and data:
- wikihow dataset contains newline characters which is useful for paragraph segmentation, the C4 and HugeNews model's sentencepiece tokenizer doesn't encode newline and loose this information.
- we update the BigPatent dataset to preserve casing, some format cleanings are also changed, please refer to change in TFDS.


# Setup

## create an instance on google cloud with GPU (optional)

Please create a project first and create an instance

```
gcloud compute instances create \
  ${VM_NAME} \
  --zone=${ZONE} \
  --machine-type=n1-highmem-8 \
  --accelerator type=nvidia-tesla-v100,count=1 \
  --boot-disk-size=500GB \
  --image-project=ml-images \
  --image-family=tf-1-15 \
  --maintenance-policy TERMINATE --restart-on-failure
```

## install library and dependencies

Clone library on github and install requirements.

```
git clone https://github.com/google-research/pegasus
cd pegasus
export PYTHONPATH=.
pip3 install -r requirements.txt
```

Download vocab, pretrained and fine-tuned checkpoints of all experiments from [Google Cloud](https://console.cloud.google.com/storage/browser/pegasus_ckpt).

Alternatively in terminal, follow the instruction and install [gsutil](https://cloud.google.com/storage/docs/gsutil_install). Then

```
mkdir ckpt
gsutil cp -r gs://pegasus_ckpt/ ckpt/

```

# Finetuning on downstream datasets

## on existing dataset

Finetune on an existing dataset `aeslc`.

```
python3 pegasus/bin/train.py --params=aeslc_transformer \
--param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model \
--train_init_checkpoint=ckpt/pegasus_ckpt/model.ckpt-1500000 \
--model_dir=ckpt/pegasus_ckpt/aeslc
```

If you would like to finetune on a subset of dataset, please refer to the [example of input pattern](https://github.com/google-research/pegasus/blob/master/pegasus/data/datasets.py#L186).

Evaluate on the finetuned dataset.

```
python3 pegasus/bin/evaluate.py --params=aeslc_transformer \
--param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,batch_size=1,beam_size=5,beam_alpha=0.6 \
--model_dir=ckpt/pegasus_ckpt/aeslc
```

Note that the above example is using a single GPU so the batch_size is much smaller
than the results reported in the paper.

## add new finetuning dataset

Two types of dataset format are supported: [TensorFlow Datasets (TFDS)](https://www.tensorflow.org/datasets) or TFRecords.

[This tutorial](https://www.tensorflow.org/datasets/add_dataset) shows how to add a new dataset in TFDS.
(The fine-tuning dataset is expected to be supervised, please provide
`supervised_keys` in dataset info).

Tfrecords format requires each record to be a tf example of `{"inputs":tf.string, "targets":tf.string}`.

For example, if you registered a TFDS dataset called `new_tfds_dataset` for training and evaluation, and have some files in tfrecord format called `new_dataset_files.tfrecord*` for test, they can be registered in `/pegasus/params/public_params.py`.

```
@registry.register("new_params")
def my_param(param_overrides):
  return public_params.transformer_params(
      {
          "train_pattern": "tfds:new_tfds_dataset,train",
          "dev_pattern": "tfds:new_tfds_dataset,validation",
          "test_pattern": "tfrecord:new_dataset_files.tfrecord*",
          "max_input_len": 512,
          "max_output_len": 128,
          "train_steps": 10000,
          "learning_rate": 0.0001,
          "batch_size": 8,
      }, param_overrides)
```

## Evaluation metrics.

Evaluation results can be found in `mode_dir`. Summarization metrics are automatically
calculated for each evaluation point.

-   [ROUGE](https://www.aclweb.org/anthology/W04-1013.pdf) is the main metric
    for summarization quality.

-   [BLEU](https://www.aclweb.org/anthology/P02-1040.pdf) is an alternative
    quality metric for language generation.

-   [Extractive Fragments Coverage & Density](https://arxiv.org/pdf/1804.11283.pdf)
    are metrics that measures the abstractiveness of the summary.


-   Repetition Rates measures generation repetition failure modes.

-   Length statistics measures the length distribution of decodes comparing to gold summary.

Several types of output files can be found in `model_dir`

-   text_metrics-*.txt: above metrics in text format. Each row contains metric
    name, 95% lower bound value, mean value, 95% upper bound value.
-   inputs-*.txt, targets-*.txt, predictions-*.txt: raw text files of model
    inputs/outputs.


# Pre-training

Pretraining (on C4 or any other corpus) requires a customly built tensorflow that includes ops for on-the-fly parsing that processes raw text document into model inputs and targets ids. Please refer to pegasus/ops/pretrain_parsing_ops.cc and pegasus/data/parsers.py for details.

# Acknowledgements
Contains parts of code and design for training and evaluation of summarization models originally by Ben Goodrich <bgoodrich@google.com>.


