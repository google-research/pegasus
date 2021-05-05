help:
	@cat Makefile

IMAGE=google/pegasus
PROJECT=pegasus
DOCKER_FILE=Dockerfile
SRC?=$(shell dirname `pwd`)
DATA?="${HOME}/Data"

build:
	docker build -t $(IMAGE) -f $(DOCKER_FILE) .

bash: build
	docker run --init -it --gpus all --hostname $(PROJECT)-run -v $(SRC):/src/workspace -v $(DATA):/data --mount type=volume,source=$(PROJECT)-data,target=/src/ckpt/ --name $(PROJECT)-run $(IMAGE) bash

ipython: build
	docker run --init -it --gpus all --hostname $(PROJECT)-run -v $(SRC):/src/workspace -v $(DATA):/data --mount type=volume,source=$(PROJECT)-data,target=/src/ckpt/ --name $(PROJECT)-run $(IMAGE) ipython

notebook: build
	docker run --init -it --gpus all --hostname $(PROJECT)-run -v $(SRC):/src/workspace -v $(DATA):/data --mount type=volume,source=$(PROJECT)-data,target=/src/ckpt/ --net=host --name $(PROJECT)-run $(IMAGE) jupyter notebook --port=8888 --ip=0.0.0.0

test: build
	docker run --init -it --gpus all --hostname $(PROJECT)-run -v $(SRC):/src/workspace -v $(DATA):/data --mount type=volume,source=$(PROJECT)-data,target=/src/ckpt/ --name $(PROJECT) $(IMAGE) python3 pegasus/bin/train.py --params=aeslc_transformer --param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model --train_init_checkpoint=ckpt/pegasus_ckpt/model.ckpt-1500000 --model_dir=ckpt/pegasus_ckpt/aeslc
