# Using PEGASUS via Docker

This directory contains the `Dockerfile` to make it easy to get up and running with
PEGASUS via [Docker](http://www.docker.com/).

## Installing Docker

General installation instructions are
[on the Docker site](https://docs.docker.com/installation/).

## Installing GPU support

For GPU support install NVIDIA drivers (ideally the latest) and the [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker).

## Running the container

`Makefile` is used to simplify docker commands within `make` commands.

Build the container and start a bash

    $ make bash

Build the container and start an iPython shell

	$ make ipython

Build the container and start a Jupyter Notebook

	$ make notebook

Build the container and start the finetuning test described in the repository's README:

	$ make test

Mount a volume for external data sets

    $ make DATA=~/mydata

Prints all make tasks

    $ make help
