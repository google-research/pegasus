FROM tensorflow/tensorflow:1.15.2-gpu

ARG USER=pegasus
ARG USER_UID=1000

RUN apt-get update && apt-get install -y \
	wget \
	git \
	man \
	nano

ENV CONDA_PATH /opt/conda
ENV PATH ${CONDA_PATH}/bin:${PATH}

RUN wget --quiet --no-check-certificate https://repo.anaconda.com/miniconda/Miniconda3-py37_4.8.3-Linux-x86_64.sh \
    && echo "bb2e3cedd2e78a8bb6872ab3ab5b1266a90f8c7004a22d8dc2ea5effeb6a439a *Miniconda3-py37_4.8.3-Linux-x86_64.sh" | sha256sum -c - \
    && /bin/bash Miniconda3-py37_4.8.3-Linux-x86_64.sh -f -b -p ${CONDA_PATH} \
    && rm Miniconda3-py37_4.8.3-Linux-x86_64.sh \
    && echo export PATH=${CONDA_PATH}/bin:'${PATH}' > /etc/profile.d/conda.sh

RUN mkdir /src \
    && git clone https://github.com/google-research/pegasus /src/

RUN groupadd ${USER} && useradd -m -u ${USER_UID} -g ${USER} -s /bin/bash ${USER} \
    && chown -R ${USER_UID} /src \
    && chown -R ${USER_UID} ${CONDA_PATH}

USER ${USER}

# ENV PATH="/home/${USER}/.local/bin:$PATH"

ARG python_version=3.6

RUN conda config --append channels conda-forge
RUN conda install -y \
	python=${python_version} \
	notebook \
    && pip install --upgrade pip \
    && pip install \
	six==1.14.0 \
	protobuf==3.12.0 \
	cloudpickle==1.2.0 \
	gsutil \
    && pip install -r /src/requirements.txt

RUN mkdir /src/ckpt/

COPY docker_entrypoint.sh /src/

COPY bash.bashrc /etc/

WORKDIR /src

ENV PYTHONPATH="/src/:$PYTHONPATH"

ENTRYPOINT ["sh", "/src/docker_entrypoint.sh"]
CMD ["sh"]
