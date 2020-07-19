FROM tensorflow/tensorflow:1.15.2-gpu

ARG USER=pegasus
ARG USER_UID=1000

RUN apt-get update && apt-get install -y \
	git \
	man \
	nano

RUN mkdir /src \
    && git clone https://github.com/google-research/pegasus /src/

RUN groupadd ${USER} && useradd -m -u ${USER_UID} -g ${USER} -s /bin/bash ${USER} \
    && chown -R ${USER} /src

USER ${USER}

ENV PATH="/home/${USER}/.local/bin:$PATH"

RUN pip3 install --user --upgrade pip \
    && pip3 install --user six==1.14.0 protobuf==3.12.0 cloudpickle==1.2.0 gsutil jupyter \
    && pip3 install --user -r /src/requirements.txt

RUN mkdir /src/ckpt/

COPY docker_entrypoint.sh /src/

COPY bash.bashrc /etc/

WORKDIR /src

ENV PYTHONPATH="/src/:$PYTHONPATH"

ENTRYPOINT ["sh", "/src/docker_entrypoint.sh"]
CMD ["sh"]
