#!/bin/sh

if [ ! -d /src/ckpt/pegasus_ckpt/ ]
then
    gsutil cp -r gs://pegasus_ckpt /src/ckpt/
fi

exec "$@"
