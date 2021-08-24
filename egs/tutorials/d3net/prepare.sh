#!/bin/bash

ID="18o9GUtBdSdAw4gsReH4FQ3UJfDLE8v8F"

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${ID}" > /dev/null
code="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"  
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${code}&id=${ID}" -o "model.zip"

unzip "model.zip"