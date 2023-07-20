#!/bin/bash

set -a
source .devcontainer/.aws_env
set +a

# poetry source add --default mirrors https://pypi.tuna.tsinghua.edu.cn/simple/ # tsinghua mirror
export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
poetry config virtualenvs.in-project true
poetry install
source .venv/bin/activate

# mkdir ~/.aws
# echo -e "[default]\nregion = us-east-1\noutput = json" > ~/.aws/config
# echo -e "[default]\naws_access_key_id = $aws_access_key_id\naws_secret_access_key = $aws_secret_access_key" > ~/.aws/credentials
# dvc pull
