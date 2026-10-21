#!/usr/bin/env bash
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
export CMAKE_POLICY_VERSION_MINIMUM=3.5
export SETUPTOOLS_USE_DISTUTILS=stdlib
cd /content
add-apt-repository ppa:deadsnakes/ppa -y
apt-get update -y
apt-get install -y python3.10 python3.10-dev python3.10-venv python3.10-distutils libdbus-1-dev pkg-config build-essential cmake libsndfile1
python3.10 -m venv /content/tokine-202211
source /content/tokine-202211/bin/activate
python -m pip install pip==23.0.1 wheel==0.45.1 setuptools==65.5.0
python -m pip install numpy==1.23.5 Cython==0.29.32 cmake==3.22.1
git clone --depth 1 --branch 0.10.3 https://github.com/shirowanisan/espnet.git /content/tokine-espnet
test "$(git -C /content/tokine-espnet rev-parse HEAD)" = abab1e48c06eddb4bd3bc8762176317d12d2b192
git -C /content/tokine-espnet rev-parse HEAD
python -m pip install pyopenjtalk==0.1.3 pysptk==0.2.0 --no-build-isolation
cd /content/tokine-espnet
python -m pip install -r requirements-cuda113.txt
cd /content
python -c 'import torch, espnet; assert espnet.__version__ == "202211"; print("ENV_READY", torch.__version__, espnet.__version__, torch.cuda.is_available())'
