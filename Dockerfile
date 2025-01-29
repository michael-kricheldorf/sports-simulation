ARG PYTORCH="2.1.0"
ARG CUDA="12.1"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

# To fix GPG key error when running apt-get update
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN ln -fs /usr/share/zoneinfo/America/New_York /etc/localtime

RUN apt-get update 
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install tzdata
RUN apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx libgtk2.0-dev pkg-config wget\
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install xtcocotools
RUN pip install cython
RUN pip install xtcocotools

# Install MMEngine and MMCV
RUN pip install openmim
RUN mim install mmengine "mmcv==2.1.0"

# Install MMPose
RUN conda clean --all
ARG CACHEBUST=1
RUN git clone https://github.com/michael-kricheldorf/mmpose.git /mmpose
RUN git clone https://github.com/michael-kricheldorf/roboflow-sports.git /sports
RUN pip install /sports
WORKDIR /mmpose
RUN git checkout main
ENV FORCE_CUDA="1"
#ENV MMCV_WITH_OPS=1
RUN pip install "numpy<2"
RUN pip install "torch>=1.8"
RUN pip install json-tricks
RUN pip install mmdet

# inserted due to an error, from https://stackoverflow.com/questions/67120450/error-2unspecified-error-the-function-is-not-implemented-rebuild-the-libra

RUN pip install --no-cache-dir -e .

RUN pip uninstall opencv-python -y

RUN pip install ultralytics
RUN pip install gdown

WORKDIR /sports-simulation

# commands to download all the pytorch files / models
# sourced from https://chemicloud.com/blog/download-google-drive-files-using-wget/
RUN mkdir models && cd models
# football-ball-detection
RUN wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1usHpmtC5eYd0LScISVOEA9CveBQn9u4t' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1usHpmtC5eYd0LScISVOEA9CveBQn9u4t" -O football-ball-detection.pt && rm -rf /tmp/cookies.txt
# football-pitch-detecton.pt
RUN wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1cmJzXldUTkXxGcW9lCo34OGu7ws6-T7l' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1cmJzXldUTkXxGcW9lCo34OGu7ws6-T7l" -O football-pitch-detection.pt && rm -rf /tmp/cookies.txt
# football-player-detection.pt
RUN wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1k9jr2mHHz1LSGbK1GynyzIhlbwUhmjNZ' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1k9jr2mHHz1LSGbK1GynyzIhlbwUhmjNZ" -O football-player-detection.pt && rm -rf /tmp/cookies.txt
# rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth 
RUN wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1a0CURtF1BfnNjcOoZoi6ukcz9Wmf02tp' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1a0CURtF1BfnNjcOoZoi6ukcz9Wmf02tp" -O rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth && rm -rf /tmp/cookies.txt
# rtmw3d-l_8xb64_cocktail14-384x288-794dbc78_20240626.pth
RUN wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=10F_9FPYokoMRijvbeH4JfD3lXmc4Eytf' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=10F_9FPYokoMRijvbeH4JfD3lXmc4Eytf" -O rtmw3d-l_8xb64_cocktail14-384x288-794dbc78_20240626.pth && rm -rf /tmp/cookies.txt
# rtmw3d-l_8xb64_cocktail14-384x288.py
RUN wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=13WiZCbIm_twBxeRrgmxJaOfQs1QJ10XS' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=13WiZCbIm_twBxeRrgmxJaOfQs1QJ10XS" -O rtmw3d-l_8xb64_cocktail14-384x288.py && rm -rf /tmp/cookies.txt

# commands to download all the sample footage
RUN cd .. && mkdir videos && cd videos

# download videos using the same style as above
# I didn't implement this because it's all soccer sample video footage
# but if you want to do this with football, just use the above format and 
# follow the linked guide on how to download big stuff from Google Drive