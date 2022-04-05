FROM git.uwaterloo.ca:5050/watonomous/registry/misc/rowan-ml-assignments-cuda
WORKDIR /project

# RUN apt-get update && apt-get install -y python3.8-venv

# ENV VIRTUAL_ENV=/home/docker/venv
# RUN python3 -m venv $VIRTUAL_ENV
# ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN apt-get update && apt-get install -y libopenexr-dev openexr
RUN pip3 install --upgrade pip 
RUN pip3 install numpy tqdm matplotlib
RUN pip3 install torch==1.10.0+cu111 torchvision==0.11.0+cu111 -f https://download.pytorch.org/whl/cu111/torch_stable.html
RUN pip3 install scikit-image pandas OpenEXR