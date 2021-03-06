# Base image
FROM python:3

# Set working directory
WORKDIR /home

# Install required packages using apt
RUN apt-get update && \
    apt-get install -y apt-utils && \
    apt-get install -y \
        ffmpeg \
        gdal-bin \
        gfortran \
        libgdal-dev \
        libgeos-dev \
        libproj-dev \
        proj-bin \
        proj-data \
        python-gdal \
        wget

# Copy requirements file into container
COPY requirements.txt /tmp/requirements.txt

# Install required Python packages using pip
RUN pip install numpy && \
    pip install -r /tmp/requirements.txt

# Install latest version of Badlands from GitHub
RUN cd .. && \
    git clone https://github.com/badlands-model/badlands.git && \
    git clone https://github.com/badlands-model/badlands-companion.git && \
    cd badlands/badlands && \
    python setup.py install && \
    cd ../../badlands-companion && \
    python setup.py install && \
    cd .. && \
    rm -rf ./badlands*

# Enter Bash shell
CMD [ "/bin/bash" ]
