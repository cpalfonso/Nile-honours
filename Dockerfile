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
        python-gdal

# Copy requirements file into container
COPY requirements.txt /tmp/requirements.txt

# Install required Python packages using pip
RUN pip install numpy && \
    pip install -r /tmp/requirements.txt

# Enter Bash shell
CMD [ "/bin/bash" ]
