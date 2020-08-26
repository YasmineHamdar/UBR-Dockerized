FROM tensorflow/tensorflow:2.1.0-gpu-py3-jupyter

WORKDIR /apps

# Configure tzdata timezone so it doesn't ask it in command line during build
ENV TZ=Asia/Beirut
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Update packages and install some essential linux programs
RUN apt-get update
RUN apt-get install -y vim apt-utils apt-transport-https software-properties-common

# Update pip
RUN pip install --upgrade pip

# R Setup
# -------------------
# Add repository where R can be fetched from
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9
RUN add-apt-repository 'deb https://cloud.r-project.org/bin/linux/ubuntu bionic-cran35/'
RUN apt update
# Download & Install R and R build essentials
RUN apt install -y r-base 
RUN apt install -y build-essential

# Install curl so R later can download packages 
RUN apt-get -y install libcurl4-openssl-dev

# Copy R file to container & run it to install packages
COPY r_setup.r ./
RUN chmod +x ./r_setup.r
RUN Rscript ./r_setup.r

# Copy text file to container & install all python libraries in this text file
COPY libraries.txt ./
RUN pip install -r libraries.txt