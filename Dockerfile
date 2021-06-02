# set base image
FROM pytorch/pytorch

# set the working directory in the container
WORKDIR ./

# copy the dependencies file to the working directory
COPY requirements.txt .

# install dependencies
RUN pip install -r requirements.txt
