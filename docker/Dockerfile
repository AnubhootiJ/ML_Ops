FROM ubuntu:18.04
COPY docker.py /exp/docker.py
COPY requirements.txt /exp/requirements.txt
RUN apt-get update && apt-get install -y python python-pip
RUN pip install --no-cache-dir -r /exp/requirements.txt
RUN mkdir /exp/models
WORKDIR /exp
CMD [ "python", "docker.py" ]
