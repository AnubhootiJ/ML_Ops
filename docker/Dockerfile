FROM ubuntu:18.04
COPY plot_digits_classification.py /exp/plot_digits_classification.py
COPY requirements.txt /exp/requirements.txt
RUN apt-get update && apt-get install -y python3 python3-pip
RUN pip3 install --no-cache-dir -r /exp/requirements.txt
RUN mkdir /exp/models
WORKDIR /exp
CMD [ "python3", "plot_digits_classification.py" ]