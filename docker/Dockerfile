FROM ubuntu:18.04
COPY apis/assg_10.py /exp/assg_10.py
COPY requirements.txt /exp/requirements.txt
COPY models/model_0.001.joblib /exp/model_0.001.joblib
COPY models/decisionTree/dec_32_(0.15,0.15).joblib /exp/dec_32_(0.15,0.15).joblib
RUN apt-get update && apt-get install -y python3.7 python3-pip
RUN pip3 install --no-cache-dir -r /exp/requirements.txt
RUN apt-get install -y curl
WORKDIR /exp
ENV LANG='C.UTF-8' LC_ALL='C.UTF-8'
ENV FLASK_APP assg_10
CMD ["flask", "run"]