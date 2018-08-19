FROM python:2.7.15

RUN pip install --upgrade pip
RUN pip install numpy
RUN pip install scipy
RUN pip install scikit-learn
RUN pip install theano
RUN pip install keras
RUN pip install h5py
RUN pip install scikit-image
RUN pip install seaborn
RUN pip install ghalton
RUN pip install ipdb
RUN pip install tensorflow
RUN pip install guppy

COPY ./ /app

WORKDIR /app/

CMD while true; do sleep 1000; done
