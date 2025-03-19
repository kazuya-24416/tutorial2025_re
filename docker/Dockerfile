FROM nvcr.io/nvidia/pytorch:24.01-py3
ENV UID=1000006
ENV USER_NAME=kaneko.20033

# install essential softwares
# RUN if [ -e /etc/apt/sources.list.d/cuda.list ] ; then rm /etc/apt/sources.list.d/cuda.list; fi ; apt update
RUN apt update
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata
RUN apt install -y vim zsh git ssh sudo language-pack-en tmux libssl-dev libmysqlclient-dev sqlite3 build-essential
RUN update-locale LANG=en_US.UTF-8
RUN python -m pip install --upgrade pip
RUN python -m pip install setuptools packaging

RUN apt update && apt install -y curl
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

COPY requirements.txt /workspace/requirements.txt
RUN python -m pip install -r /workspace/requirements.txt

RUN adduser --uid ${UID} --shell /bin/zsh --force-badname --disabled-password ${USER_NAME} \
    && echo ${USER_NAME}:${USER_NAME} | chpasswd \
    && usermod -aG sudo ${USER_NAME}
RUN ln -s /workspace /home/${USER_NAME}/workspace && chown ${USER_NAME}:${USER_NAME} /workspace && chmod 755 /workspace
USER ${USER_NAME}
WORKDIR /workspace