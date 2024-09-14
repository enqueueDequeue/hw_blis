# Perfer using build kit while building
# Traditional build would transfer context in every run of build which is wasteful
# DOCKER_BUILDKIT=1 docker build . -t xilinx

FROM ubuntu:22.04 as extract

COPY ./xilinx/FPGAs_AdaptiveSoCs_Unified_2024.1_0522_2023.tar.gz /build/
RUN tar -xvf /build/FPGAs_AdaptiveSoCs_Unified_2024.1_0522_2023.tar.gz -C /build/

FROM ubuntu:22.04 as install

COPY --from=extract /build/FPGAs_AdaptiveSoCs_Unified_2024.1_0522_2023/ /root/xilinx_setup

RUN apt update && \
    apt install -y \
    libncurses5 \
    libtinfo5 \
    libncurses5-dev \
    libncursesw5-dev

RUN cd /root/xilinx_setup && \ 
    ./xsetup --agree 3rdPartyEULA,XilinxEULA --batch Install --product "Vitis" --edition "Vitis Unified Software Platform" --location /root/xilinx

FROM ubuntu:22.04

COPY --from=install /root/xilinx /root/xilinx
COPY --from=install /root/.Xilinx /root/.Xilinx

RUN apt update && \
    apt install -y \
    language-pack-en-base \
    build-essential \
    clang

RUN apt install -y \
    libncurses5 \
    libtinfo5 \
    libncurses5-dev \
    libncursesw5-dev
