# syntax=docker/dockerfile:1.4
FROM --platform=linux/amd64 ubuntu:22.04

LABEL maintainer="FPGA Development"
LABEL description="Xilinx Vivado/Vitis HLS with JTAG support"

ENV DEBIAN_FRONTEND=noninteractive
ENV XILINX_DIR=/opt/Xilinx
ENV XILINX_VERSION=2023.1
ENV XILINX_LICENSE_FILE=/root/.Xilinx/Xilinx.lic

# Install dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    tar \
    xz-utils \
    libncurses-dev \
    libstdc++6 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libusb-1.0-0 \
    openjdk-11-jdk-headless \
    python3 \
    python3-pip \
    git \
    make \
    gcc \
    g++ \
    kmod \
    sudo \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create directories
RUN mkdir -p ${XILINX_DIR} \
    && mkdir -p /root/.Xilinx \
    && mkdir -p /workspace

# Copy license file
COPY licenses/xilinx.lic ${XILINX_LICENSE_FILE}

# Copy and extract installer
COPY xilinx_installers/Xilinx.tar.gz ${XILINX_DIR}/Xilinx.tar.gz

# Extract the unified installer tarball
RUN echo "Extracting Xilinx Unified installer (~10-30 min)..." && \
    tar -xzf ${XILINX_DIR}/Xilinx.tar.gz -C ${XILINX_DIR}/ && \
    rm -f ${XILINX_DIR}/Xilinx.tar.gz && \
    ls -la ${XILINX_DIR}/

# Create install config for headless installation
RUN echo '
[CODE]
BOM_BOARD=ZYNQ-7000
BOM_REVISION=Latest
BOM_SERIAL_NUM=xc7z010
INSTALL_DEVICE=yes
INSTALL_DOCS=no
INSTALL_EXAMPLES=no
INSTALL_IC=yes
INSTALL_SECURE，跌
INSTALL_SIM=no
INSTALL_SYMMETRICAL=yes
INSTALL_TARGET=all
SKIP_2ND_BOARD=yes
XISIM_VIVADO_VERSION=
' > /tmp/install.cfg

# Run Xilinx installer in batch mode
# Note: This requires license file to be valid
RUN export XILINX_LICENSE_FILE=${XILINX_LICENSE_FILE} && \
    yes | ${XILINX_DIR}/Xilinx_Unified_2023.1_0507_1903/xsetup \
        --config /tmp/install.cfg \
        --batch || true

# Alternative: Just extract and leave uninstalled for manual install
RUN chmod -R 755 ${XILINX_DIR} && \
    chmod -R 755 /root/.Xilinx

# Create fpgauser
RUN useradd -m -s /bin/bash fpgauser && \
    usermod -aG sudo fpgauser && \
    chown -R fpgauser:fpgauser /workspace

# Copy workspace files
COPY --chown=fpgauser:fpgauser ../hls /workspace/hls
COPY --chown=fpgauser:fpgauser ../firmware /workspace/firmware
COPY --chown=fpgauser:fpgauser ../scripts /workspace/scripts

USER fpgauser
WORKDIR /workspace

# Add Xilinx to PATH
ENV PATH="${XILINX_DIR}/Vivado/2023.1/bin:${XILINX_DIR}/Vitis_HLS/2023.1/bin:${PATH}"
ENV LD_LIBRARY_PATH="${XILINX_DIR}/Vivado/2023.1/lib/lnx64.o:${XILINX_DIR}/Vitis_HLS/2023.1/lib/lnx64:${LD_LIBRARY_PATH}"

CMD ["/bin/bash"]