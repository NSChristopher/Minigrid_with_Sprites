# Use the base Codespace image
FROM mcr.microsoft.com/devcontainers/universal:2

# Install necessary packages
RUN apt-get update && apt-get install -y \
    tigervnc-standalone-server \
    novnc \
    websockify

# Install websockify via pip
RUN pip3 install websockify

# Clone the latest noVNC code
RUN git clone https://github.com/novnc/noVNC.git /opt/noVNC

# Make sure the novnc_proxy script is executable
RUN chmod +x /opt/noVNC/utils/novnc_proxy

# Expose ports
EXPOSE 6080 5901