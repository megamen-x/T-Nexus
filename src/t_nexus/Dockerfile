# Use an official Python image with CUDA support
FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime

WORKDIR /app

# Copy repository into the image so `pyproject.toml` is present in /app
COPY . /app

# Install the package from the application directory
RUN python -m pip install --upgrade pip && python -m pip install -e . 

# Copy server entrypoint and make it executable
# COPY docker/server_entrypoint.sh /app/docker/server_entrypoint.sh
# RUN chmod +x /app/docker/server_entrypoint.sh

EXPOSE 8000