#!/bin/bash

# Script to ensure Docker GPU support is properly configured after WSL restart
echo "Setting up Docker GPU support..."

# Stop snap docker if running
sudo snap stop docker 2>/dev/null || true

# Regenerate CDI configuration
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml

# Configure containerd for CDI
sudo nvidia-ctk runtime configure --runtime=containerd --config=/etc/containerd/config.toml

# Configure docker runtime
sudo nvidia-ctk runtime configure --runtime=docker

# Restart containerd and docker services properly
echo "Restarting services..."
sudo systemctl restart containerd
sudo systemctl restart docker.socket
sudo systemctl restart docker

# Wait for Docker to be fully ready
echo "Waiting for Docker to start..."
sleep 3

# Verify socket exists
if [ ! -S /var/run/docker.sock ]; then
    echo "Docker socket not found, restarting socket service..."
    sudo systemctl restart docker.socket
    sleep 2
fi

echo "Docker GPU setup complete!"
echo ""
echo "Testing GPU access..."
if sudo docker run --rm --device=nvidia.com/gpu=all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi >/dev/null 2>&1; then
    echo "✅ GPU access is working!"
    echo ""
    echo "Your DreamerV2 command is ready to use:"
    echo "sudo docker run -it --rm --device=nvidia.com/gpu=all \\"
    echo "  -e XLA_FLAGS=--xla_gpu_strict_conv_algorithm_picker=false \\"
    echo "  -e TF_XLA_FLAGS=--tf_xla_enable_xla_devices=false \\"
    echo "  -v ~/logdir:/logdir \\"
    echo "  dreamerv2 \\"
    echo "  python3 dreamerv2/dreamerv2/train.py \\"
    echo "  --logdir /logdir/atari_pong/dreamerv2/1 \\"
    echo "  --configs atari \\"
    echo "  --task atari_pong"
else
    echo "❌ GPU access test failed. Please check the logs above."
fi