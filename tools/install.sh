#!/bin/bash
# luatorch installer
# checks dependencies, builds the library, runs smoke test

set -e

echo "luatorch installer"
echo ""

# check luajit
if ! command -v luajit &> /dev/null; then
    echo "ERROR: luajit not found"
    echo "install it:"
    echo "  ubuntu/debian: sudo apt install luajit"
    echo "  fedora:        sudo dnf install luajit"
    echo "  mac:           brew install luajit"
    echo "  from source:   https://luajit.org/download.html"
    exit 1
fi
echo "found luajit: $(luajit -v 2>&1 | head -1)"

# check cmake
if ! command -v cmake &> /dev/null; then
    echo "ERROR: cmake not found"
    echo "install it:"
    echo "  ubuntu/debian: sudo apt install cmake"
    echo "  mac:           brew install cmake"
    exit 1
fi
echo "found cmake: $(cmake --version | head -1)"

# check nvcc
if ! command -v nvcc &> /dev/null; then
    echo "ERROR: nvcc not found (CUDA toolkit not installed)"
    echo "install it from: https://developer.nvidia.com/cuda-downloads"
    echo "or: sudo apt install nvidia-cuda-toolkit"
    exit 1
fi
echo "found nvcc: $(nvcc --version | grep release)"

# detect gpu architecture
GPU_ARCH=""
if command -v nvidia-smi &> /dev/null; then
    # get compute capability from nvidia-smi
    COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
    if [ -n "$COMPUTE_CAP" ]; then
        GPU_ARCH="$COMPUTE_CAP"
        echo "detected gpu compute capability: ${COMPUTE_CAP:0:1}.${COMPUTE_CAP:1}"
    fi
fi

if [ -z "$GPU_ARCH" ]; then
    echo "could not detect gpu architecture, using native"
    GPU_ARCH="native"
fi

# check for nccl (optional, for multi gpu)
NCCL_FLAG=""
if pkg-config --exists nccl 2>/dev/null || [ -f /usr/lib/x86_64-linux-gnu/libnccl.so ] || [ -f /usr/local/lib/libnccl.so ]; then
    echo "found nccl: multi-gpu support enabled"
    NCCL_FLAG="-DLUAFLOW_USE_NCCL=ON"
else
    echo "nccl not found: multi-gpu support disabled (single gpu is fine)"
    NCCL_FLAG="-DLUAFLOW_USE_NCCL=OFF"
fi

echo ""

# build
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "building luatorch..."
mkdir -p build
cd build

cmake .. -DCMAKE_CUDA_ARCHITECTURES="$GPU_ARCH" $NCCL_FLAG 2>&1 | tail -5
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4) 2>&1 | tail -3

echo ""

# copy library to where ffi.load can find it
if [ -f libluatorch.so ]; then
    cp libluatorch.so "$PROJECT_DIR/lua/luatorch/"
    echo "copied libluatorch.so to lua/luatorch/"
elif [ -f libluatorch.dylib ]; then
    cp libluatorch.dylib "$PROJECT_DIR/lua/luatorch/"
    echo "copied libluatorch.dylib to lua/luatorch/"
else
    echo "ERROR: build produced no library file"
    exit 1
fi

cd "$PROJECT_DIR"

# smoke test
echo ""
echo "running smoke test (xor.lua)..."
LUA_PATH="lua/?.lua;lua/?/init.lua;;" LUA_CPATH="lua/luatorch/?.so;lua/luatorch/?.dylib;;" luajit examples/xor.lua 2>&1 | tail -5

echo ""
echo "luatorch installed successfully!"
echo ""
echo "next steps:"
echo "  run examples:    LUA_PATH='lua/?.lua;lua/?/init.lua;;' luajit examples/mnist.lua"
echo "  run benchmarks:  LUA_PATH='lua/?.lua;lua/?/init.lua;;' luajit tools/benchmark.lua"
echo "  train gpt:       LUA_PATH='lua/?.lua;lua/?/init.lua;;' luajit examples/gpt.lua"
