#!/bin/bash
# Usage: ./create.sh <problem_name>
# Example: ./create.sh convolution

if [ -z "$1" ]; then
    echo "Usage: ./create.sh <name>"
    exit 1
fi

NAME=$1
SRC_DIR="src"
TEMPLATE_FILE="template.cu" # Optional: if you want a separate template file

# 1. Find the Next Number
# Lists files, extracts the first 3 digits, sorts numerically, takes the last one.
LAST_FILE=$(ls $SRC_DIR/[0-9][0-9][0-9]*.cu 2>/dev/null | sort | tail -n 1)

if [ -z "$LAST_FILE" ]; then
    NEXT_NUM="001"
else
    # Extract number (e.g. src/005_foo.cu -> 005)
    LAST_NUM=$(basename "$LAST_FILE" | cut -c 1-3)
    # Increment (10# forces base 10 to avoid octal confusion with 008)
    NEXT_NUM=$(printf "%03d" $((10#$LAST_NUM + 1)))
fi

NEW_FILE="$SRC_DIR/${NEXT_NUM}_${NAME}.cu"

# 2. Generate the File content (The Boilerplate)
cat <<EOF > "$NEW_FILE"
#include <iostream>
#include <cuda_runtime.h>

// --- KERNEL: ${NAME} ---
__global__ void ${NAME}_kernel(float* d_in, float* d_out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // TODO: Implement ${NAME} logic
        d_out[idx] = d_in[idx];
    }
}

int main() {
    int N = 1 << 20; // 1M elements
    size_t bytes = N * sizeof(float);

    // Allocate Unified Memory (Read by CPU & GPU)
    float *data;
    cudaMallocManaged(&data, bytes);

    // Initialize
    for(int i=0; i<N; i++) data[i] = 1.0f;

    // Launch
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    cudaEventRecord(start);
    ${NAME}_kernel<<<blocks, threads>>>(data, data, N);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    printf("Kernel: ${NAME} | Time: %.3f ms\n", ms);

    cudaFree(data);
    return 0;
}
EOF

echo "✅ Created: $NEW_FILE"