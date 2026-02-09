#!/bin/bash
# Usage: ./run.sh 001
rm -rf build/*
# 1. AUTO-FIX: If 'build' folder doesn't exist, create and configure it
if [ ! -d "build" ] || [ ! -f "build/CMakeCache.txt" ]; then
    echo "⚠️  Build folder not found or empty. Initializing CMake..."
    # Ensure you are using the right NVCC path here if needed
    cmake -B build -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.6/bin/nvcc
fi

# 2. Find the target file (e.g., searches for "001*.cu" inside src/)
# We use 'find' to get the full filename, then strip path and extension
SRC_FILE=$(find src -name "$1*.cu" | head -n 1)

if [ -z "$SRC_FILE" ]; then
    echo "❌ Error: No file found starting with '$1' in src/"
    exit 1
fi

# Extract target name: "src/001_vector_add.cu" -> "001_vector_add"
FILENAME=$(basename -- "$SRC_FILE")
TARGET_NAME="${FILENAME%.*}"

echo "🔨 Building Target: $TARGET_NAME"

# 3. Build ONLY that specific target using CMake
# --target ensures we don't recompile the whole world, just this file
cmake --build build --target "$TARGET_NAME" -j$(nproc)

# 4. Run if build was successful
if [ $? -eq 0 ]; then
    echo "✅ Build Success! Running..."
    echo "----------------------------------------"
    ./build/"$TARGET_NAME"
else
    echo "❌ Build Failed."
    exit 1
fi