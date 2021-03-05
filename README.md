# Type conversion test
## How to run
CMake needs to know where the LLVM and MLIR Cmake files are located. This can be done using LLVM_DIR and MLIR_DIR.
For example:

```
mkdir build
cd build
cmake -DLLVM_DIR=/mnt/c/llvm/debug/lib/cmake/llvm -DMLIR_DIR=/mnt/c/llvm/debug/lib/cmake/mlir ..
make
./test
```
