cmake_minimum_required(VERSION 3.16)
project(test)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-exceptions -fno-rtti")

# LLVM
set_tests_properties(${noArgsTests} PROPERTIES TIMEOUT 10)
find_package(LLVM REQUIRED CONFIG)
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(AddLLVM)
include_directories(${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

# MLIR
find_package(MLIR REQUIRED CONFIG)
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include(AddMLIR)
include_directories(${MLIR_INCLUDE_DIRS})
add_definitions(${MLIR_DEFINITIONS})

add_executable(test main.cpp)

target_link_libraries(
        test PUBLIC
        MLIRIR
        MLIRSCF
        MLIRSCFToStandard
        MLIRStandard
        MLIRStandardToLLVM
        MLIRTargetLLVMIR
        MLIRTransforms
        MLIRExecutionEngine
)
