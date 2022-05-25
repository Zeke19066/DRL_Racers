cmake_minimum_required(VERSION 3.16)
project("Custom_Keymapper")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)  # optional, ensure standard is supported
set(CMAKE_CXX_EXTENSIONS OFF)  # optional, keep compiler extensions off


set(MODULE_SOURCE "${PROJECT_SOURCE_DIR}/module")
set(EXTERNALS "${PROJECT_SOURCE_DIR}/externals")

# pybind11
add_subdirectory(${EXTERNALS}/pybind11)

# module
pybind11_add_module (
    Custom_Keymapper
    ${MODULE_SOURCE}/module.cpp
)

target_include_directories (
    Custom_Keymapper
    PRIVATE ${MODULE_SOURCE}
)




