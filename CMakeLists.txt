cmake_minimum_required(VERSION 3.5)

project(UsingMKL)

list(APPEND CMAKE_MODULE_PATH C:/MyLib/CMake)
# include(Common)
find_package(MKL REQUIRED)

# set( BLA_STATIC ON) 
# set( BLA_VENDOR "Intel10_64lp_seq") 
# find_package( BLAS ) 
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
add_subdirectory(src)
