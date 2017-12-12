INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/include
                    $ENV{NVCUDASAMPLES_ROOT}/common/inc
                    C:/MyLib/yaolibrary
                    ${MKL_INCLUDE_DIR}
                    )


# Demo
# set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -mkl=parallel")
add_executable(mkl_demo mkl_demo.cpp)
target_link_libraries(mkl_demo ${MKL_LP_INTELTHREAD_LIBRARIES} "C:/Program Files (x86)/IntelSWTools/compilers_and_libraries_2018.0.124/windows/compiler/lib/intel64_win/libiomp5md.lib")


