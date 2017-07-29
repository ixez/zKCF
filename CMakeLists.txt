cmake_minimum_required(VERSION 3.0)
project(zKCF)

find_package(OpenCV REQUIRED)

set(src_dir ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(src
        ${src_dir}/fhog.cpp
        ${src_dir}/KCF.cpp
        ${src_dir}/Run.cpp
)

set(include_dir ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(include
        ${include_dir}/KCF.h
        ${include_dir}/fhog.hpp
        ${include_dir}/labdata.hpp
        ${include_dir}/recttools.hpp
        ${include_dir}/ffttools.hpp
)

add_executable(zKCF ${src} ${include})
target_include_directories(zKCF PRIVATE ${include_dir})
target_link_libraries(zKCF ${OpenCV_LIBS})




