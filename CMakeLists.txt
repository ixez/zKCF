cmake_minimum_required(VERSION 3.0)
project(zKCF)

find_package(OpenCV REQUIRED)

set(src_dir ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(src
        ${src_dir}/fhog.cpp
        ${src_dir}/kcftracker.cpp
        ${src_dir}/Run.cpp
)

set(include_dir ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(zKCF ${src})
target_include_directories(zKCF PRIVATE ${include_dir})
target_link_libraries(zKCF ${OpenCV_LIBS})




