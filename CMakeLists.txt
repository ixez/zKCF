cmake_minimum_required(VERSION 3.0)
project(zKCF)

option(BUILD_VGG "Build VGG features." OFF)

find_package(OpenCV REQUIRED)
set(dep_libs ${OpenCV_LIBS})
set(dep_include_dirs "")

set(src_dir ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(src
        ${src_dir}/TaskConfig.cpp

        ${src_dir}/FFTTools.cpp
        ${src_dir}/KCF.cpp
        ${src_dir}/Run.cpp

        ${src_dir}/FkFactory.cpp

        ${src_dir}/Kernels/GaussianKernel.cpp

        ${src_dir}/Features/HogFeature.cpp
        ${src_dir}/Features/HogLabFeature.cpp
        ${src_dir}/Features/RawFeature.cpp

        ${src_dir}/Features/Impl/HogFeature/fhog.cpp
)

set(include_dir ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(include
        ${include_dir}/TaskConfig.h

        ${include_dir}/ITracker.h
        ${include_dir}/KCF.h
        ${include_dir}/recttools.hpp
        ${include_dir}/FFTTools.hpp

        ${include_dir}/Def.h
        ${include_dir}/FkFactory.h

        ${include_dir}/Kernels/IKernel.h
        ${include_dir}/Kernels/GaussianKernel.h

        ${include_dir}/Features/IFeature.h
        ${include_dir}/Features/HogFeature.h
        ${include_dir}/Features/HogLabFeature.h
        ${include_dir}/Features/RawFeature.h

        ${include_dir}/Features/Impl/HogFeature/fhog.hpp
        ${include_dir}/Features/Impl/HogLabFeature/labdata.hpp
)

if(BUILD_VGG)
    find_package(Caffe REQUIRED)
    set(dep_include_dirs ${Caffe_INCLUDE_DIRS})
    set(dep_libs ${dep_libs} ${Caffe_LIBRARIES})
    set(include ${include} ${include_dir}/Features/VggFeature.h)
    set(src ${src} ${src_dir}/Features/VggFeature.cpp)
    set(def ${def} BUILD_VGG)
endif()

add_executable(zKCF ${src} ${include})
target_include_directories(zKCF PRIVATE ${include_dir} ${dep_include_dirs})
target_link_libraries(zKCF ${dep_libs})
target_compile_definitions(zKCF PRIVATE ${def})

set_target_properties(zKCF PROPERTIES
        CXX_STANDARD 11
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
)

