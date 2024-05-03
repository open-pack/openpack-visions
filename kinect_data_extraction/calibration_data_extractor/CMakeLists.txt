cmake_minimum_required(VERSION 3.10.2)
project(K4aFrameExtractor)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(k4a)
find_package(k4arecord)

add_executable(k4a_calibration_data_extractor
    k4a_calibration_data_extractor.cpp)

target_link_libraries(k4a_calibration_data_extractor PRIVATE
    k4a::k4a
    k4a::k4arecord
    /opt/libjpeg-turbo/lib64/libturbojpeg.so
    ${OpenCV_LIBS}
    stdc++fs
    )
