cmake_minimum_required(VERSION 3.0.0)

cmake_policy(SET CMP0054 NEW)

if(WIN32)
    add_compile_options(/wd4624)
    add_compile_options(/wd4067)
    add_compile_options(/wd4819)
    add_compile_options(/wd4805)
    add_compile_options(/D_CRT_SECURE_NO_WARNINGS)
endif()

project(playground VERSION 0.1.0)

find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include_directories(${OpenCV_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS})

add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${TORCH_LIBRARIES})
