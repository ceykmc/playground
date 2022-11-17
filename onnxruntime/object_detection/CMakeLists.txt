cmake_minimum_required(VERSION 3.0.0)

if(WIN32)
    add_compile_options(/wd4244)
    add_compile_options(/wd4267)
    add_compile_options(/D_CRT_SECURE_NO_WARNINGS)
endif()

project(playground VERSION 0.1.0)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(${PROJECT_NAME} object_detection.cpp)

target_include_directories(${PROJECT_NAME} PUBLIC ${ONNXRUNTIME_DIR}/include)
target_link_directories(${PROJECT_NAME} PUBLIC ${ONNXRUNTIME_DIR}/lib)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS}
    onnxruntime onnxruntime_providers_cuda
    onnxruntime_providers_shared onnxruntime_providers_tensorrt)
