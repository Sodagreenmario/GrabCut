cmake_minimum_required(VERSION 3.15)
project(GrabCut)

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

set(CMAKE_CXX_STANDARD 14)
set(SOURCE_FILES GCApplication.cpp GCApplication.h main.cpp GMM.cpp GMM.h grabcut.cpp grabcut.h graph.cpp graph.h maxflow.cpp)

add_executable(GrabCut ${SOURCE_FILES})
target_link_libraries(GrabCut ${OpenCV_LIBS})