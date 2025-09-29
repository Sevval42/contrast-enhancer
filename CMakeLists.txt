cmake_minimum_required(VERSION 3.5...3.30)
project(contrast_enhancer)

set(CMAKE_CXX_STANDARD 11)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")

file(GLOB_RECURSE SOURCE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)

include(FetchContent)

FetchContent_Declare(
    stb
    GIT_REPOSITORY https://github.com/nothings/stb.git
    GIT_TAG master
)

FetchContent_Declare(
  yaml-cpp
  GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
  GIT_TAG        0.8.0  # pick a stable release tag
)

FetchContent_MakeAvailable(
    stb
    yaml-cpp
)

find_package(Vulkan REQUIRED)

find_path(CFITSIO_INCLUDE_DIR fitsio.h
          HINTS /opt/homebrew/include /usr/local/include)
find_library(CFITSIO_LIBRARY cfitsio
             HINTS /opt/homebrew/lib /usr/local/lib)

if(NOT CFITSIO_INCLUDE_DIR OR NOT CFITSIO_LIBRARY)
    message(FATAL_ERROR "CFITSIO not found. Please install")
endif()

if (UNIX)
    add_custom_target(build_shaders ALL
        COMMAND "${PROJECT_SOURCE_DIR}/shaders/compile.sh"
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/shaders"
    )
elseif(WIN32)
    add_custom_target(build_shaders ALL
        COMMAND "${PROJECT_SOURCE_DIR}/shaders/compile.bat"
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/shaders"
    )
endif()

add_executable(contrast_enhancer ${SOURCE_FILES})

target_link_libraries(contrast_enhancer PUBLIC Vulkan::Vulkan)
target_link_libraries(contrast_enhancer PRIVATE yaml-cpp)

target_include_directories(contrast_enhancer PRIVATE ${stb_SOURCE_DIR})

target_include_directories(contrast_enhancer PRIVATE ${CFITSIO_INCLUDE_DIR})
target_link_libraries(contrast_enhancer PRIVATE ${CFITSIO_LIBRARY})
