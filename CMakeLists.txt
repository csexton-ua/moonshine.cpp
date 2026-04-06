cmake_minimum_required(VERSION 3.14)
project(moonshine.cpp VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

option(MOONSHINE_BUILD_EXAMPLES "Build example programs" ON)
option(MOONSHINE_BUILD_TESTS "Build test programs" ON)

add_subdirectory(ggml)

add_library(moonshine STATIC src/moonshine.cpp src/moonshine-tokenizer.cpp src/moonshine-audio.cpp)
target_include_directories(moonshine PUBLIC src)
target_include_directories(moonshine PRIVATE ggml/include)
target_link_libraries(moonshine PRIVATE ggml)

add_library(moonshine-streaming STATIC src/moonshine-streaming.cpp src/moonshine-tokenizer.cpp)
target_include_directories(moonshine-streaming PUBLIC src)
target_include_directories(moonshine-streaming PRIVATE ggml/include)
target_link_libraries(moonshine-streaming PRIVATE ggml)

add_library(moonshine-detect STATIC src/moonshine-detect.cpp)
target_include_directories(moonshine-detect PUBLIC src)
target_include_directories(moonshine-detect PRIVATE ggml/include)
target_link_libraries(moonshine-detect PRIVATE ggml)

if(NOT MSVC)
    target_compile_options(moonshine PRIVATE -Wall -Wextra)
    target_compile_options(moonshine-streaming PRIVATE -Wall -Wextra)
endif()

if(MOONSHINE_BUILD_EXAMPLES)
    add_subdirectory(examples/main)
    add_subdirectory(examples/stream-server)
endif()

if(MOONSHINE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Install rules
install(TARGETS moonshine moonshine-streaming moonshine-detect
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)
install(FILES
    src/moonshine.h
    src/moonshine-streaming.h
    src/moonshine-detect.h
    src/moonshine-audio.h
    DESTINATION include
)
