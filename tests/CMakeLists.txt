add_executable(test-moonshine test-moonshine.cpp)
target_link_libraries(test-moonshine PRIVATE moonshine ggml)
target_include_directories(test-moonshine PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/ggml/include
)

add_test(
    NAME moonshine-basic
    COMMAND test-moonshine
        --fixtures ${CMAKE_SOURCE_DIR}/tests/fixtures
        --model ${CMAKE_SOURCE_DIR}/models/moonshine-tiny-f32.gguf
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

add_executable(test-streaming-frontend test-streaming-frontend.cpp)
target_link_libraries(test-streaming-frontend PRIVATE moonshine-streaming ggml)
target_include_directories(test-streaming-frontend PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/ggml/include
)

add_test(
    NAME streaming-frontend
    COMMAND test-streaming-frontend
        -m ${CMAKE_SOURCE_DIR}/models/moonshine-streaming-tiny-f32.gguf
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

add_executable(test-streaming-encoder test-streaming-encoder.cpp)
target_link_libraries(test-streaming-encoder PRIVATE moonshine-streaming ggml)
target_include_directories(test-streaming-encoder PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/ggml/include
)

add_test(
    NAME streaming-encoder
    COMMAND test-streaming-encoder
        -m ${CMAKE_SOURCE_DIR}/models/moonshine-streaming-tiny-f32.gguf
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

add_executable(test-streaming-transcribe test-streaming-transcribe.cpp)
target_link_libraries(test-streaming-transcribe PRIVATE moonshine-streaming ggml)
target_include_directories(test-streaming-transcribe PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/ggml/include
)

add_test(
    NAME streaming-transcribe
    COMMAND test-streaming-transcribe
        -m ${CMAKE_SOURCE_DIR}/models/moonshine-streaming-tiny-f32.gguf
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

add_executable(test-streaming-incremental test-streaming-incremental.cpp)
target_link_libraries(test-streaming-incremental PRIVATE moonshine-streaming ggml)
target_include_directories(test-streaming-incremental PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/ggml/include
)

add_test(
    NAME streaming-incremental
    COMMAND test-streaming-incremental
        -m ${CMAKE_SOURCE_DIR}/models/moonshine-streaming-tiny-f32.gguf
        -a ${CMAKE_SOURCE_DIR}/tests/fixtures/beckett.wav
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
