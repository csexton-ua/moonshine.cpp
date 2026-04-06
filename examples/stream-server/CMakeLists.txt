add_executable(moonshine-stream-server main.cpp)
target_link_libraries(moonshine-stream-server PRIVATE moonshine-streaming moonshine-detect ggml)
target_include_directories(moonshine-stream-server PRIVATE ${CMAKE_SOURCE_DIR}/ggml/include)

install(TARGETS moonshine-stream-server RUNTIME DESTINATION bin)
