add_executable(moonshine-cli main.cpp)
target_link_libraries(moonshine-cli PRIVATE moonshine moonshine-streaming moonshine-detect ggml)
target_include_directories(moonshine-cli PRIVATE ${CMAKE_SOURCE_DIR}/ggml/include)

install(TARGETS moonshine-cli RUNTIME DESTINATION bin)
