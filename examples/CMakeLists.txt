add_dependency(argparse https://github.com/p-ranav/argparse 0b51382 TRUE)
add_dependency(spdlog https://github.com/gabime/spdlog v1.11.0 TRUE)

set(CMAKE_C_FLAGS_DEBUG "-g -DDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -DDEBUG")

add_executable(main main.cpp)
target_link_libraries(main PRIVATE minigpt4 ggml argparse spdlog)

