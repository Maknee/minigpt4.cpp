cmake_minimum_required(VERSION 3.3)
project(minigpt4.cpp C CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include(FetchContent)

# General
option(MINIGPT4_BUILD_WITH_OPENCV      "minigpt4: build opencv (loading and encoding in c++)"     OFF)
option(MINIGPT4_BUILD_EXAMPLES         "minigpt4: build examples"                                 OFF)
option(MINIGPT4_BUILD_SHARED_LIBRARY   "minigpt4: build as a shared library"                      ON)
option(MINIGPT4_STATIC                 "minigpt4: static link libraries"                          OFF)
option(MINIGPT4_NATIVE                 "minigpt4: enable -march=native flag"                      OFF)
option(MINIGPT4_LTO                    "minigpt4: enable link time optimization"                  OFF)

# Debug
option(MINIGPT4_ALL_WARNINGS           "minigpt4: enable all compiler warnings"                   ON)
option(MINIGPT4_GPROF                  "minigpt4: enable gprof"                                   OFF)

# Sanitizers
option(MINIGPT4_SANITIZE_THREAD        "minigpt4: enable thread sanitizer"                        OFF)
option(MINIGPT4_SANITIZE_ADDRESS       "minigpt4: enable address sanitizer"                       OFF)
option(MINIGPT4_SANITIZE_UNDEFINED     "minigpt4: enable undefined sanitizer"                     OFF)

# Instruction set specific
option(MINIGPT4_AVX                    "minigpt4: enable AVX"                                     ON)
option(MINIGPT4_AVX2                   "minigpt4: enable AVX2"                                    ON)
option(MINIGPT4_AVX512                 "minigpt4: enable AVX512"                                  OFF)
option(MINIGPT4_FMA                    "minigpt4: enable FMA"                                     ON)

# 3rd party libs
option(MINIGPT4_ACCELERATE             "minigpt4: enable Accelerate framework"                    ON)
option(MINIGPT4_OPENBLAS               "minigpt4: use OpenBLAS"                                   OFF)
option(MINIGPT4_CUBLAS                 "minigpt4: use cuBLAS"                                     OFF)

# Build only shared library without building tests and extras
option(MINIGPT4_STANDALONE             "minigpt4: build only MINIGPT4 library"                    OFF)

#
# Compile flags
#

set(CMAKE_C_FLAGS_DEBUG "-g -DDEBUG") 
set(CMAKE_CXX_FLAGS_DEBUG "-g -DDEBUG")

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED true)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED true)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

if (NOT MSVC)
    if (MINIGPT4_SANITIZE_THREAD)
        add_compile_options(-fsanitize=thread)
        link_libraries(-fsanitize=thread)
    endif()

    if (MINIGPT4_SANITIZE_ADDRESS)
        add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
        link_libraries(-fsanitize=address)
    endif()

    if (MINIGPT4_SANITIZE_UNDEFINED)
        add_compile_options(-fsanitize=undefined)
        link_libraries(-fsanitize=undefined)
    endif()
endif()

if (APPLE AND MINIGPT4_ACCELERATE)
    find_library(ACCELERATE_FRAMEWORK Accelerate)
    if (ACCELERATE_FRAMEWORK)
        message(STATUS "Accelerate framework found")

        add_compile_definitions(GGML_USE_ACCELERATE)
        set(MINIGPT4_EXTRA_LIBS ${MINIGPT4_EXTRA_LIBS} ${ACCELERATE_FRAMEWORK})
    else()
        message(WARNING "Accelerate framework not found")
    endif()
endif()

if (MINIGPT4_OPENBLAS)
    if (MINIGPT4_STATIC)
        set(BLA_STATIC ON)
    endif()

    set(BLA_VENDOR OpenBLAS)
    find_package(BLAS)
    if (BLAS_FOUND)
        message(STATUS "OpenBLAS found")

        add_compile_definitions(GGML_USE_OPENBLAS)
        add_link_options(${BLAS_LIBRARIES})
    else()
        message(WARNING "OpenBLAS not found")
    endif()
endif()

if (MINIGPT4_CUBLAS)
    cmake_minimum_required(VERSION 3.17)

    find_package(CUDAToolkit)
    if (CUDAToolkit_FOUND)
        message(STATUS "cuBLAS found")

        enable_language(CUDA)

        set(GGML_CUDA_SOURCES ${CMAKE_SOURCE_DIR}/ggml/src/ggml-cuda.cu ${CMAKE_SOURCE_DIR}/ggml/src/ggml-cuda.h)

        add_compile_definitions(GGML_USE_CUBLAS)

        if (MINIGPT4_STATIC)
            set(MINIGPT4_EXTRA_LIBS ${MINIGPT4_EXTRA_LIBS} CUDA::cudart_static CUDA::cublas_static CUDA::cublasLt_static)
        else()
            set(MINIGPT4_EXTRA_LIBS ${MINIGPT4_EXTRA_LIBS} CUDA::cudart CUDA::cublas CUDA::cublasLt)
        endif()

    else()
        message(WARNING "cuBLAS not found")
    endif()
endif()

if (MINIGPT4_ALL_WARNINGS)
    if (NOT MSVC)
        set(c_flags
            -Wall
            -Wextra
            -Wpedantic
            -Wcast-qual
            -Wdouble-promotion
            -Wshadow
            -Wstrict-prototypes
            -Wpointer-arith
            -Wno-unused-function
        )
        set(cxx_flags
            -Wall
            -Wextra
            -Wpedantic
            -Wcast-qual
            -Wno-unused-function
            -Wno-multichar
        )
    else()
        set(c_flags
            -W4
        )
        set(cxx_flags
            -W4
        )
    endif()

    add_compile_options(
            "$<$<COMPILE_LANGUAGE:C>:${c_flags}>"
            "$<$<COMPILE_LANGUAGE:CXX>:${cxx_flags}>"
    )

endif()

if (MINIGPT4_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT result OUTPUT output)
    if (result)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    else()
        message(WARNING "IPO is not supported: ${output}")
    endif()
endif()

# Architecture specific
# TODO: probably these flags need to be tweaked on some architectures
#       feel free to update the Makefile for your architecture and send a pull request or issue
message(STATUS "CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
if (NOT MSVC)
    if (MINIGPT4_STATIC)
        add_link_options(-static)
        if (MINGW)
            add_link_options(-static-libgcc -static-libstdc++)
        endif()
    endif()
    if (MINIGPT4_GPROF)
        add_compile_options(-pg)
    endif()
    if (MINIGPT4_NATIVE)
        add_compile_options(-march=native)
    endif()
endif()

if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm" OR ${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64")
    message(STATUS "ARM detected")
    if (MSVC)
        # TODO: arm msvc?
    else()
        if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64")
            add_compile_options(-mcpu=native)
        endif()
        # TODO: armv6,7,8 version specific flags
    endif()
elseif (${CMAKE_SYSTEM_PROCESSOR} MATCHES "^(x86_64|i686|AMD64)$")
    message(STATUS "x86 detected")
    if (MSVC)
        if (MINIGPT4_AVX512)
            add_compile_options($<$<COMPILE_LANGUAGE:C>:/arch:AVX512>)
            add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/arch:AVX512>)
            # MSVC has no compile-time flags enabling specific
            # AVX512 extensions, neither it defines the
            # macros corresponding to the extensions.
            # Do it manually.
        elseif (MINIGPT4_AVX2)
            add_compile_options($<$<COMPILE_LANGUAGE:C>:/arch:AVX2>)
            add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2>)
        elseif (MINIGPT4_AVX)
            add_compile_options($<$<COMPILE_LANGUAGE:C>:/arch:AVX>)
            add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/arch:AVX>)
        endif()
    else()
        add_compile_options(-mf16c)
        if (MINIGPT4_FMA)
            add_compile_options(-mfma)
        endif()
        if (MINIGPT4_AVX)
            add_compile_options(-mavx)
        endif()
        if (MINIGPT4_AVX2)
            add_compile_options(-mavx2)
        endif()
        if (MINIGPT4_AVX512)
            add_compile_options(-mavx512f)
            add_compile_options(-mavx512bw)
        endif()
    endif()
else()
    # TODO: support PowerPC
    message(STATUS "Unknown architecture")
endif()

#
# Build libraries
#

if (MSVC)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
endif()

if (MINIGPT4_BUILD_SHARED_LIBRARY)
    set(MINIGPT4_LIBRARY_BUILD SHARED)
else()
    set(MINIGPT4_LIBRARY_BUILD STATIC)
endif()

macro(add_dependency)
    SET(dependency_name ${ARGV0})
    SET(endpoint_url ${ARGV1})
    SET(endpoint_tag ${ARGV2})
    SET(do_build_with_cmake ${ARGV3})

    FetchContent_Declare(
            ${dependency_name}
            GIT_REPOSITORY ${endpoint_url}
            GIT_TAG ${endpoint_tag}
    )

    FetchContent_GetProperties(${dependency_name})

    if (NOT ${dependency_name}_POPULATED)
        FetchContent_Populate(${dependency_name})
        message(STATUS "Working on ${dependency_name}")

        if (${do_build_with_cmake})
            add_subdirectory(${${dependency_name}_SOURCE_DIR} ${${dependency_name}_BINARY_DIR})
        else ()
            message("\tHeader only")
        endif ()
    endif ()
endmacro()

set(MINIGPT4_MSVC_USE_STATIC_CRT on CACHE BOOL "Use MT flags when compiling in MSVC")
if (MSVC)
  if (MINIGPT4_MSVC_USE_STATIC_CRT)
     message("-- Using static CRT linking ${MINIGPT4_MSVC_USE_STATIC_CRT}")
     foreach(flag_var CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
                          CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO
                          CMAKE_C_FLAGS_DEBUG CMAKE_C_FLAGS_RELEASE
                          CMAKE_C_FLAGS_MINSIZEREL CMAKE_C_FLAGS_RELWITHDEBINFO)
       string(REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
     endforeach()
  endif()
endif()

if (MINIGPT4_BUILD_SHARED_LIBRARY)
    # hack...
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    # set_property(TARGET fmt PROPERTY POSITION_INDEPENDENT_CODE ON)
endif()

#add_dependency(ggml https://github.com/ggerganov/ggml 93b94a2d41e880cb2abfb708535d5b04ad05b7a5 TRUE)
add_dependency(fmt https://github.com/fmtlib/fmt 9.1.0 TRUE)
add_dependency(unordered_dense https://github.com/martinus/unordered_dense v4.0.0 TRUE)
add_dependency(stb https://github.com/nothings/stb 5736b15 FALSE)
add_dependency(spdlog https://github.com/gabime/spdlog v1.11.0 TRUE)
add_dependency(nlohmann_json https://github.com/nlohmann/json v3.11.2 TRUE)

set(EXPECTED_BUILD_TESTS OFF)
add_dependency(tl_expected https://github.com/TartanLlama/expected v1.1.0 TRUE)

set(LLAMA_STATIC ${MINIGPT4_STATIC})
set(LLAMA_NATIVE ${MINIGPT4_NATIVE})
set(LLAMA_LTO ${MINIGPT4_LTO})
set(LLAMA_AVX ${MINIGPT4_AVX})
set(LLAMA_AVX2 ${MINIGPT4_AVX2})
set(LLAMA_AVX512 ${MINIGPT4_AVX512})
set(LLAMA_AVX512_VBMI ${MINIGPT4_AVX512_VBMI})
set(LLAMA_AVX512_VNNI ${MINIGPT4_AVX512_VNNI})
set(LLAMA_FMA ${MINIGPT4_FMA})
set(LLAMA_ACCELERATE ${MINIGPT4_ACCELERATE})
set(GGML_USE_K_QUANTS ON)
add_dependency(llama_cpp https://github.com/ggerganov/llama.cpp master-31cfbb1 TRUE)

set(OPENCV_INCLUDE_DIRS "")
set(OPENCV_LIBS "")
set(PILLOW_RESIZE_INCLUDE_DIRS "")
set(PILLOW_RESIZE_LIBS "")

if (MINIGPT4_BUILD_WITH_OPENCV)
    if (MINIGPT4_STATIC)
        set(OpenCV_STATIC ON)
    endif()
    find_package(OpenCV REQUIRED)
    set(OPENCV_INCLUDE_DIRS ${OpenCV_INCLUDE_DIRS})
    set(OPENCV_LIBS ${OpenCV_LIBS})

    add_dependency(pillow_resize https://github.com/zurutech/pillow-resize 4427c50 TRUE)

    set(PILLOW_RESIZE_INCLUDE_DIRS ${pillow_resize_SOURCE_DIR}/include/PillowResize)
    set(PILLOW_RESIZE_LIBS PillowResize)
    add_compile_definitions(MINIGPT4_BUILD_WITH_OPENCV)
else()
    add_dependency(magic_enum https://github.com/Neargye/magic_enum v0.9.3 TRUE)
endif()

add_library(minigpt4 ${MINIGPT4_LIBRARY_BUILD}
            minigpt4.cpp
            minigpt4.h)

target_include_directories(minigpt4 PUBLIC
    .

    ${fmt_SOURCE_DIR}
#    ${ggml_SOURCE_DIR}
    ${unordered_dense_SOURCE_DIR}
    ${stb_SOURCE_DIR}
    ${spdlog_SOURCE_DIR}
    ${nlohmann_json_SOURCE_DIR}
    ${tokenizers_cpp_SOURCE_DIR}
    ${llama_cpp_SOURCE_DIR}
    ${magic_enum_SOURCE_DIR}
    ${tl_expected_SOURCE_DIR}/include/tl

    ${OPENCV_INCLUDE_DIRS}
    ${PILLOW_RESIZE_INCLUDE_DIRS}
)

target_link_libraries(minigpt4 PUBLIC
    fmt
#    ggml
    unordered_dense
    spdlog
    nlohmann_json
    llama
    magic_enum
    expected
    
    ${OPENCV_LIBS}
    ${PILLOW_RESIZE_LIBS}
)

target_link_libraries(minigpt4 PRIVATE ${CLIP23_EXTRA_LIBS})

if (MSVC)
  if (CMAKE_BUILD_TYPE EQUAL "DEBUG")
    target_compile_options(minigpt4 PUBLIC "/ZI")
    target_link_options(minigpt4 PUBLIC "/INCREMENTAL")
  endif()
endif()

if (MINIGPT4_BUILD_SHARED_LIBRARY)
    set_target_properties(minigpt4 PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(minigpt4 PRIVATE MINIGPT4_SHARED MINIGPT4_BUILD)
endif()

if (MINIGPT4_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()