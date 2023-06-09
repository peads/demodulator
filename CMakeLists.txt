cmake_minimum_required(VERSION 3.22)
SET(PROJECT_NAME demodulator)
set(CMAKE_C_STANDARD 99)

#add_compile_options(-g)
add_compile_options(-O3)

option(IS_NVIDIA OFF)
option(DEFAULT_BUF_SIZE OFF)
option(IS_INTEL OFF)

if(IS_INTEL)
    add_compile_definitions(IS_INTEL)
endif()

if (IS_NVIDIA)
    set(CMAKE_CUDA_ARCHITECTURES 52 53 61 62 70 72 75)
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
    add_compile_definitions(__CUDACC__)
    project(${PROJECT_NAME} LANGUAGES CUDA)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -m64 -ftz=true -use_fast_math -Xcompiler=\"-Wno-unused-result -fno-dwarf2-cfi-asm -fno-asynchronous-unwind-tables -fno-exceptions -fomit-frame-pointer -fno-math-errno -fno-trapping-math\" -res-usage ")
    FILE(GLOB SRCS "${CMAKE_CURRENT_SOURCE_DIR}/cuda/*.cu")
    include_directories("${CMAKE_HOME_DIRECTORY}/include")

    add_executable(${PROJECT_NAME} ${SRCS})
    set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
elseif(IS_INTEL)
    add_compile_definitions(DEFAULT_BUF_SIZE=2048)
    project(${PROJECT_NAME} LANGUAGES C ASM)
    if (CMAKE_C_COMPILER_ID STREQUAL "Intel")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native -mavx2 -mfma -fomit-frame-pointer -fno-math-errno -fno-trapping-math -diag-disable=10441")
    elseif (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native -mtune=intel -mavx2 -mfma -fno-dwarf2-cfi-asm -fno-asynchronous-unwind-tables -fno-exceptions -fomit-frame-pointer -fno-math-errno -fno-trapping-math")
    elseif(CMAKE_C_COMPILER_ID STREQUAL "GNU")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-unused-result -march=native -mtune=intel -mavx2 -fno-dwarf2-cfi-asm -fno-asynchronous-unwind-tables -fno-exceptions -fomit-frame-pointer -fno-math-errno -fno-trapping-math")
    endif()

    FILE(GLOB SRCS "${CMAKE_CURRENT_SOURCE_DIR}/asm/*.S")
    include_directories("${CMAKE_HOME_DIRECTORY}/include")
    add_executable(${PROJECT_NAME} src/main.c ${SRCS})

    if(CMAKE_C_COMPILER_ID STREQUAL "GNU")
        target_link_libraries(${PROJECT_NAME} PUBLIC m)
    endif()
else()
    project(${PROJECT_NAME} LANGUAGES C)
    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native -mavx2 -mfma -fno-dwarf2-cfi-asm -fno-asynchronous-unwind-tables -fno-exceptions -fomit-frame-pointer -fno-math-errno -fno-trapping-math")
    elseif(CMAKE_C_COMPILER_ID STREQUAL "GNU")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-unused-result -mavx2 -mfma -march=native -mtune=native -fno-dwarf2-cfi-asm -fno-asynchronous-unwind-tables -fno-exceptions -fomit-frame-pointer -fno-math-errno -fno-trapping-math")
    endif()

    FILE(GLOB SRCS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c")
    include_directories("${CMAKE_HOME_DIRECTORY}/include")
    add_executable(${PROJECT_NAME} ${SRCS} include/matrix.h)

    if(CMAKE_C_COMPILER_ID STREQUAL "GNU")
        target_link_libraries(${PROJECT_NAME} PUBLIC m)
    endif()
endif()
