cmake_minimum_required(VERSION 3.22)
SET(PROJECT_NAME demodulator)
set(CMAKE_C_STANDARD 99)

#add_compile_options(-g)
add_compile_options(-O3)

if (CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "x86_64")
    project(${PROJECT_NAME} LANGUAGES C ASM)
    if (CMAKE_C_COMPILER_ID STREQUAL "Intel")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native -mavx2 -fomit-frame-pointer -fno-math-errno -fno-trapping-math -diag-disable=10441")
    elseif (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native -mtune=intel -mavx2 -fno-dwarf2-cfi-asm -fno-asynchronous-unwind-tables -fno-exceptions -fomit-frame-pointer -fno-math-errno -fno-trapping-math")
    elseif(CMAKE_C_COMPILER_ID STREQUAL "GNU")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-unused-result -march=native -mtune=intel -mavx2 -fno-dwarf2-cfi-asm -fno-asynchronous-unwind-tables -fno-exceptions -fomit-frame-pointer -fno-math-errno -fno-trapping-math")
    endif()

    FILE(GLOB SRCS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c" "${CMAKE_CURRENT_SOURCE_DIR}/asm/*.S")
    include_directories("${CMAKE_HOME_DIRECTORY}/include")
    add_executable(${PROJECT_NAME} ${SRCS})

    if(CMAKE_C_COMPILER_ID STREQUAL "GNU")
        target_link_libraries(${PROJECT_NAME} PUBLIC m)
    endif()
else()
    set(CMAKE_CUDA_ARCHITECTURES 30 32 53 61 62 70 72 75)
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda-10.2/bin/nvcc")
    project(${PROJECT_NAME} LANGUAGES CUDA)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-unused-result -fno-dwarf2-cfi-asm -fno-asynchronous-unwind-tables -fno-exceptions -fomit-frame-pointer -fno-math-errno -fno-trapping-math")

    FILE(GLOB SRCS "${CMAKE_CURRENT_SOURCE_DIR}/cuda/*.cu")
    include_directories("${CMAKE_HOME_DIRECTORY}/include" "${CMAKE_HOME_DIRECTORY}/cuda")

    add_executable(${PROJECT_NAME} ${SRCS} cuda/matrix.cuh)

    target_link_libraries(${PROJECT_NAME} PUBLIC m)
endif()

