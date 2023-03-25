cmake_minimum_required(VERSION 3.22)
SET(PROJECT_NAME demodulator)
project(${PROJECT_NAME} LANGUAGES C)
set(CMAKE_C_STANDARD 99)

#add_compile_options(-g)
add_compile_options(-O3)

if (CMAKE_C_COMPILER_ID STREQUAL "Intel")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native -mavx2 -fomit-frame-pointer -fno-math-errno -fno-trapping-math -diag-disable=10441")
else()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native -mtune=intel -mavx2 -fomit-frame-pointer -fno-math-errno -fno-trapping-math")
endif()

FILE(GLOB SRCS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c")
include_directories("${CMAKE_HOME_DIRECTORY}/include")

add_executable(${PROJECT_NAME} ${SRCS})

if(CMAKE_C_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-unused-result -fno-dwarf2-cfi-asm -fno-asynchronous-unwind-tables -fno-exceptions -mno-avx256-split-unaligned-load")
    target_link_libraries(${PROJECT_NAME} PUBLIC m)
endif()
