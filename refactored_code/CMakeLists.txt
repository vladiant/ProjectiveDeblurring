cmake_minimum_required(VERSION 3.10)

project(ProjectiveDeblur)

add_executable(
    ${PROJECT_NAME}
    include/BicubicInterpolation.h
    src/BicubicInterpolation.cpp
    include/warping.h
    src/warping.cpp
    include/bitmap.h
    src/bitmap.cpp
    include/homography.h
    src/homography.cpp
    include/ImResize.h
    src/ImResize.cpp
    include/svdcmp.h
    src/svdcmp.cpp
    include/IErrorCalculator.h
    include/EmptyErrorCalculator.h
    src/EmptyErrorCalculator.cpp
    include/RMSErrorCalculator.h
    src/RMSErrorCalculator.cpp
    include/INoiseGenerator.h
    include/GaussianNoiseGenerator.h
    src/GaussianNoiseGenerator.cpp
    include/IBlurImageGenerator.h
    include/MotionBlurImageGenerator.h
    src/MotionBlurImageGenerator.cpp
    include/ProjectiveMotionRL.h
    src/ProjectiveMotionRL.cpp
    include/ProjectiveMotionRLMultiScaleGray.h
    src/ProjectiveMotionRLMultiScaleGray.cpp
    ProjectiveMotionRL_main.cpp
)

set_target_properties(
    ${PROJECT_NAME}
    PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
)

target_include_directories(
    ${PROJECT_NAME}
    PUBLIC
        include
)

target_compile_options(
    ${PROJECT_NAME}
    PRIVATE
         $<$<CXX_COMPILER_ID:MSVC>:/W4 /WX>
         $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic>
         $<$<CXX_COMPILER_ID:Clang>:-fsanitize=address>
)

target_link_libraries(
   ${PROJECT_NAME}
   PRIVATE
      $<$<CXX_COMPILER_ID:Clang>:-fsanitize=address>
)
