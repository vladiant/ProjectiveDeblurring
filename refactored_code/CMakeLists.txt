cmake_minimum_required(VERSION 3.10)

project(ProjectiveDeblur)

add_library(
    ${PROJECT_NAME}
    include/BicubicInterpolation.h
    src/BicubicInterpolation.cpp
    include/warping.h
    src/warping.cpp
    include/bitmap.h
    src/bitmap.cpp
    include/Homography.hpp
    src/Homography.cpp
    include/ImResize.h
    src/ImResize.cpp
    include/svdcmp.h
    src/svdcmp.cpp
    include/IErrorCalculator.hpp
    include/EmptyErrorCalculator.hpp
    src/EmptyErrorCalculator.cpp
    include/RMSErrorCalculator.hpp
    src/RMSErrorCalculator.cpp
    include/INoiseGenerator.hpp
    include/GaussianNoiseGenerator.hpp
    src/GaussianNoiseGenerator.cpp
    include/IBlurImageGenerator.hpp
    include/MotionBlurImageGenerator.hpp
    src/MotionBlurImageGenerator.cpp
    include/MotionBlurMaker.hpp
    src/MotionBlurMaker.cpp
    include/BlurUtils.hpp
    src/BlurUtils.cpp
    include/DeblurParameters.hpp
    include/RLDeblurrer.hpp
    src/RLDeblurrer.cpp
    include/IRegularizer.hpp
    include/EmptyRegularizer.hpp
    src/EmptyRegularizer.cpp
    include/TVRegularizer.hpp
    src/TVRegularizer.cpp
    include/LaplacianRegularizer.hpp
    src/LaplacianRegularizer.cpp
    include/BilateralRegularizer.hpp
    src/BilateralRegularizer.cpp
    include/BilateralLaplacianRegularizer.hpp
    src/BilateralLaplacianRegularizer.cpp
    include/ProjectiveMotionRLMultiScaleGray.hpp
    src/ProjectiveMotionRLMultiScaleGray.cpp
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

add_executable(
    Example
    ProjectiveMotionRL_main.cpp
)

set_target_properties(
    Example
    PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
)

target_link_libraries(
   Example
   PRIVATE
      ${PROJECT_NAME}
)


add_executable(
    BasicDeblur
    BasicDeblur.cpp
)

set_target_properties(
    BasicDeblur
    PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
)

target_link_libraries(
    BasicDeblur
   PRIVATE
      ${PROJECT_NAME}
)