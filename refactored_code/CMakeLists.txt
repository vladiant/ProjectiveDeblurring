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
    include/BlurKernelGenerator.hpp
    src/BlurKernelGenerator.cpp
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
    Convergence
    Convergence.cpp
)

set_target_properties(
    Convergence
    PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
)

target_link_libraries(
    Convergence
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


add_executable(
    TVRegDeblur
    TVRegDeblur.cpp
)

set_target_properties(
    TVRegDeblur
    PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
)

target_link_libraries(
    TVRegDeblur
    PRIVATE
      ${PROJECT_NAME}
)

add_executable(
    LaplRegDeblur
    LaplRegDeblur.cpp
)

set_target_properties(
    LaplRegDeblur
    PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
)

target_link_libraries(
    LaplRegDeblur
    PRIVATE
      ${PROJECT_NAME}
)

add_executable(
    BilateralRegDeblur
    BilateralRegDeblur.cpp
)

set_target_properties(
    BilateralRegDeblur
    PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
)

target_link_libraries(
    BilateralRegDeblur
    PRIVATE
      ${PROJECT_NAME}
)

add_executable(
    BilateralLaplRegDeblur
    BilateralLaplRegDeblur.cpp
)

set_target_properties(
    BilateralLaplRegDeblur
    PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
)

target_link_libraries(
    BilateralLaplRegDeblur
    PRIVATE
      ${PROJECT_NAME}
)

add_executable(
    BasicMultiscaleDeblur
    BasicMultiscaleDeblur.cpp
)

set_target_properties(
    BasicMultiscaleDeblur
    PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
)

target_link_libraries(
    BasicMultiscaleDeblur
    PRIVATE
      ${PROJECT_NAME}
)


add_executable(
    BlurKernelDeblur
    BlurKernelDeblur.cpp
)

set_target_properties(
    BlurKernelDeblur
    PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
)

target_link_libraries(
    BlurKernelDeblur
    PRIVATE
      ${PROJECT_NAME}
)