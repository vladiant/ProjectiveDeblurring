cmake_minimum_required(VERSION 3.10)

project(ProjectiveDeblur)

find_package(Threads REQUIRED)

add_executable(
    ${PROJECT_NAME}
    BicubicInterpolation.h
    BicubicInterpolation.cpp
    bitmap.h
    bitmap.cpp
    homography.h
    homography.cpp
    ImResize.h
    ImResize.cpp
    svdcmp.h
    svdcmp.cpp
    ProjectiveMotionRL.cpp
    ProjectiveMotionRL_main.cpp
)

set_target_properties(
    ${PROJECT_NAME}
    PROPERTIES
        CXX_STANDARD 11
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
)

# target_link_libraries(
#     ${PROJECT_NAME}
#     PRIVATE
#         Threads::Threads
# )

target_include_directories(
    ${PROJECT_NAME}
    PUBLIC
        .
)
