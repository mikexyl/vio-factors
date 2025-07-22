cmake_minimum_required(VERSION 3.10)
project(vio-factors VERSION 0.0.1 LANGUAGES CXX)

find_package(GTSAM REQUIRED)
find_package(GTSAM_UNSTABLE REQUIRED)

# Include GTSAM CMake tools
find_package(GTSAMCMakeTools)
include(GtsamBuildTypes) # Load build type flags and default to Debug mode
include(GtsamTesting) # Easy functions for creating unit tests and scripts

# Create static library
add_library(${PROJECT_NAME} SHARED
  src/ExpandingIsotropic.cpp
)

target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(${PROJECT_NAME}
  PUBLIC
  gtsam
  gtsam_unstable
)

add_executable(SmartFactorExample
  examples/ISAM2Example_SmartFactor.cpp
)
target_link_libraries(SmartFactorExample
  PRIVATE
  ${PROJECT_NAME}
  gtsam
  gtsam_unstable
)
target_include_directories(SmartFactorExample
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(SmartStereoFactorExample
  examples/SmartStereoProjectionFactorExample.cpp
)
target_link_libraries(SmartStereoFactorExample
  PRIVATE
  ${PROJECT_NAME}
  gtsam
  gtsam_unstable
)


install(DIRECTORY include/vio_factors DESTINATION include)

install(TARGETS ${PROJECT_NAME}
    EXPORT ${PROJECT_NAME}Targets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(EXPORT ${PROJECT_NAME}Targets
    FILE ${PROJECT_NAME}Targets.cmake
    DESTINATION lib/cmake/${PROJECT_NAME}
)



# Package config (optional, for downstream find_package)
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY AnyNewerVersion
)
install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
  DESTINATION lib/cmake/${PROJECT_NAME}
)
# Generate and install the Config.cmake file
configure_package_config_file(
  cmake/${PROJECT_NAME}Config.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
  INSTALL_DESTINATION lib/cmake/${PROJECT_NAME}
)
