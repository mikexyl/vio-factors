cmake_minimum_required(VERSION 3.10)
project(vio-factors VERSION 0.0.1 LANGUAGES CXX)

find_package(GTSAM REQUIRED)
find_package(GTSAM_UNSTABLE REQUIRED)

# Include GTSAM CMake tools
find_package(GTSAMCMakeTools)
include(GtsamBuildTypes) # Load build type flags and default to Debug mode
include(GtsamTesting) # Easy functions for creating unit tests and scripts

# Create static library
add_library(vio-factors SHARED
  src/ExpandingIsotropic.cpp
)

target_include_directories(vio-factors PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(vio-factors
  PUBLIC
  gtsam
  gtsam_unstable
)

add_executable(SmartFactorExample
  examples/ISAM2Example_SmartFactor.cpp
)
target_link_libraries(SmartFactorExample
  PRIVATE
  vio-factors
  gtsam
  gtsam_unstable
)
target_include_directories(SmartFactorExample
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Install targets
install(TARGETS vio-factors
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
install(DIRECTORY include/vio_factors DESTINATION include)

# Package config (optional, for downstream find_package)
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/vio-factorsConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY AnyNewerVersion
)
install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/vio-factorsConfigVersion.cmake"
  DESTINATION lib/cmake/vio-factors
)
