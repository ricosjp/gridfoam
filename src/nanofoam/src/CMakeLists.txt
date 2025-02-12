add_subdirectory(libsrc)

# We are now ready to compile the actual extension module
nanobind_add_module(
  # Name of the extension
  ${ModuleName}_ext

  # Target the stable ABI for Python 3.12+, which reduces
  # the number of binary wheels that must be built. This
  # does nothing on older Python versions
  STABLE_ABI

  # Build libnanobind statically and merge it into the
  # extension (which itself remains a shared library)
  #
  # If your project builds multiple extensions, you can
  # replace this flag by NB_SHARED to conserve space by
  # reusing a shared libnanobind across libraries
  NB_STATIC

  # Source code goes here
  register.cpp
)
target_compile_definitions(${ModuleName}_ext PRIVATE MODULE_NAME=${ModuleName}_ext)
target_include_directories(${ModuleName}_ext PRIVATE libsrc)
target_link_libraries(${ModuleName}_ext PRIVATE ${ModuleName}_lib)

# Install directive for scikit-build-core
install(TARGETS ${ModuleName}_ext LIBRARY DESTINATION ${ModuleName})