# Compile project code with current optimization mode configured in CMake
add_library(${ModuleName}_lib STATIC)

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# SOURCE FILES ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
file(GLOB CXX_FILES *.cpp)
target_sources(${ModuleName}_lib PRIVATE ${CXX_FILES})

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# LIBRARIES :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# target_link_libraries(${ModuleName}_lib OpenGL::GL glfw)


#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# COMPILER OPTIONS ::::::::::::::::::::::::::::::::::::::::::::::::::::::
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Need position independent code (-fPIC) to link into python extension
set_target_properties(${ModuleName}_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_compile_definitions(${ModuleName}_lib PUBLIC
$<$<CONFIG:Debug>:DEBUG>
$<$<NOT:$<CONFIG:Debug>>:NDEBUG>
)

target_compile_features(${ModuleName}_lib PUBLIC cxx_std_17)

target_compile_options(${ModuleName}_lib PUBLIC
    # warnings
    $<$<CXX_COMPILER_ID:Clang>:-Wall -Weverything -Wno-c++98-compat -Wno-c++98-compat-pedantic>
    $<$<CXX_COMPILER_ID:GCC>:-pedantic -Wall -Wextra -Wcast-align -Wcast-qual -Wctor-dtor-privacy -Wdisabled-optimization -Wformat=2 -Winit-self -Wlogical-op -Wmissing-declarations -Wmissing-include-dirs -Wnoexcept -Wold-style-cast -Woverloaded-virtual -Wredundant-decls -Wshadow -Wsign-conversion -Wsign-promo -Wstrict-null-sentinel -Wstrict-overflow=5 -Wswitch-default -Wundef -Werror -Wno-unused>
    $<$<CXX_COMPILER_ID:MSVC>:/W4>

    # flags for each build type
    $<$<CXX_COMPILER_ID:Clang>:
        $<$<CONFIG:Release>:-O3>                  # Release
        $<$<CONFIG:Debug>:-O0 -g>                 # Debug
        $<$<CONFIG:RelWithDebgInfo>:-O3 -g>       # RelWithDebInfo
    >
    $<$<CXX_COMPILER_ID:GCC>:
        $<$<CONFIG:Release>:-O3>                  # Release
        $<$<CONFIG:Debug>:-O0 -g>                 # Debug
        $<$<CONFIG:RelWithDebgInfo>:-O3 -g>       # RelWithDebInfo
    >

    $<$<CXX_COMPILER_ID:MSVC>: 
        $<$<CONFIG:Release>:/O2 /Zi /MP>          # Release
        $<$<CONFIG:Debug>:/Od /ZI /MP>            # Debug
        $<$<CONFIG:RelWithDebgInfo>:/O2 /ZI /MP>  # RelWithDebInfo
    >
)

