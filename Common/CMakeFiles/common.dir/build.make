# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# The program to use to edit the cache.
CMAKE_EDIT_COMMAND = /usr/bin/ccmake

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/opencog/Desktop/Min/EM-DeSTIN

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/opencog/Desktop/Min/EM-DeSTIN

# Include any dependencies generated for this target.
include Common/CMakeFiles/common.dir/depend.make

# Include the progress variables for this target.
include Common/CMakeFiles/common.dir/progress.make

# Include the compile flags for this target's objects.
include Common/CMakeFiles/common.dir/flags.make

Common/CMakeFiles/common.dir/DestinNetworkAlt.cpp.o: Common/CMakeFiles/common.dir/flags.make
Common/CMakeFiles/common.dir/DestinNetworkAlt.cpp.o: Common/DestinNetworkAlt.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/opencog/Desktop/Min/EM-DeSTIN/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object Common/CMakeFiles/common.dir/DestinNetworkAlt.cpp.o"
	cd /home/opencog/Desktop/Min/EM-DeSTIN/Common && /usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/common.dir/DestinNetworkAlt.cpp.o -c /home/opencog/Desktop/Min/EM-DeSTIN/Common/DestinNetworkAlt.cpp

Common/CMakeFiles/common.dir/DestinNetworkAlt.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/common.dir/DestinNetworkAlt.cpp.i"
	cd /home/opencog/Desktop/Min/EM-DeSTIN/Common && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/opencog/Desktop/Min/EM-DeSTIN/Common/DestinNetworkAlt.cpp > CMakeFiles/common.dir/DestinNetworkAlt.cpp.i

Common/CMakeFiles/common.dir/DestinNetworkAlt.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/common.dir/DestinNetworkAlt.cpp.s"
	cd /home/opencog/Desktop/Min/EM-DeSTIN/Common && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/opencog/Desktop/Min/EM-DeSTIN/Common/DestinNetworkAlt.cpp -o CMakeFiles/common.dir/DestinNetworkAlt.cpp.s

Common/CMakeFiles/common.dir/DestinNetworkAlt.cpp.o.requires:
.PHONY : Common/CMakeFiles/common.dir/DestinNetworkAlt.cpp.o.requires

Common/CMakeFiles/common.dir/DestinNetworkAlt.cpp.o.provides: Common/CMakeFiles/common.dir/DestinNetworkAlt.cpp.o.requires
	$(MAKE) -f Common/CMakeFiles/common.dir/build.make Common/CMakeFiles/common.dir/DestinNetworkAlt.cpp.o.provides.build
.PHONY : Common/CMakeFiles/common.dir/DestinNetworkAlt.cpp.o.provides

Common/CMakeFiles/common.dir/DestinNetworkAlt.cpp.o.provides.build: Common/CMakeFiles/common.dir/DestinNetworkAlt.cpp.o

Common/CMakeFiles/common.dir/VideoSource.cpp.o: Common/CMakeFiles/common.dir/flags.make
Common/CMakeFiles/common.dir/VideoSource.cpp.o: Common/VideoSource.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/opencog/Desktop/Min/EM-DeSTIN/CMakeFiles $(CMAKE_PROGRESS_2)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object Common/CMakeFiles/common.dir/VideoSource.cpp.o"
	cd /home/opencog/Desktop/Min/EM-DeSTIN/Common && /usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/common.dir/VideoSource.cpp.o -c /home/opencog/Desktop/Min/EM-DeSTIN/Common/VideoSource.cpp

Common/CMakeFiles/common.dir/VideoSource.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/common.dir/VideoSource.cpp.i"
	cd /home/opencog/Desktop/Min/EM-DeSTIN/Common && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/opencog/Desktop/Min/EM-DeSTIN/Common/VideoSource.cpp > CMakeFiles/common.dir/VideoSource.cpp.i

Common/CMakeFiles/common.dir/VideoSource.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/common.dir/VideoSource.cpp.s"
	cd /home/opencog/Desktop/Min/EM-DeSTIN/Common && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/opencog/Desktop/Min/EM-DeSTIN/Common/VideoSource.cpp -o CMakeFiles/common.dir/VideoSource.cpp.s

Common/CMakeFiles/common.dir/VideoSource.cpp.o.requires:
.PHONY : Common/CMakeFiles/common.dir/VideoSource.cpp.o.requires

Common/CMakeFiles/common.dir/VideoSource.cpp.o.provides: Common/CMakeFiles/common.dir/VideoSource.cpp.o.requires
	$(MAKE) -f Common/CMakeFiles/common.dir/build.make Common/CMakeFiles/common.dir/VideoSource.cpp.o.provides.build
.PHONY : Common/CMakeFiles/common.dir/VideoSource.cpp.o.provides

Common/CMakeFiles/common.dir/VideoSource.cpp.o.provides.build: Common/CMakeFiles/common.dir/VideoSource.cpp.o

# Object files for target common
common_OBJECTS = \
"CMakeFiles/common.dir/DestinNetworkAlt.cpp.o" \
"CMakeFiles/common.dir/VideoSource.cpp.o"

# External object files for target common
common_EXTERNAL_OBJECTS =

Common/libcommon.so: Common/CMakeFiles/common.dir/DestinNetworkAlt.cpp.o
Common/libcommon.so: Common/CMakeFiles/common.dir/VideoSource.cpp.o
Common/libcommon.so: Common/CMakeFiles/common.dir/build.make
Common/libcommon.so: Common/CMakeFiles/common.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX shared library libcommon.so"
	cd /home/opencog/Desktop/Min/EM-DeSTIN/Common && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/common.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
Common/CMakeFiles/common.dir/build: Common/libcommon.so
.PHONY : Common/CMakeFiles/common.dir/build

Common/CMakeFiles/common.dir/requires: Common/CMakeFiles/common.dir/DestinNetworkAlt.cpp.o.requires
Common/CMakeFiles/common.dir/requires: Common/CMakeFiles/common.dir/VideoSource.cpp.o.requires
.PHONY : Common/CMakeFiles/common.dir/requires

Common/CMakeFiles/common.dir/clean:
	cd /home/opencog/Desktop/Min/EM-DeSTIN/Common && $(CMAKE_COMMAND) -P CMakeFiles/common.dir/cmake_clean.cmake
.PHONY : Common/CMakeFiles/common.dir/clean

Common/CMakeFiles/common.dir/depend:
	cd /home/opencog/Desktop/Min/EM-DeSTIN && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/opencog/Desktop/Min/EM-DeSTIN /home/opencog/Desktop/Min/EM-DeSTIN/Common /home/opencog/Desktop/Min/EM-DeSTIN /home/opencog/Desktop/Min/EM-DeSTIN/Common /home/opencog/Desktop/Min/EM-DeSTIN/Common/CMakeFiles/common.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : Common/CMakeFiles/common.dir/depend

