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
include Common/CMakeFiles/testDestinNetworkAlt.dir/depend.make

# Include the progress variables for this target.
include Common/CMakeFiles/testDestinNetworkAlt.dir/progress.make

# Include the compile flags for this target's objects.
include Common/CMakeFiles/testDestinNetworkAlt.dir/flags.make

Common/CMakeFiles/testDestinNetworkAlt.dir/TestDestinNetworkAlt.cpp.o: Common/CMakeFiles/testDestinNetworkAlt.dir/flags.make
Common/CMakeFiles/testDestinNetworkAlt.dir/TestDestinNetworkAlt.cpp.o: Common/TestDestinNetworkAlt.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/opencog/Desktop/Min/EM-DeSTIN/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object Common/CMakeFiles/testDestinNetworkAlt.dir/TestDestinNetworkAlt.cpp.o"
	cd /home/opencog/Desktop/Min/EM-DeSTIN/Common && /usr/bin/g++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/testDestinNetworkAlt.dir/TestDestinNetworkAlt.cpp.o -c /home/opencog/Desktop/Min/EM-DeSTIN/Common/TestDestinNetworkAlt.cpp

Common/CMakeFiles/testDestinNetworkAlt.dir/TestDestinNetworkAlt.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/testDestinNetworkAlt.dir/TestDestinNetworkAlt.cpp.i"
	cd /home/opencog/Desktop/Min/EM-DeSTIN/Common && /usr/bin/g++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/opencog/Desktop/Min/EM-DeSTIN/Common/TestDestinNetworkAlt.cpp > CMakeFiles/testDestinNetworkAlt.dir/TestDestinNetworkAlt.cpp.i

Common/CMakeFiles/testDestinNetworkAlt.dir/TestDestinNetworkAlt.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/testDestinNetworkAlt.dir/TestDestinNetworkAlt.cpp.s"
	cd /home/opencog/Desktop/Min/EM-DeSTIN/Common && /usr/bin/g++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/opencog/Desktop/Min/EM-DeSTIN/Common/TestDestinNetworkAlt.cpp -o CMakeFiles/testDestinNetworkAlt.dir/TestDestinNetworkAlt.cpp.s

Common/CMakeFiles/testDestinNetworkAlt.dir/TestDestinNetworkAlt.cpp.o.requires:
.PHONY : Common/CMakeFiles/testDestinNetworkAlt.dir/TestDestinNetworkAlt.cpp.o.requires

Common/CMakeFiles/testDestinNetworkAlt.dir/TestDestinNetworkAlt.cpp.o.provides: Common/CMakeFiles/testDestinNetworkAlt.dir/TestDestinNetworkAlt.cpp.o.requires
	$(MAKE) -f Common/CMakeFiles/testDestinNetworkAlt.dir/build.make Common/CMakeFiles/testDestinNetworkAlt.dir/TestDestinNetworkAlt.cpp.o.provides.build
.PHONY : Common/CMakeFiles/testDestinNetworkAlt.dir/TestDestinNetworkAlt.cpp.o.provides

Common/CMakeFiles/testDestinNetworkAlt.dir/TestDestinNetworkAlt.cpp.o.provides.build: Common/CMakeFiles/testDestinNetworkAlt.dir/TestDestinNetworkAlt.cpp.o

# Object files for target testDestinNetworkAlt
testDestinNetworkAlt_OBJECTS = \
"CMakeFiles/testDestinNetworkAlt.dir/TestDestinNetworkAlt.cpp.o"

# External object files for target testDestinNetworkAlt
testDestinNetworkAlt_EXTERNAL_OBJECTS =

Common/testDestinNetworkAlt: Common/CMakeFiles/testDestinNetworkAlt.dir/TestDestinNetworkAlt.cpp.o
Common/testDestinNetworkAlt: Common/libcommon.so
Common/testDestinNetworkAlt: EMDestin/libdestinalt.so
Common/testDestinNetworkAlt: Common/CMakeFiles/testDestinNetworkAlt.dir/build.make
Common/testDestinNetworkAlt: Common/CMakeFiles/testDestinNetworkAlt.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable testDestinNetworkAlt"
	cd /home/opencog/Desktop/Min/EM-DeSTIN/Common && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/testDestinNetworkAlt.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
Common/CMakeFiles/testDestinNetworkAlt.dir/build: Common/testDestinNetworkAlt
.PHONY : Common/CMakeFiles/testDestinNetworkAlt.dir/build

Common/CMakeFiles/testDestinNetworkAlt.dir/requires: Common/CMakeFiles/testDestinNetworkAlt.dir/TestDestinNetworkAlt.cpp.o.requires
.PHONY : Common/CMakeFiles/testDestinNetworkAlt.dir/requires

Common/CMakeFiles/testDestinNetworkAlt.dir/clean:
	cd /home/opencog/Desktop/Min/EM-DeSTIN/Common && $(CMAKE_COMMAND) -P CMakeFiles/testDestinNetworkAlt.dir/cmake_clean.cmake
.PHONY : Common/CMakeFiles/testDestinNetworkAlt.dir/clean

Common/CMakeFiles/testDestinNetworkAlt.dir/depend:
	cd /home/opencog/Desktop/Min/EM-DeSTIN && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/opencog/Desktop/Min/EM-DeSTIN /home/opencog/Desktop/Min/EM-DeSTIN/Common /home/opencog/Desktop/Min/EM-DeSTIN /home/opencog/Desktop/Min/EM-DeSTIN/Common /home/opencog/Desktop/Min/EM-DeSTIN/Common/CMakeFiles/testDestinNetworkAlt.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : Common/CMakeFiles/testDestinNetworkAlt.dir/depend

