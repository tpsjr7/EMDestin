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
include SOM/CMakeFiles/testSOM.dir/depend.make

# Include the progress variables for this target.
include SOM/CMakeFiles/testSOM.dir/progress.make

# Include the compile flags for this target's objects.
include SOM/CMakeFiles/testSOM.dir/flags.make

SOM/CMakeFiles/testSOM.dir/TestSom.cpp.o: SOM/CMakeFiles/testSOM.dir/flags.make
SOM/CMakeFiles/testSOM.dir/TestSom.cpp.o: SOM/TestSom.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/opencog/Desktop/Min/EM-DeSTIN/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object SOM/CMakeFiles/testSOM.dir/TestSom.cpp.o"
	cd /home/opencog/Desktop/Min/EM-DeSTIN/SOM && /usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/testSOM.dir/TestSom.cpp.o -c /home/opencog/Desktop/Min/EM-DeSTIN/SOM/TestSom.cpp

SOM/CMakeFiles/testSOM.dir/TestSom.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/testSOM.dir/TestSom.cpp.i"
	cd /home/opencog/Desktop/Min/EM-DeSTIN/SOM && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/opencog/Desktop/Min/EM-DeSTIN/SOM/TestSom.cpp > CMakeFiles/testSOM.dir/TestSom.cpp.i

SOM/CMakeFiles/testSOM.dir/TestSom.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/testSOM.dir/TestSom.cpp.s"
	cd /home/opencog/Desktop/Min/EM-DeSTIN/SOM && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/opencog/Desktop/Min/EM-DeSTIN/SOM/TestSom.cpp -o CMakeFiles/testSOM.dir/TestSom.cpp.s

SOM/CMakeFiles/testSOM.dir/TestSom.cpp.o.requires:
.PHONY : SOM/CMakeFiles/testSOM.dir/TestSom.cpp.o.requires

SOM/CMakeFiles/testSOM.dir/TestSom.cpp.o.provides: SOM/CMakeFiles/testSOM.dir/TestSom.cpp.o.requires
	$(MAKE) -f SOM/CMakeFiles/testSOM.dir/build.make SOM/CMakeFiles/testSOM.dir/TestSom.cpp.o.provides.build
.PHONY : SOM/CMakeFiles/testSOM.dir/TestSom.cpp.o.provides

SOM/CMakeFiles/testSOM.dir/TestSom.cpp.o.provides.build: SOM/CMakeFiles/testSOM.dir/TestSom.cpp.o

# Object files for target testSOM
testSOM_OBJECTS = \
"CMakeFiles/testSOM.dir/TestSom.cpp.o"

# External object files for target testSOM
testSOM_EXTERNAL_OBJECTS =

SOM/testSOM: SOM/CMakeFiles/testSOM.dir/TestSom.cpp.o
SOM/testSOM: SOM/libsom.so
SOM/testSOM: SOM/CMakeFiles/testSOM.dir/build.make
SOM/testSOM: SOM/CMakeFiles/testSOM.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable testSOM"
	cd /home/opencog/Desktop/Min/EM-DeSTIN/SOM && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/testSOM.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
SOM/CMakeFiles/testSOM.dir/build: SOM/testSOM
.PHONY : SOM/CMakeFiles/testSOM.dir/build

SOM/CMakeFiles/testSOM.dir/requires: SOM/CMakeFiles/testSOM.dir/TestSom.cpp.o.requires
.PHONY : SOM/CMakeFiles/testSOM.dir/requires

SOM/CMakeFiles/testSOM.dir/clean:
	cd /home/opencog/Desktop/Min/EM-DeSTIN/SOM && $(CMAKE_COMMAND) -P CMakeFiles/testSOM.dir/cmake_clean.cmake
.PHONY : SOM/CMakeFiles/testSOM.dir/clean

SOM/CMakeFiles/testSOM.dir/depend:
	cd /home/opencog/Desktop/Min/EM-DeSTIN && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/opencog/Desktop/Min/EM-DeSTIN /home/opencog/Desktop/Min/EM-DeSTIN/SOM /home/opencog/Desktop/Min/EM-DeSTIN /home/opencog/Desktop/Min/EM-DeSTIN/SOM /home/opencog/Desktop/Min/EM-DeSTIN/SOM/CMakeFiles/testSOM.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : SOM/CMakeFiles/testSOM.dir/depend

