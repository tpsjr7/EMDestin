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
include EMDestin/CMakeFiles/destinalt.dir/depend.make

# Include the progress variables for this target.
include EMDestin/CMakeFiles/destinalt.dir/progress.make

# Include the compile flags for this target's objects.
include EMDestin/CMakeFiles/destinalt.dir/flags.make

EMDestin/CMakeFiles/destinalt.dir/src/destin.cpp.o: EMDestin/CMakeFiles/destinalt.dir/flags.make
EMDestin/CMakeFiles/destinalt.dir/src/destin.cpp.o: EMDestin/src/destin.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/opencog/Desktop/Min/EM-DeSTIN/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object EMDestin/CMakeFiles/destinalt.dir/src/destin.cpp.o"
	cd /home/opencog/Desktop/Min/EM-DeSTIN/EMDestin && /usr/bin/g++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/destinalt.dir/src/destin.cpp.o -c /home/opencog/Desktop/Min/EM-DeSTIN/EMDestin/src/destin.cpp

EMDestin/CMakeFiles/destinalt.dir/src/destin.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/destinalt.dir/src/destin.cpp.i"
	cd /home/opencog/Desktop/Min/EM-DeSTIN/EMDestin && /usr/bin/g++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/opencog/Desktop/Min/EM-DeSTIN/EMDestin/src/destin.cpp > CMakeFiles/destinalt.dir/src/destin.cpp.i

EMDestin/CMakeFiles/destinalt.dir/src/destin.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/destinalt.dir/src/destin.cpp.s"
	cd /home/opencog/Desktop/Min/EM-DeSTIN/EMDestin && /usr/bin/g++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/opencog/Desktop/Min/EM-DeSTIN/EMDestin/src/destin.cpp -o CMakeFiles/destinalt.dir/src/destin.cpp.s

EMDestin/CMakeFiles/destinalt.dir/src/destin.cpp.o.requires:
.PHONY : EMDestin/CMakeFiles/destinalt.dir/src/destin.cpp.o.requires

EMDestin/CMakeFiles/destinalt.dir/src/destin.cpp.o.provides: EMDestin/CMakeFiles/destinalt.dir/src/destin.cpp.o.requires
	$(MAKE) -f EMDestin/CMakeFiles/destinalt.dir/build.make EMDestin/CMakeFiles/destinalt.dir/src/destin.cpp.o.provides.build
.PHONY : EMDestin/CMakeFiles/destinalt.dir/src/destin.cpp.o.provides

EMDestin/CMakeFiles/destinalt.dir/src/destin.cpp.o.provides.build: EMDestin/CMakeFiles/destinalt.dir/src/destin.cpp.o

EMDestin/CMakeFiles/destinalt.dir/src/node.cpp.o: EMDestin/CMakeFiles/destinalt.dir/flags.make
EMDestin/CMakeFiles/destinalt.dir/src/node.cpp.o: EMDestin/src/node.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/opencog/Desktop/Min/EM-DeSTIN/CMakeFiles $(CMAKE_PROGRESS_2)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object EMDestin/CMakeFiles/destinalt.dir/src/node.cpp.o"
	cd /home/opencog/Desktop/Min/EM-DeSTIN/EMDestin && /usr/bin/g++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/destinalt.dir/src/node.cpp.o -c /home/opencog/Desktop/Min/EM-DeSTIN/EMDestin/src/node.cpp

EMDestin/CMakeFiles/destinalt.dir/src/node.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/destinalt.dir/src/node.cpp.i"
	cd /home/opencog/Desktop/Min/EM-DeSTIN/EMDestin && /usr/bin/g++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/opencog/Desktop/Min/EM-DeSTIN/EMDestin/src/node.cpp > CMakeFiles/destinalt.dir/src/node.cpp.i

EMDestin/CMakeFiles/destinalt.dir/src/node.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/destinalt.dir/src/node.cpp.s"
	cd /home/opencog/Desktop/Min/EM-DeSTIN/EMDestin && /usr/bin/g++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/opencog/Desktop/Min/EM-DeSTIN/EMDestin/src/node.cpp -o CMakeFiles/destinalt.dir/src/node.cpp.s

EMDestin/CMakeFiles/destinalt.dir/src/node.cpp.o.requires:
.PHONY : EMDestin/CMakeFiles/destinalt.dir/src/node.cpp.o.requires

EMDestin/CMakeFiles/destinalt.dir/src/node.cpp.o.provides: EMDestin/CMakeFiles/destinalt.dir/src/node.cpp.o.requires
	$(MAKE) -f EMDestin/CMakeFiles/destinalt.dir/build.make EMDestin/CMakeFiles/destinalt.dir/src/node.cpp.o.provides.build
.PHONY : EMDestin/CMakeFiles/destinalt.dir/src/node.cpp.o.provides

EMDestin/CMakeFiles/destinalt.dir/src/node.cpp.o.provides.build: EMDestin/CMakeFiles/destinalt.dir/src/node.cpp.o

EMDestin/CMakeFiles/destinalt.dir/src/util.cpp.o: EMDestin/CMakeFiles/destinalt.dir/flags.make
EMDestin/CMakeFiles/destinalt.dir/src/util.cpp.o: EMDestin/src/util.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/opencog/Desktop/Min/EM-DeSTIN/CMakeFiles $(CMAKE_PROGRESS_3)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object EMDestin/CMakeFiles/destinalt.dir/src/util.cpp.o"
	cd /home/opencog/Desktop/Min/EM-DeSTIN/EMDestin && /usr/bin/g++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/destinalt.dir/src/util.cpp.o -c /home/opencog/Desktop/Min/EM-DeSTIN/EMDestin/src/util.cpp

EMDestin/CMakeFiles/destinalt.dir/src/util.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/destinalt.dir/src/util.cpp.i"
	cd /home/opencog/Desktop/Min/EM-DeSTIN/EMDestin && /usr/bin/g++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/opencog/Desktop/Min/EM-DeSTIN/EMDestin/src/util.cpp > CMakeFiles/destinalt.dir/src/util.cpp.i

EMDestin/CMakeFiles/destinalt.dir/src/util.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/destinalt.dir/src/util.cpp.s"
	cd /home/opencog/Desktop/Min/EM-DeSTIN/EMDestin && /usr/bin/g++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/opencog/Desktop/Min/EM-DeSTIN/EMDestin/src/util.cpp -o CMakeFiles/destinalt.dir/src/util.cpp.s

EMDestin/CMakeFiles/destinalt.dir/src/util.cpp.o.requires:
.PHONY : EMDestin/CMakeFiles/destinalt.dir/src/util.cpp.o.requires

EMDestin/CMakeFiles/destinalt.dir/src/util.cpp.o.provides: EMDestin/CMakeFiles/destinalt.dir/src/util.cpp.o.requires
	$(MAKE) -f EMDestin/CMakeFiles/destinalt.dir/build.make EMDestin/CMakeFiles/destinalt.dir/src/util.cpp.o.provides.build
.PHONY : EMDestin/CMakeFiles/destinalt.dir/src/util.cpp.o.provides

EMDestin/CMakeFiles/destinalt.dir/src/util.cpp.o.provides.build: EMDestin/CMakeFiles/destinalt.dir/src/util.cpp.o

EMDestin/CMakeFiles/destinalt.dir/src/cent_image_gen.cpp.o: EMDestin/CMakeFiles/destinalt.dir/flags.make
EMDestin/CMakeFiles/destinalt.dir/src/cent_image_gen.cpp.o: EMDestin/src/cent_image_gen.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/opencog/Desktop/Min/EM-DeSTIN/CMakeFiles $(CMAKE_PROGRESS_4)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object EMDestin/CMakeFiles/destinalt.dir/src/cent_image_gen.cpp.o"
	cd /home/opencog/Desktop/Min/EM-DeSTIN/EMDestin && /usr/bin/g++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/destinalt.dir/src/cent_image_gen.cpp.o -c /home/opencog/Desktop/Min/EM-DeSTIN/EMDestin/src/cent_image_gen.cpp

EMDestin/CMakeFiles/destinalt.dir/src/cent_image_gen.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/destinalt.dir/src/cent_image_gen.cpp.i"
	cd /home/opencog/Desktop/Min/EM-DeSTIN/EMDestin && /usr/bin/g++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/opencog/Desktop/Min/EM-DeSTIN/EMDestin/src/cent_image_gen.cpp > CMakeFiles/destinalt.dir/src/cent_image_gen.cpp.i

EMDestin/CMakeFiles/destinalt.dir/src/cent_image_gen.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/destinalt.dir/src/cent_image_gen.cpp.s"
	cd /home/opencog/Desktop/Min/EM-DeSTIN/EMDestin && /usr/bin/g++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/opencog/Desktop/Min/EM-DeSTIN/EMDestin/src/cent_image_gen.cpp -o CMakeFiles/destinalt.dir/src/cent_image_gen.cpp.s

EMDestin/CMakeFiles/destinalt.dir/src/cent_image_gen.cpp.o.requires:
.PHONY : EMDestin/CMakeFiles/destinalt.dir/src/cent_image_gen.cpp.o.requires

EMDestin/CMakeFiles/destinalt.dir/src/cent_image_gen.cpp.o.provides: EMDestin/CMakeFiles/destinalt.dir/src/cent_image_gen.cpp.o.requires
	$(MAKE) -f EMDestin/CMakeFiles/destinalt.dir/build.make EMDestin/CMakeFiles/destinalt.dir/src/cent_image_gen.cpp.o.provides.build
.PHONY : EMDestin/CMakeFiles/destinalt.dir/src/cent_image_gen.cpp.o.provides

EMDestin/CMakeFiles/destinalt.dir/src/cent_image_gen.cpp.o.provides.build: EMDestin/CMakeFiles/destinalt.dir/src/cent_image_gen.cpp.o

EMDestin/CMakeFiles/destinalt.dir/src/em.cpp.o: EMDestin/CMakeFiles/destinalt.dir/flags.make
EMDestin/CMakeFiles/destinalt.dir/src/em.cpp.o: EMDestin/src/em.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/opencog/Desktop/Min/EM-DeSTIN/CMakeFiles $(CMAKE_PROGRESS_5)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object EMDestin/CMakeFiles/destinalt.dir/src/em.cpp.o"
	cd /home/opencog/Desktop/Min/EM-DeSTIN/EMDestin && /usr/bin/g++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/destinalt.dir/src/em.cpp.o -c /home/opencog/Desktop/Min/EM-DeSTIN/EMDestin/src/em.cpp

EMDestin/CMakeFiles/destinalt.dir/src/em.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/destinalt.dir/src/em.cpp.i"
	cd /home/opencog/Desktop/Min/EM-DeSTIN/EMDestin && /usr/bin/g++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/opencog/Desktop/Min/EM-DeSTIN/EMDestin/src/em.cpp > CMakeFiles/destinalt.dir/src/em.cpp.i

EMDestin/CMakeFiles/destinalt.dir/src/em.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/destinalt.dir/src/em.cpp.s"
	cd /home/opencog/Desktop/Min/EM-DeSTIN/EMDestin && /usr/bin/g++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/opencog/Desktop/Min/EM-DeSTIN/EMDestin/src/em.cpp -o CMakeFiles/destinalt.dir/src/em.cpp.s

EMDestin/CMakeFiles/destinalt.dir/src/em.cpp.o.requires:
.PHONY : EMDestin/CMakeFiles/destinalt.dir/src/em.cpp.o.requires

EMDestin/CMakeFiles/destinalt.dir/src/em.cpp.o.provides: EMDestin/CMakeFiles/destinalt.dir/src/em.cpp.o.requires
	$(MAKE) -f EMDestin/CMakeFiles/destinalt.dir/build.make EMDestin/CMakeFiles/destinalt.dir/src/em.cpp.o.provides.build
.PHONY : EMDestin/CMakeFiles/destinalt.dir/src/em.cpp.o.provides

EMDestin/CMakeFiles/destinalt.dir/src/em.cpp.o.provides.build: EMDestin/CMakeFiles/destinalt.dir/src/em.cpp.o

EMDestin/CMakeFiles/destinalt.dir/src/learn_strats.cpp.o: EMDestin/CMakeFiles/destinalt.dir/flags.make
EMDestin/CMakeFiles/destinalt.dir/src/learn_strats.cpp.o: EMDestin/src/learn_strats.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/opencog/Desktop/Min/EM-DeSTIN/CMakeFiles $(CMAKE_PROGRESS_6)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object EMDestin/CMakeFiles/destinalt.dir/src/learn_strats.cpp.o"
	cd /home/opencog/Desktop/Min/EM-DeSTIN/EMDestin && /usr/bin/g++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/destinalt.dir/src/learn_strats.cpp.o -c /home/opencog/Desktop/Min/EM-DeSTIN/EMDestin/src/learn_strats.cpp

EMDestin/CMakeFiles/destinalt.dir/src/learn_strats.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/destinalt.dir/src/learn_strats.cpp.i"
	cd /home/opencog/Desktop/Min/EM-DeSTIN/EMDestin && /usr/bin/g++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/opencog/Desktop/Min/EM-DeSTIN/EMDestin/src/learn_strats.cpp > CMakeFiles/destinalt.dir/src/learn_strats.cpp.i

EMDestin/CMakeFiles/destinalt.dir/src/learn_strats.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/destinalt.dir/src/learn_strats.cpp.s"
	cd /home/opencog/Desktop/Min/EM-DeSTIN/EMDestin && /usr/bin/g++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/opencog/Desktop/Min/EM-DeSTIN/EMDestin/src/learn_strats.cpp -o CMakeFiles/destinalt.dir/src/learn_strats.cpp.s

EMDestin/CMakeFiles/destinalt.dir/src/learn_strats.cpp.o.requires:
.PHONY : EMDestin/CMakeFiles/destinalt.dir/src/learn_strats.cpp.o.requires

EMDestin/CMakeFiles/destinalt.dir/src/learn_strats.cpp.o.provides: EMDestin/CMakeFiles/destinalt.dir/src/learn_strats.cpp.o.requires
	$(MAKE) -f EMDestin/CMakeFiles/destinalt.dir/build.make EMDestin/CMakeFiles/destinalt.dir/src/learn_strats.cpp.o.provides.build
.PHONY : EMDestin/CMakeFiles/destinalt.dir/src/learn_strats.cpp.o.provides

EMDestin/CMakeFiles/destinalt.dir/src/learn_strats.cpp.o.provides.build: EMDestin/CMakeFiles/destinalt.dir/src/learn_strats.cpp.o

EMDestin/CMakeFiles/destinalt.dir/src/belief_transform.cpp.o: EMDestin/CMakeFiles/destinalt.dir/flags.make
EMDestin/CMakeFiles/destinalt.dir/src/belief_transform.cpp.o: EMDestin/src/belief_transform.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/opencog/Desktop/Min/EM-DeSTIN/CMakeFiles $(CMAKE_PROGRESS_7)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object EMDestin/CMakeFiles/destinalt.dir/src/belief_transform.cpp.o"
	cd /home/opencog/Desktop/Min/EM-DeSTIN/EMDestin && /usr/bin/g++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/destinalt.dir/src/belief_transform.cpp.o -c /home/opencog/Desktop/Min/EM-DeSTIN/EMDestin/src/belief_transform.cpp

EMDestin/CMakeFiles/destinalt.dir/src/belief_transform.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/destinalt.dir/src/belief_transform.cpp.i"
	cd /home/opencog/Desktop/Min/EM-DeSTIN/EMDestin && /usr/bin/g++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/opencog/Desktop/Min/EM-DeSTIN/EMDestin/src/belief_transform.cpp > CMakeFiles/destinalt.dir/src/belief_transform.cpp.i

EMDestin/CMakeFiles/destinalt.dir/src/belief_transform.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/destinalt.dir/src/belief_transform.cpp.s"
	cd /home/opencog/Desktop/Min/EM-DeSTIN/EMDestin && /usr/bin/g++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/opencog/Desktop/Min/EM-DeSTIN/EMDestin/src/belief_transform.cpp -o CMakeFiles/destinalt.dir/src/belief_transform.cpp.s

EMDestin/CMakeFiles/destinalt.dir/src/belief_transform.cpp.o.requires:
.PHONY : EMDestin/CMakeFiles/destinalt.dir/src/belief_transform.cpp.o.requires

EMDestin/CMakeFiles/destinalt.dir/src/belief_transform.cpp.o.provides: EMDestin/CMakeFiles/destinalt.dir/src/belief_transform.cpp.o.requires
	$(MAKE) -f EMDestin/CMakeFiles/destinalt.dir/build.make EMDestin/CMakeFiles/destinalt.dir/src/belief_transform.cpp.o.provides.build
.PHONY : EMDestin/CMakeFiles/destinalt.dir/src/belief_transform.cpp.o.provides

EMDestin/CMakeFiles/destinalt.dir/src/belief_transform.cpp.o.provides.build: EMDestin/CMakeFiles/destinalt.dir/src/belief_transform.cpp.o

# Object files for target destinalt
destinalt_OBJECTS = \
"CMakeFiles/destinalt.dir/src/destin.cpp.o" \
"CMakeFiles/destinalt.dir/src/node.cpp.o" \
"CMakeFiles/destinalt.dir/src/util.cpp.o" \
"CMakeFiles/destinalt.dir/src/cent_image_gen.cpp.o" \
"CMakeFiles/destinalt.dir/src/em.cpp.o" \
"CMakeFiles/destinalt.dir/src/learn_strats.cpp.o" \
"CMakeFiles/destinalt.dir/src/belief_transform.cpp.o"

# External object files for target destinalt
destinalt_EXTERNAL_OBJECTS =

EMDestin/libdestinalt.so: EMDestin/CMakeFiles/destinalt.dir/src/destin.cpp.o
EMDestin/libdestinalt.so: EMDestin/CMakeFiles/destinalt.dir/src/node.cpp.o
EMDestin/libdestinalt.so: EMDestin/CMakeFiles/destinalt.dir/src/util.cpp.o
EMDestin/libdestinalt.so: EMDestin/CMakeFiles/destinalt.dir/src/cent_image_gen.cpp.o
EMDestin/libdestinalt.so: EMDestin/CMakeFiles/destinalt.dir/src/em.cpp.o
EMDestin/libdestinalt.so: EMDestin/CMakeFiles/destinalt.dir/src/learn_strats.cpp.o
EMDestin/libdestinalt.so: EMDestin/CMakeFiles/destinalt.dir/src/belief_transform.cpp.o
EMDestin/libdestinalt.so: EMDestin/CMakeFiles/destinalt.dir/build.make
EMDestin/libdestinalt.so: EMDestin/CMakeFiles/destinalt.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX shared library libdestinalt.so"
	cd /home/opencog/Desktop/Min/EM-DeSTIN/EMDestin && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/destinalt.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
EMDestin/CMakeFiles/destinalt.dir/build: EMDestin/libdestinalt.so
.PHONY : EMDestin/CMakeFiles/destinalt.dir/build

EMDestin/CMakeFiles/destinalt.dir/requires: EMDestin/CMakeFiles/destinalt.dir/src/destin.cpp.o.requires
EMDestin/CMakeFiles/destinalt.dir/requires: EMDestin/CMakeFiles/destinalt.dir/src/node.cpp.o.requires
EMDestin/CMakeFiles/destinalt.dir/requires: EMDestin/CMakeFiles/destinalt.dir/src/util.cpp.o.requires
EMDestin/CMakeFiles/destinalt.dir/requires: EMDestin/CMakeFiles/destinalt.dir/src/cent_image_gen.cpp.o.requires
EMDestin/CMakeFiles/destinalt.dir/requires: EMDestin/CMakeFiles/destinalt.dir/src/em.cpp.o.requires
EMDestin/CMakeFiles/destinalt.dir/requires: EMDestin/CMakeFiles/destinalt.dir/src/learn_strats.cpp.o.requires
EMDestin/CMakeFiles/destinalt.dir/requires: EMDestin/CMakeFiles/destinalt.dir/src/belief_transform.cpp.o.requires
.PHONY : EMDestin/CMakeFiles/destinalt.dir/requires

EMDestin/CMakeFiles/destinalt.dir/clean:
	cd /home/opencog/Desktop/Min/EM-DeSTIN/EMDestin && $(CMAKE_COMMAND) -P CMakeFiles/destinalt.dir/cmake_clean.cmake
.PHONY : EMDestin/CMakeFiles/destinalt.dir/clean

EMDestin/CMakeFiles/destinalt.dir/depend:
	cd /home/opencog/Desktop/Min/EM-DeSTIN && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/opencog/Desktop/Min/EM-DeSTIN /home/opencog/Desktop/Min/EM-DeSTIN/EMDestin /home/opencog/Desktop/Min/EM-DeSTIN /home/opencog/Desktop/Min/EM-DeSTIN/EMDestin /home/opencog/Desktop/Min/EM-DeSTIN/EMDestin/CMakeFiles/destinalt.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : EMDestin/CMakeFiles/destinalt.dir/depend

