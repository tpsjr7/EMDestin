# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

# Default target executed when no arguments are given to make.
default_target: all
.PHONY : default_target

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

#=============================================================================
# Targets provided globally by CMake.

# Special rule for the target edit_cache
edit_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake cache editor..."
	/usr/bin/ccmake -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : edit_cache

# Special rule for the target edit_cache
edit_cache/fast: edit_cache
.PHONY : edit_cache/fast

# Special rule for the target rebuild_cache
rebuild_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake to regenerate build system..."
	/usr/bin/cmake -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : rebuild_cache

# Special rule for the target rebuild_cache
rebuild_cache/fast: rebuild_cache
.PHONY : rebuild_cache/fast

# The main all target
all: cmake_check_build_system
	$(CMAKE_COMMAND) -E cmake_progress_start /home/opencog/Desktop/Min/EM-DeSTIN/CMakeFiles /home/opencog/Desktop/Min/EM-DeSTIN/CMakeFiles/progress.marks
	$(MAKE) -f CMakeFiles/Makefile2 all
	$(CMAKE_COMMAND) -E cmake_progress_start /home/opencog/Desktop/Min/EM-DeSTIN/CMakeFiles 0
.PHONY : all

# The main clean target
clean:
	$(MAKE) -f CMakeFiles/Makefile2 clean
.PHONY : clean

# The main clean target
clean/fast: clean
.PHONY : clean/fast

# Prepare targets for installation.
preinstall: all
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall

# Prepare targets for installation.
preinstall/fast:
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall/fast

# clear depends
depend:
	$(CMAKE_COMMAND) -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 1
.PHONY : depend

#=============================================================================
# Target rules for targets named common

# Build rule for target.
common: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 common
.PHONY : common

# fast build rule for target.
common/fast:
	$(MAKE) -f Common/CMakeFiles/common.dir/build.make Common/CMakeFiles/common.dir/build
.PHONY : common/fast

#=============================================================================
# Target rules for targets named testCifarSource

# Build rule for target.
testCifarSource: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 testCifarSource
.PHONY : testCifarSource

# fast build rule for target.
testCifarSource/fast:
	$(MAKE) -f Common/CMakeFiles/testCifarSource.dir/build.make Common/CMakeFiles/testCifarSource.dir/build
.PHONY : testCifarSource/fast

#=============================================================================
# Target rules for targets named testDestinNetworkAlt

# Build rule for target.
testDestinNetworkAlt: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 testDestinNetworkAlt
.PHONY : testDestinNetworkAlt

# fast build rule for target.
testDestinNetworkAlt/fast:
	$(MAKE) -f Common/CMakeFiles/testDestinNetworkAlt.dir/build.make Common/CMakeFiles/testDestinNetworkAlt.dir/build
.PHONY : testDestinNetworkAlt/fast

#=============================================================================
# Target rules for targets named destin

# Build rule for target.
destin: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 destin
.PHONY : destin

# fast build rule for target.
destin/fast:
	$(MAKE) -f EMDestin/CMakeFiles/destin.dir/build.make EMDestin/CMakeFiles/destin.dir/build
.PHONY : destin/fast

#=============================================================================
# Target rules for targets named destinalt

# Build rule for target.
destinalt: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 destinalt
.PHONY : destinalt

# fast build rule for target.
destinalt/fast:
	$(MAKE) -f EMDestin/CMakeFiles/destinalt.dir/build.make EMDestin/CMakeFiles/destinalt.dir/build
.PHONY : destinalt/fast

#=============================================================================
# Target rules for targets named destinalt_test

# Build rule for target.
destinalt_test: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 destinalt_test
.PHONY : destinalt_test

# fast build rule for target.
destinalt_test/fast:
	$(MAKE) -f EMDestin/CMakeFiles/destinalt_test.dir/build.make EMDestin/CMakeFiles/destinalt_test.dir/build
.PHONY : destinalt_test/fast

#=============================================================================
# Target rules for targets named test

# Build rule for target.
test: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 test
.PHONY : test

# fast build rule for target.
test/fast:
	$(MAKE) -f EMDestin/CMakeFiles/test.dir/build.make EMDestin/CMakeFiles/test.dir/build
.PHONY : test/fast

#=============================================================================
# Target rules for targets named som

# Build rule for target.
som: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 som
.PHONY : som

# fast build rule for target.
som/fast:
	$(MAKE) -f SOM/CMakeFiles/som.dir/build.make SOM/CMakeFiles/som.dir/build
.PHONY : som/fast

#=============================================================================
# Target rules for targets named testSOM

# Build rule for target.
testSOM: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 testSOM
.PHONY : testSOM

# fast build rule for target.
testSOM/fast:
	$(MAKE) -f SOM/CMakeFiles/testSOM.dir/build.make SOM/CMakeFiles/testSOM.dir/build
.PHONY : testSOM/fast

#=============================================================================
# Target rules for targets named stereovision

# Build rule for target.
stereovision: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 stereovision
.PHONY : stereovision

# fast build rule for target.
stereovision/fast:
	$(MAKE) -f EMdestin_test/CMakeFiles/stereovision.dir/build.make EMdestin_test/CMakeFiles/stereovision.dir/build
.PHONY : stereovision/fast

#=============================================================================
# Target rules for targets named test2

# Build rule for target.
test2: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 test2
.PHONY : test2

# fast build rule for target.
test2/fast:
	$(MAKE) -f EMdestin_test/CMakeFiles/test2.dir/build.make EMdestin_test/CMakeFiles/test2.dir/build
.PHONY : test2/fast

# Help Target
help:
	@echo "The following are some of the valid targets for this Makefile:"
	@echo "... all (the default if no target is provided)"
	@echo "... clean"
	@echo "... depend"
	@echo "... edit_cache"
	@echo "... rebuild_cache"
	@echo "... common"
	@echo "... testCifarSource"
	@echo "... testDestinNetworkAlt"
	@echo "... destin"
	@echo "... destinalt"
	@echo "... destinalt_test"
	@echo "... test"
	@echo "... som"
	@echo "... testSOM"
	@echo "... stereovision"
	@echo "... test2"
.PHONY : help



#=============================================================================
# Special targets to cleanup operation of make.

# Special rule to run CMake to check the build system integrity.
# No rule that depends on this can have commands that come from listfiles
# because they might be regenerated.
cmake_check_build_system:
	$(CMAKE_COMMAND) -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 0
.PHONY : cmake_check_build_system

