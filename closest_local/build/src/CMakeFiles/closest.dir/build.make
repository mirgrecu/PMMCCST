# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.30

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /Users/mgrecu/homebrew/Cellar/cmake/3.30.2/bin/cmake

# The command to remove a file.
RM = /Users/mgrecu/homebrew/Cellar/cmake/3.30.2/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/mgrecu/PMMCCST/closest

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/mgrecu/PMMCCST/closest/build

# Include any dependencies generated for this target.
include src/CMakeFiles/closest.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include src/CMakeFiles/closest.dir/compiler_depend.make

# Include the progress variables for this target.
include src/CMakeFiles/closest.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/closest.dir/flags.make

src/CMakeFiles/closest.dir/auxf.c.o: src/CMakeFiles/closest.dir/flags.make
src/CMakeFiles/closest.dir/auxf.c.o: /Users/mgrecu/PMMCCST/closest/src/auxf.c
src/CMakeFiles/closest.dir/auxf.c.o: src/CMakeFiles/closest.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/mgrecu/PMMCCST/closest/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object src/CMakeFiles/closest.dir/auxf.c.o"
	cd /Users/mgrecu/PMMCCST/closest/build/src && /Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT src/CMakeFiles/closest.dir/auxf.c.o -MF CMakeFiles/closest.dir/auxf.c.o.d -o CMakeFiles/closest.dir/auxf.c.o -c /Users/mgrecu/PMMCCST/closest/src/auxf.c

src/CMakeFiles/closest.dir/auxf.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing C source to CMakeFiles/closest.dir/auxf.c.i"
	cd /Users/mgrecu/PMMCCST/closest/build/src && /Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/mgrecu/PMMCCST/closest/src/auxf.c > CMakeFiles/closest.dir/auxf.c.i

src/CMakeFiles/closest.dir/auxf.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling C source to assembly CMakeFiles/closest.dir/auxf.c.s"
	cd /Users/mgrecu/PMMCCST/closest/build/src && /Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/mgrecu/PMMCCST/closest/src/auxf.c -o CMakeFiles/closest.dir/auxf.c.s

src/CMakeFiles/closest.dir/dist.c.o: src/CMakeFiles/closest.dir/flags.make
src/CMakeFiles/closest.dir/dist.c.o: /Users/mgrecu/PMMCCST/closest/src/dist.c
src/CMakeFiles/closest.dir/dist.c.o: src/CMakeFiles/closest.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/mgrecu/PMMCCST/closest/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building C object src/CMakeFiles/closest.dir/dist.c.o"
	cd /Users/mgrecu/PMMCCST/closest/build/src && /Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT src/CMakeFiles/closest.dir/dist.c.o -MF CMakeFiles/closest.dir/dist.c.o.d -o CMakeFiles/closest.dir/dist.c.o -c /Users/mgrecu/PMMCCST/closest/src/dist.c

src/CMakeFiles/closest.dir/dist.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing C source to CMakeFiles/closest.dir/dist.c.i"
	cd /Users/mgrecu/PMMCCST/closest/build/src && /Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/mgrecu/PMMCCST/closest/src/dist.c > CMakeFiles/closest.dir/dist.c.i

src/CMakeFiles/closest.dir/dist.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling C source to assembly CMakeFiles/closest.dir/dist.c.s"
	cd /Users/mgrecu/PMMCCST/closest/build/src && /Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/mgrecu/PMMCCST/closest/src/dist.c -o CMakeFiles/closest.dir/dist.c.s

src/CMakeFiles/closest.dir/cull.c.o: src/CMakeFiles/closest.dir/flags.make
src/CMakeFiles/closest.dir/cull.c.o: /Users/mgrecu/PMMCCST/closest/src/cull.c
src/CMakeFiles/closest.dir/cull.c.o: src/CMakeFiles/closest.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/mgrecu/PMMCCST/closest/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building C object src/CMakeFiles/closest.dir/cull.c.o"
	cd /Users/mgrecu/PMMCCST/closest/build/src && /Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT src/CMakeFiles/closest.dir/cull.c.o -MF CMakeFiles/closest.dir/cull.c.o.d -o CMakeFiles/closest.dir/cull.c.o -c /Users/mgrecu/PMMCCST/closest/src/cull.c

src/CMakeFiles/closest.dir/cull.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing C source to CMakeFiles/closest.dir/cull.c.i"
	cd /Users/mgrecu/PMMCCST/closest/build/src && /Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/mgrecu/PMMCCST/closest/src/cull.c > CMakeFiles/closest.dir/cull.c.i

src/CMakeFiles/closest.dir/cull.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling C source to assembly CMakeFiles/closest.dir/cull.c.s"
	cd /Users/mgrecu/PMMCCST/closest/build/src && /Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/mgrecu/PMMCCST/closest/src/cull.c -o CMakeFiles/closest.dir/cull.c.s

src/CMakeFiles/closest.dir/cell.c.o: src/CMakeFiles/closest.dir/flags.make
src/CMakeFiles/closest.dir/cell.c.o: /Users/mgrecu/PMMCCST/closest/src/cell.c
src/CMakeFiles/closest.dir/cell.c.o: src/CMakeFiles/closest.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/mgrecu/PMMCCST/closest/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building C object src/CMakeFiles/closest.dir/cell.c.o"
	cd /Users/mgrecu/PMMCCST/closest/build/src && /Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT src/CMakeFiles/closest.dir/cell.c.o -MF CMakeFiles/closest.dir/cell.c.o.d -o CMakeFiles/closest.dir/cell.c.o -c /Users/mgrecu/PMMCCST/closest/src/cell.c

src/CMakeFiles/closest.dir/cell.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing C source to CMakeFiles/closest.dir/cell.c.i"
	cd /Users/mgrecu/PMMCCST/closest/build/src && /Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/mgrecu/PMMCCST/closest/src/cell.c > CMakeFiles/closest.dir/cell.c.i

src/CMakeFiles/closest.dir/cell.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling C source to assembly CMakeFiles/closest.dir/cell.c.s"
	cd /Users/mgrecu/PMMCCST/closest/build/src && /Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/mgrecu/PMMCCST/closest/src/cell.c -o CMakeFiles/closest.dir/cell.c.s

src/CMakeFiles/closest.dir/brut.c.o: src/CMakeFiles/closest.dir/flags.make
src/CMakeFiles/closest.dir/brut.c.o: /Users/mgrecu/PMMCCST/closest/src/brut.c
src/CMakeFiles/closest.dir/brut.c.o: src/CMakeFiles/closest.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/mgrecu/PMMCCST/closest/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building C object src/CMakeFiles/closest.dir/brut.c.o"
	cd /Users/mgrecu/PMMCCST/closest/build/src && /Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT src/CMakeFiles/closest.dir/brut.c.o -MF CMakeFiles/closest.dir/brut.c.o.d -o CMakeFiles/closest.dir/brut.c.o -c /Users/mgrecu/PMMCCST/closest/src/brut.c

src/CMakeFiles/closest.dir/brut.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing C source to CMakeFiles/closest.dir/brut.c.i"
	cd /Users/mgrecu/PMMCCST/closest/build/src && /Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/mgrecu/PMMCCST/closest/src/brut.c > CMakeFiles/closest.dir/brut.c.i

src/CMakeFiles/closest.dir/brut.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling C source to assembly CMakeFiles/closest.dir/brut.c.s"
	cd /Users/mgrecu/PMMCCST/closest/build/src && /Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/mgrecu/PMMCCST/closest/src/brut.c -o CMakeFiles/closest.dir/brut.c.s

src/CMakeFiles/closest.dir/tree.c.o: src/CMakeFiles/closest.dir/flags.make
src/CMakeFiles/closest.dir/tree.c.o: /Users/mgrecu/PMMCCST/closest/src/tree.c
src/CMakeFiles/closest.dir/tree.c.o: src/CMakeFiles/closest.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/mgrecu/PMMCCST/closest/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building C object src/CMakeFiles/closest.dir/tree.c.o"
	cd /Users/mgrecu/PMMCCST/closest/build/src && /Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT src/CMakeFiles/closest.dir/tree.c.o -MF CMakeFiles/closest.dir/tree.c.o.d -o CMakeFiles/closest.dir/tree.c.o -c /Users/mgrecu/PMMCCST/closest/src/tree.c

src/CMakeFiles/closest.dir/tree.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing C source to CMakeFiles/closest.dir/tree.c.i"
	cd /Users/mgrecu/PMMCCST/closest/build/src && /Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/mgrecu/PMMCCST/closest/src/tree.c > CMakeFiles/closest.dir/tree.c.i

src/CMakeFiles/closest.dir/tree.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling C source to assembly CMakeFiles/closest.dir/tree.c.s"
	cd /Users/mgrecu/PMMCCST/closest/build/src && /Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/mgrecu/PMMCCST/closest/src/tree.c -o CMakeFiles/closest.dir/tree.c.s

# Object files for target closest
closest_OBJECTS = \
"CMakeFiles/closest.dir/auxf.c.o" \
"CMakeFiles/closest.dir/dist.c.o" \
"CMakeFiles/closest.dir/cull.c.o" \
"CMakeFiles/closest.dir/cell.c.o" \
"CMakeFiles/closest.dir/brut.c.o" \
"CMakeFiles/closest.dir/tree.c.o"

# External object files for target closest
closest_EXTERNAL_OBJECTS =

src/libclosest.dylib: src/CMakeFiles/closest.dir/auxf.c.o
src/libclosest.dylib: src/CMakeFiles/closest.dir/dist.c.o
src/libclosest.dylib: src/CMakeFiles/closest.dir/cull.c.o
src/libclosest.dylib: src/CMakeFiles/closest.dir/cell.c.o
src/libclosest.dylib: src/CMakeFiles/closest.dir/brut.c.o
src/libclosest.dylib: src/CMakeFiles/closest.dir/tree.c.o
src/libclosest.dylib: src/CMakeFiles/closest.dir/build.make
src/libclosest.dylib: src/CMakeFiles/closest.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/mgrecu/PMMCCST/closest/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Linking C shared library libclosest.dylib"
	cd /Users/mgrecu/PMMCCST/closest/build/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/closest.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/CMakeFiles/closest.dir/build: src/libclosest.dylib
.PHONY : src/CMakeFiles/closest.dir/build

src/CMakeFiles/closest.dir/clean:
	cd /Users/mgrecu/PMMCCST/closest/build/src && $(CMAKE_COMMAND) -P CMakeFiles/closest.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/closest.dir/clean

src/CMakeFiles/closest.dir/depend:
	cd /Users/mgrecu/PMMCCST/closest/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/mgrecu/PMMCCST/closest /Users/mgrecu/PMMCCST/closest/src /Users/mgrecu/PMMCCST/closest/build /Users/mgrecu/PMMCCST/closest/build/src /Users/mgrecu/PMMCCST/closest/build/src/CMakeFiles/closest.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : src/CMakeFiles/closest.dir/depend

