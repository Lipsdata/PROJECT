# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.28

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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/duludulu/Documents/hls/source

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/duludulu/Documents/hls/build/linux

# Include any dependencies generated for this target.
include CMakeFiles/testlenet.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/testlenet.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/testlenet.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/testlenet.dir/flags.make

CMakeFiles/testlenet.dir/mt19937ar.cpp.o: CMakeFiles/testlenet.dir/flags.make
CMakeFiles/testlenet.dir/mt19937ar.cpp.o: /home/duludulu/Documents/hls/source/mt19937ar.cpp
CMakeFiles/testlenet.dir/mt19937ar.cpp.o: CMakeFiles/testlenet.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/duludulu/Documents/hls/build/linux/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/testlenet.dir/mt19937ar.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/testlenet.dir/mt19937ar.cpp.o -MF CMakeFiles/testlenet.dir/mt19937ar.cpp.o.d -o CMakeFiles/testlenet.dir/mt19937ar.cpp.o -c /home/duludulu/Documents/hls/source/mt19937ar.cpp

CMakeFiles/testlenet.dir/mt19937ar.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/testlenet.dir/mt19937ar.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/duludulu/Documents/hls/source/mt19937ar.cpp > CMakeFiles/testlenet.dir/mt19937ar.cpp.i

CMakeFiles/testlenet.dir/mt19937ar.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/testlenet.dir/mt19937ar.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/duludulu/Documents/hls/source/mt19937ar.cpp -o CMakeFiles/testlenet.dir/mt19937ar.cpp.s

CMakeFiles/testlenet.dir/stb_image_impl.cpp.o: CMakeFiles/testlenet.dir/flags.make
CMakeFiles/testlenet.dir/stb_image_impl.cpp.o: /home/duludulu/Documents/hls/source/stb_image_impl.cpp
CMakeFiles/testlenet.dir/stb_image_impl.cpp.o: CMakeFiles/testlenet.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/duludulu/Documents/hls/build/linux/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/testlenet.dir/stb_image_impl.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/testlenet.dir/stb_image_impl.cpp.o -MF CMakeFiles/testlenet.dir/stb_image_impl.cpp.o.d -o CMakeFiles/testlenet.dir/stb_image_impl.cpp.o -c /home/duludulu/Documents/hls/source/stb_image_impl.cpp

CMakeFiles/testlenet.dir/stb_image_impl.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/testlenet.dir/stb_image_impl.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/duludulu/Documents/hls/source/stb_image_impl.cpp > CMakeFiles/testlenet.dir/stb_image_impl.cpp.i

CMakeFiles/testlenet.dir/stb_image_impl.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/testlenet.dir/stb_image_impl.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/duludulu/Documents/hls/source/stb_image_impl.cpp -o CMakeFiles/testlenet.dir/stb_image_impl.cpp.s

CMakeFiles/testlenet.dir/stdafx.cpp.o: CMakeFiles/testlenet.dir/flags.make
CMakeFiles/testlenet.dir/stdafx.cpp.o: /home/duludulu/Documents/hls/source/stdafx.cpp
CMakeFiles/testlenet.dir/stdafx.cpp.o: CMakeFiles/testlenet.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/duludulu/Documents/hls/build/linux/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/testlenet.dir/stdafx.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/testlenet.dir/stdafx.cpp.o -MF CMakeFiles/testlenet.dir/stdafx.cpp.o.d -o CMakeFiles/testlenet.dir/stdafx.cpp.o -c /home/duludulu/Documents/hls/source/stdafx.cpp

CMakeFiles/testlenet.dir/stdafx.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/testlenet.dir/stdafx.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/duludulu/Documents/hls/source/stdafx.cpp > CMakeFiles/testlenet.dir/stdafx.cpp.i

CMakeFiles/testlenet.dir/stdafx.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/testlenet.dir/stdafx.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/duludulu/Documents/hls/source/stdafx.cpp -o CMakeFiles/testlenet.dir/stdafx.cpp.s

CMakeFiles/testlenet.dir/TestLeNet.cpp.o: CMakeFiles/testlenet.dir/flags.make
CMakeFiles/testlenet.dir/TestLeNet.cpp.o: /home/duludulu/Documents/hls/source/TestLeNet.cpp
CMakeFiles/testlenet.dir/TestLeNet.cpp.o: CMakeFiles/testlenet.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/duludulu/Documents/hls/build/linux/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/testlenet.dir/TestLeNet.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/testlenet.dir/TestLeNet.cpp.o -MF CMakeFiles/testlenet.dir/TestLeNet.cpp.o.d -o CMakeFiles/testlenet.dir/TestLeNet.cpp.o -c /home/duludulu/Documents/hls/source/TestLeNet.cpp

CMakeFiles/testlenet.dir/TestLeNet.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/testlenet.dir/TestLeNet.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/duludulu/Documents/hls/source/TestLeNet.cpp > CMakeFiles/testlenet.dir/TestLeNet.cpp.i

CMakeFiles/testlenet.dir/TestLeNet.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/testlenet.dir/TestLeNet.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/duludulu/Documents/hls/source/TestLeNet.cpp -o CMakeFiles/testlenet.dir/TestLeNet.cpp.s

CMakeFiles/testlenet.dir/hls.cpp.o: CMakeFiles/testlenet.dir/flags.make
CMakeFiles/testlenet.dir/hls.cpp.o: /home/duludulu/Documents/hls/source/hls.cpp
CMakeFiles/testlenet.dir/hls.cpp.o: CMakeFiles/testlenet.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/duludulu/Documents/hls/build/linux/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/testlenet.dir/hls.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/testlenet.dir/hls.cpp.o -MF CMakeFiles/testlenet.dir/hls.cpp.o.d -o CMakeFiles/testlenet.dir/hls.cpp.o -c /home/duludulu/Documents/hls/source/hls.cpp

CMakeFiles/testlenet.dir/hls.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/testlenet.dir/hls.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/duludulu/Documents/hls/source/hls.cpp > CMakeFiles/testlenet.dir/hls.cpp.i

CMakeFiles/testlenet.dir/hls.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/testlenet.dir/hls.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/duludulu/Documents/hls/source/hls.cpp -o CMakeFiles/testlenet.dir/hls.cpp.s

CMakeFiles/testlenet.dir/tgcapture.cpp.o: CMakeFiles/testlenet.dir/flags.make
CMakeFiles/testlenet.dir/tgcapture.cpp.o: /home/duludulu/Documents/hls/source/tgcapture.cpp
CMakeFiles/testlenet.dir/tgcapture.cpp.o: CMakeFiles/testlenet.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/duludulu/Documents/hls/build/linux/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/testlenet.dir/tgcapture.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/testlenet.dir/tgcapture.cpp.o -MF CMakeFiles/testlenet.dir/tgcapture.cpp.o.d -o CMakeFiles/testlenet.dir/tgcapture.cpp.o -c /home/duludulu/Documents/hls/source/tgcapture.cpp

CMakeFiles/testlenet.dir/tgcapture.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/testlenet.dir/tgcapture.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/duludulu/Documents/hls/source/tgcapture.cpp > CMakeFiles/testlenet.dir/tgcapture.cpp.i

CMakeFiles/testlenet.dir/tgcapture.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/testlenet.dir/tgcapture.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/duludulu/Documents/hls/source/tgcapture.cpp -o CMakeFiles/testlenet.dir/tgcapture.cpp.s

# Object files for target testlenet
testlenet_OBJECTS = \
"CMakeFiles/testlenet.dir/mt19937ar.cpp.o" \
"CMakeFiles/testlenet.dir/stb_image_impl.cpp.o" \
"CMakeFiles/testlenet.dir/stdafx.cpp.o" \
"CMakeFiles/testlenet.dir/TestLeNet.cpp.o" \
"CMakeFiles/testlenet.dir/hls.cpp.o" \
"CMakeFiles/testlenet.dir/tgcapture.cpp.o"

# External object files for target testlenet
testlenet_EXTERNAL_OBJECTS =

testlenet: CMakeFiles/testlenet.dir/mt19937ar.cpp.o
testlenet: CMakeFiles/testlenet.dir/stb_image_impl.cpp.o
testlenet: CMakeFiles/testlenet.dir/stdafx.cpp.o
testlenet: CMakeFiles/testlenet.dir/TestLeNet.cpp.o
testlenet: CMakeFiles/testlenet.dir/hls.cpp.o
testlenet: CMakeFiles/testlenet.dir/tgcapture.cpp.o
testlenet: CMakeFiles/testlenet.dir/build.make
testlenet: CMakeFiles/testlenet.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/duludulu/Documents/hls/build/linux/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Linking CXX executable testlenet"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/testlenet.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/testlenet.dir/build: testlenet
.PHONY : CMakeFiles/testlenet.dir/build

CMakeFiles/testlenet.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/testlenet.dir/cmake_clean.cmake
.PHONY : CMakeFiles/testlenet.dir/clean

CMakeFiles/testlenet.dir/depend:
	cd /home/duludulu/Documents/hls/build/linux && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/duludulu/Documents/hls/source /home/duludulu/Documents/hls/source /home/duludulu/Documents/hls/build/linux /home/duludulu/Documents/hls/build/linux /home/duludulu/Documents/hls/build/linux/CMakeFiles/testlenet.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/testlenet.dir/depend

