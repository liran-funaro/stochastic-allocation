OBJ_DIR=obj
BIN_DIR=bin
CC=g++-8

CPP_FILES=$(shell find . -type f -name '*.cpp')
HEADER_FILES=$(shell find . -type f \( -iname \*.hpp -o -iname \*.h \))

CPP_FLAGS=-std=c++11 -O3 -Wall -Wextra -Werror -pedantic-errors -funroll-loops -fdiagnostics-color=always
SHARED_LIB_FLAGS=-c -fPIC
BIN_FILE=bin/stochalloclib
LIB=$(BIN_FILE).so
LIB_DEBUG=$(BIN_FILE)_debug.so

COMPILE=$(CC) $(CPP_FLAGS) *.cpp
DEBUG_FLAGS=-g -D DEBUG

all: buildpath $(LIB)
	
debug: buildpath $(LIB_DEBUG)

$(LIB): $(CPP_FILES) $(HEADER_FILES)
	$(COMPILE) $(SHARED_LIB_FLAGS)
	@rm -f $(OBJ_DIR)/*.o
	@mv *.o $(OBJ_DIR)/
	g++ -shared -Wl,-soname,$(LIB) -o $(LIB)  $(OBJ_DIR)/*.o

$(LIB_DEBUG): $(CPP_FILES) $(HEADER_FILES)
	$(COMPILE) $(SHARED_LIB_FLAGS) $(DEBUG_FLAGS)
	@rm -f $(OBJ_DIR)/*.o
	@mv *.o $(OBJ_DIR)/
	g++ -shared -Wl,-soname,$(LIB_DEBUG) -o $(LIB_DEBUG)  $(OBJ_DIR)/*.o

buildpath:
	@if [ ! -d "$(OBJ_DIR)" ]; then mkdir $(OBJ_DIR); fi
	@if [ ! -d "$(BIN_DIR)" ]; then mkdir $(BIN_DIR); fi

clean:
	rm -f *.o
	if [ -d "obj" ]; then rm -R obj; fi

clean-all: clean
	if [ -d "bin" ]; then rm -R bin; fi
