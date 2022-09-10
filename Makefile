#!/bin/bash

CC=nvcc
LDFLAGS=-lGLEW -lGLU -lGL -lglfw
DEBUG=-lineinfo

INC_DIR=includes
OBJ=src/main.o src/Utilities.o src/Shader.o src/Window.o 
all: $(OBJ)
	$(CC) $(OBJ) $(LDFLAGS) $(DEBUG) -o main

%.o: %.cpp
	$(CC) $(LDFLAGS) $(DEBUG) -c $< -o $@

%.o: %.cu
	$(CC) $(LDFLAGS) $(DEBUG) -c $< -o $@

clean:
	rm -v src/*.o