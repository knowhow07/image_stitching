# Makefile for C++ version of HW4 project
CXX = g++
CXXFLAGS = -std=c++17 `pkg-config --cflags --libs opencv4`

SRCS = main.cpp helpers.cpp hw4_challenge1.cpp runTests.cpp 
OBJS = $(SRCS:.cpp=.o)
TARGET = hw4_project

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) -o $@ $^ $(CXXFLAGS)

%.o: %.cpp
	$(CXX) -c $< $(CXXFLAGS)

clean:
	rm -f $(OBJS) $(TARGET)
