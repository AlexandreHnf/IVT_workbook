FLAGS= -Wall -Wextra -pedantic
all: Main

Main: main.cpp
	g++ $(FLAGS) -c main.cpp 
	g++ $(FLAGS) main.o -o main
