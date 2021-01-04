FLAGS= -Wall -Wextra -pedantic
all: Main

Main: IVT2020_Heneffe_Alexandre_main.cpp
	g++ $(FLAGS) -c IVT2020_Heneffe_Alexandre_main.cpp 
	g++ $(FLAGS) IVT2020_Heneffe_Alexandre_main.o -o main
