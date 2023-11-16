FLAGS=-O3 -fopenmp -lm
CC=gcc
RM=rm -f

# Executáveis
EXEC1=tsp
EXEC2=mpi-tsp

all: $(EXEC1) $(EXEC2)

# Compilação e linkagem para tsp
$(EXEC1): 
	$(CC) -c $(FLAGS) $(EXEC1).c -o $(EXEC1).o
	$(CC) $(EXEC1).o -o $(EXEC1) $(FLAGS)

# Compilação e linkagem para tsp-parallel
$(EXEC2): 
	mpicc -c $(FLAGS) $(EXEC2).c -o $(EXEC2).o
	mpicc $(EXEC2).o -o $(EXEC2) $(FLAGS)

run:
	./$(EXEC1) < best/14.in

run2:
	mpirun -np 8 $(EXEC2) < best/14.in

clean:
	$(RM) $(EXEC1).o $(EXEC1) $(EXEC2).o $(EXEC2)