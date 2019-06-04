CFLAGS = -I .
LIBS = -lm
CC = cc
FLAGS = 

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS) $(FLAGS)

network: network_test.o random_ring.o collectives.o subcomms.o utils.o
	$(CC) -o network_test utils.o random_ring.o collectives.o subcomms.o network_test.o $(CFLAGS) $(FLAGS) $(LIBS)

load: network_load_test.o random_ring.o collectives.o subcomms.o utils.o congestors.o
	$(CC) -o network_load_test utils.o congestors.o random_ring.o collectives.o subcomms.o network_load_test.o $(CFLAGS) $(FLAGS) $(LIBS)

default: network

all: network load

clean:
	rm -f *.o
	rm -f network_test
	rm -f network_load_test
