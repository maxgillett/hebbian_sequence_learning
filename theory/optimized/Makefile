UNAME := $(shell uname)
DISTRO := $(shell uname -a | sed -n 's/Linux\s*\([^0-9-]*\).*$$/\1/p')

EXEDIR=./targets/fedora

ifeq ($(UNAME), Darwin)
	EXEDIR=./targets/darwin
endif
ifeq ($(UNAME), Linux)
ifeq ($(DISTRO), ubuntu)
	EXEDIR=./targets/ubuntu
endif
endif

CC = g++

CFLAGS= -std=c++11 -Wall -pipe -O3 -ffast-math -funsafe-math-optimizations \
        -march=native -mtune=native -pedantic -Wno-unused-variable

#override CFLAGS += -O0 -DDEBUG -ggdb

LDFLAGS= -lgsl -lgslcblas -lm

SIMULATIONS=$(EXEDIR)/integration # $(EXEDIR)/test

.SECONDARY:

all: $(SIMULATIONS)

$(EXEDIR)/%: %.o $(LOCALOBJECTS)
	$(CC) $(CFLAGS) $< $(LDFLAGS) -o $@ 

%.o : %.cpp
	$(CC) $(CFLAGS) -c $<

clean: 
	rm -f *.o $(SIMULATIONS)
  
