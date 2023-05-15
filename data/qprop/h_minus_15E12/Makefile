include ../common_variables_for_make.mk

all: imag_prop real_prop tsurff tsurff_mpi isurfv

imag_prop : libqprop.a imag_prop.cc
	$(CXX) $(optimargs) -std=c++0x imag_prop.cc -o imag_prop -I. -I../base -lqprop -lm -L../../lib/x86_64

real_prop : libqprop.a real_prop.cc
	$(CXX) $(optimargs) -std=c++0x real_prop.cc -o real_prop -I. -I../base -lqprop -lm -L../../lib/x86_64  

tsurff : libqprop.a tsurff.cc
	$(CXX) $(optimargs) -std=c++0x tsurff.cc -o tsurff -I. -I../base -lqprop -lm  $(gslargs) -L../../lib/x86_64

tsurff_mpi : libqprop.a tsurff_mpi.cc
	$(CXX) $(optimargs) -std=c++0x tsurff_mpi.cc -o tsurff_mpi -I. -I../base -I/usr/include/openmpi -lqprop -lm  $(gslargs) $(mpiargs) -L../../lib/x86_64

isurfv : libqprop.a isurfv.cc
	$(CXX) $(optimargs) -std=c++0x isurfv.cc -o isurfv -I. -I../base -lqprop -lm  $(gslargs) -L../../lib/x86_64
	
libqprop.a:
	cd ../base/ && make 

clean:
	cd ../base/ && make clean
	rm -f ../../lib/x86_64/libqprop.a
