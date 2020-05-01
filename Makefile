
all: test.x twed.x

test.x: test.cu libcuTWED.so reference_implementation/reference_arrays.h
	nvcc -o $@ $< libcuTWED.so

libcuTWED.so: cuTWED.cu cuTWED.h
	nvcc -g -O3 --shared --compiler-options "-fPIC -Wall -Wextra" -o $@ $<

twed.x: reference_implementation/twed.c reference_implementation/reference_arrays.h
	gcc -g -O3 -Wall -o $@ $< -lm

clean:
	rm -f *.so
	rm -f *.o
	rm -f *.x
	rm -rf *.dSYM
.PHONY: clean

