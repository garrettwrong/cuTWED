
all: libs test.x test_32.x twed.x

libs: libcuTWED.so libcuTWED_32.so 

test.x: test.cu libcuTWED.so reference_implementation/reference_arrays.h
	nvcc -o $@ $< libcuTWED.so

test_32.x: test.cu libcuTWED_32.so reference_implementation/reference_arrays.h
	nvcc -o $@ -DREAL_t=float $< libcuTWED_32.so

libcuTWED.so: cuTWED.cu cuTWED.h
	nvcc -g -O3 --shared --compiler-options "-fPIC -Wall -Wextra" -o $@ $<

libcuTWED_32.so: cuTWED.cu cuTWED.h
	nvcc -g -O3 --shared --compiler-options "-fPIC -Wall -Wextra" -o $@ -DREAL_t=float $<

twed.x: reference_implementation/twed.c reference_implementation/reference_arrays.h
	gcc -g -O3 -Wall -o $@ $< -lm

clean:
	rm -f *.so
	rm -f *.o
	rm -f *.x
	rm -rf *.dSYM
.PHONY: clean
.PHONY: libs
