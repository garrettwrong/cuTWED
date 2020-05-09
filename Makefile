CC=gcc
NVCC=nvcc

NV_GEN= -arch=sm_75 \
 -gencode=arch=compute_30,code=sm_30 \
 -gencode=arch=compute_35,code=sm_35 \
 -gencode=arch=compute_50,code=sm_50 \
 -gencode=arch=compute_52,code=sm_52 \
 -gencode=arch=compute_60,code=sm_60 \
 -gencode=arch=compute_61,code=sm_61 \
 -gencode=arch=compute_70,code=sm_70 \
 -gencode=arch=compute_75,code=sm_75 \
 -gencode=arch=compute_75,code=compute_75

all: lib test.x testf.x twed.x

lib: libcuTWED.so

test.x: test.cu libcuTWED.so reference_implementation/reference_arrays.h
	$(NVCC) $(NV_GEN) -g -O3 -o $@ $< libcuTWED.so

testf.x: testf.cu libcuTWED.so reference_implementation/reference_arrays.h
	$(NVCC) $(NV_GEN) -g -O3 -o $@ $< libcuTWED.so

libcuTWED.so: cuTWED.cu cuTWED.h cuTWED_core.h
	$(NVCC) $(NV_GEN) -g -O3 --shared --compiler-options "-fPIC -Wall -Wextra" -o $@ $<

twed.x: reference_implementation/twed.c reference_implementation/reference_arrays.h
	$(CC) -g -O3 -Wall -o $@ $< -lm

clean:
	rm -f *.so
	rm -f *.o
	rm -f *.x
	rm -rf *.dSYM
.PHONY: all
.PHONY: clean
.PHONY: lib
