CC      = gcc
CFLAGS  = -O3 -march=native -fopenmp -Wall -Wextra
LDFLAGS = -lm

matmul: src/matmul.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

.PHONY: check bench plots clean

check: matmul
	./matmul --check v1 1024 1
	./matmul --check v2 1024 8
	./matmul --check v3 1024 8
	./matmul --check v4 1024 8

bench: matmul
	bash bench/run.sh

plots:
	python3 analysis/plot.py

clean:
	rm -f matmul
