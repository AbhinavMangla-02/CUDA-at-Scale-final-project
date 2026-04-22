NVCC ?= nvcc

SRC_DIR := src
BIN_DIR := bin
TARGET := $(BIN_DIR)/batch_image_processor
SRC := $(SRC_DIR)/batchImageProcessor.cu

NVCCFLAGS := -O3 -std=c++14

.PHONY: all clean help run

all: $(TARGET)

$(TARGET): $(SRC)
	mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $< -o $@

run: $(TARGET)
	./run.sh

clean:
	rm -f $(TARGET)
	rm -rf data/output/*

help:
	@echo "make       Build GPU batch image processor"
	@echo "make run   Build then execute default workload"
	@echo "make clean Remove built executable and generated output"
