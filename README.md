# CUDA Batch Image Processing Project

This project performs GPU-based image processing on a large dataset (200 grayscale images).  
It uses a custom CUDA kernel to run Sobel-like edge magnitude processing across the entire batch.

## Assignment Requirement Coverage

- **GPU computation included:** custom CUDA kernel (`processKernel`) in `src/batchImageProcessor.cu`
- **Large dataset included:** auto-generated synthetic dataset of **200 small images**
- **Proof of execution included:** output images in `data/output/` and a run log in `data/output/run_log.txt`


## Quick Start

```bash
chmod +x run.sh
./run.sh
```

## Example Output

After a successful run:

- `data/output/processed_0.pgm ... processed_199.pgm`
- `data/output/run_log.txt`

`run_log.txt` contains:

- timestamp
- input/output directories
- number of images processed
- image dimensions
- total pixels processed
- kernel+copy execution time in milliseconds
# image-edge-detection-cuda
