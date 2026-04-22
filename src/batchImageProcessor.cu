#include <cuda_runtime.h>

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <ctime>
#include <cerrno>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>

__global__ void processKernel(const unsigned char* input, unsigned char* output,
                              int width, int height, int batchSize) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int pixelsPerImage = width * height;
    const int totalPixels = pixelsPerImage * batchSize;
    if (idx >= totalPixels) {
        return;
    }

    const int img = idx / pixelsPerImage;
    const int p = idx % pixelsPerImage;
    const int x = p % width;
    const int y = p / width;

    const int left = (x > 0) ? (p - 1) : p;
    const int right = (x < width - 1) ? (p + 1) : p;
    const int up = (y > 0) ? (p - width) : p;
    const int down = (y < height - 1) ? (p + width) : p;

    const int base = img * pixelsPerImage;
    const int gx = static_cast<int>(input[base + right]) - static_cast<int>(input[base + left]);
    const int gy = static_cast<int>(input[base + down]) - static_cast<int>(input[base + up]);
    int mag = abs(gx) + abs(gy);
    if (mag > 255) {
        mag = 255;
    }
    output[base + p] = static_cast<unsigned char>(mag);
}

static bool readPGM(const std::string& path, std::vector<unsigned char>& data,
                    int& width, int& height) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return false;
    }

    std::string magic;
    in >> magic;
    if (magic != "P5") {
        return false;
    }

    auto skipComments = [&](std::istream& stream) {
        char c;
        stream >> std::ws;
        while (stream.peek() == '#') {
            std::string line;
            std::getline(stream, line);
            stream >> std::ws;
        }
        if (stream.peek() == '\n' || stream.peek() == '\r') {
            stream.get(c);
        }
    };

    skipComments(in);
    in >> width >> height;
    skipComments(in);
    int maxVal = 0;
    in >> maxVal;
    in.get();
    if (width <= 0 || height <= 0 || maxVal != 255) {
        return false;
    }

    data.resize(static_cast<size_t>(width) * static_cast<size_t>(height));
    in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size()));
    return in.good();
}

static bool writePGM(const std::string& path, const std::vector<unsigned char>& data,
                     int width, int height) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        return false;
    }
    out << "P5\n" << width << " " << height << "\n255\n";
    out.write(reinterpret_cast<const char*>(data.data()),
              static_cast<std::streamsize>(data.size()));
    return out.good();
}

static bool endsWith(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() &&
           s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

static bool ensureDir(const std::string& dir) {
    if (mkdir(dir.c_str(), 0755) == 0 || errno == EEXIST) {
        return true;
    }
    return false;
}

static std::vector<std::string> listPGMFiles(const std::string& dir) {
    std::vector<std::string> files;
    DIR* dp = opendir(dir.c_str());
    if (!dp) {
        return files;
    }
    struct dirent* ep;
    while ((ep = readdir(dp)) != nullptr) {
        std::string name(ep->d_name);
        if (name == "." || name == "..") {
            continue;
        }
        if (endsWith(name, ".pgm")) {
            files.push_back(dir + "/" + name);
        }
    }
    closedir(dp);
    std::sort(files.begin(), files.end());
    return files;
}

static std::string nowIsoLike() {
    const auto now = std::chrono::system_clock::now();
    const auto tt = std::chrono::system_clock::to_time_t(now);
    std::tm tm = *std::localtime(&tt);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &tm);
    return std::string(buf);
}

int main(int argc, char** argv) {
    std::string inputDir = "data/input";
    std::string outputDir = "data/output";

    if (argc > 1) {
        inputDir = argv[1];
    }
    if (argc > 2) {
        outputDir = argv[2];
    }
    const std::string logPath = outputDir + "/run_log.txt";

    if (!ensureDir(outputDir)) {
        std::cerr << "Failed creating output directory: " << outputDir << "\n";
        return 1;
    }

    const auto inputFiles = listPGMFiles(inputDir);
    if (inputFiles.empty()) {
        std::cerr << "No .pgm files found in " << inputDir << "\n";
        return 1;
    }

    std::vector<std::vector<unsigned char>> hostImages;
    hostImages.reserve(inputFiles.size());

    int width = 0;
    int height = 0;
    for (const auto& f : inputFiles) {
        std::vector<unsigned char> img;
        int w = 0;
        int h = 0;
        if (!readPGM(f, img, w, h)) {
            std::cerr << "Failed reading image: " << f << "\n";
            return 1;
        }
        if (width == 0 && height == 0) {
            width = w;
            height = h;
        }
        if (w != width || h != height) {
            std::cerr << "All images must be same size. Mismatch at: " << f << "\n";
            return 1;
        }
        hostImages.push_back(std::move(img));
    }

    const int batchSize = static_cast<int>(hostImages.size());
    const int pixelsPerImage = width * height;
    const size_t totalPixels = static_cast<size_t>(pixelsPerImage) * batchSize;
    const size_t totalBytes = totalPixels * sizeof(unsigned char);

    std::vector<unsigned char> hostInput(totalPixels);
    std::vector<unsigned char> hostOutput(totalPixels, 0);
    for (int i = 0; i < batchSize; ++i) {
        std::copy(hostImages[i].begin(), hostImages[i].end(),
                  hostInput.begin() + static_cast<size_t>(i) * pixelsPerImage);
    }

    unsigned char* dInput = nullptr;
    unsigned char* dOutput = nullptr;
    cudaError_t err;

    err = cudaMalloc(&dInput, totalBytes);
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc dInput failed: " << cudaGetErrorString(err) << "\n";
        return 1;
    }
    err = cudaMalloc(&dOutput, totalBytes);
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc dOutput failed: " << cudaGetErrorString(err) << "\n";
        cudaFree(dInput);
        return 1;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    err = cudaMemcpy(dInput, hostInput.data(), totalBytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy H2D failed: " << cudaGetErrorString(err) << "\n";
        cudaFree(dInput);
        cudaFree(dOutput);
        return 1;
    }

    const int threads = 256;
    const int blocks = static_cast<int>((totalPixels + threads - 1) / threads);
    processKernel<<<blocks, threads>>>(dInput, dOutput, width, height, batchSize);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << "\n";
        cudaFree(dInput);
        cudaFree(dOutput);
        return 1;
    }

    err = cudaMemcpy(hostOutput.data(), dOutput, totalBytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy D2H failed: " << cudaGetErrorString(err) << "\n";
        cudaFree(dInput);
        cudaFree(dOutput);
        return 1;
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsedMs = 0.0f;
    cudaEventElapsedTime(&elapsedMs, start, stop);

    for (int i = 0; i < batchSize; ++i) {
        std::vector<unsigned char> outImg(
            hostOutput.begin() + static_cast<size_t>(i) * pixelsPerImage,
            hostOutput.begin() + static_cast<size_t>(i + 1) * pixelsPerImage);

        std::ostringstream name;
        name << "processed_" << i << ".pgm";
        if (!writePGM(outputDir + "/" + name.str(), outImg, width, height)) {
            std::cerr << "Failed writing output image " << i << "\n";
            cudaFree(dInput);
            cudaFree(dOutput);
            return 1;
        }
    }

    std::ofstream log(logPath);
    if (log) {
        log << "timestamp: " << nowIsoLike() << "\n";
        log << "input_dir: " << inputDir << "\n";
        log << "output_dir: " << outputDir << "\n";
        log << "images_processed: " << batchSize << "\n";
        log << "image_size: " << width << "x" << height << "\n";
        log << "total_pixels: " << totalPixels << "\n";
        log << "kernel+copy_time_ms: " << elapsedMs << "\n";
    }

    std::cout << "Processed " << batchSize << " images on GPU.\n";
    std::cout << "Output written to: " << outputDir << "\n";
    std::cout << "Execution log: " << logPath << "\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(dInput);
    cudaFree(dOutput);

    return 0;
}
