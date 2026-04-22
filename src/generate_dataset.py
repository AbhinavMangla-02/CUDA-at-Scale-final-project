#!/usr/bin/env python3
import argparse
import os


def write_pgm(path: str, width: int, height: int, seed: int) -> None:
    header = f"P5\n{width} {height}\n255\n".encode("ascii")
    pixels = bytearray(width * height)
    for y in range(height):
        for x in range(width):
            v = (x * 3 + y * 5 + seed * 11 + (x ^ y)) % 256
            pixels[y * width + x] = v
    with open(path, "wb") as f:
        f.write(header)
        f.write(pixels)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic grayscale PGM dataset.")
    parser.add_argument("--output", default="data/input", help="Output directory")
    parser.add_argument("--count", type=int, default=200, help="Number of images")
    parser.add_argument("--width", type=int, default=256, help="Image width")
    parser.add_argument("--height", type=int, default=256, help="Image height")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    for i in range(args.count):
        path = os.path.join(args.output, f"img_{i:04d}.pgm")
        write_pgm(path, args.width, args.height, i)

    print(f"Generated {args.count} images in {args.output} ({args.width}x{args.height})")


if __name__ == "__main__":
    main()
