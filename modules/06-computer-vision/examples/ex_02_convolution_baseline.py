"""2D convolution baseline on small matrices.

Run:
    python modules/06-computer-vision/examples/ex_02_convolution_baseline.py
"""

from __future__ import annotations


def convolve2d(image: list[list[float]], kernel: list[list[float]]) -> list[list[float]]:
    """Apply valid 2D convolution."""
    out_rows = len(image) - len(kernel) + 1
    out_cols = len(image[0]) - len(kernel[0]) + 1
    output: list[list[float]] = []

    for i in range(out_rows):
        row: list[float] = []
        for j in range(out_cols):
            value = 0.0
            for ki in range(len(kernel)):
                for kj in range(len(kernel[0])):
                    value += image[i + ki][j + kj] * kernel[ki][kj]
            row.append(value)
        output.append(row)

    return output


def main() -> None:
    """Run convolution with an edge-like kernel."""
    image = [
        [1, 2, 3, 2],
        [2, 4, 5, 1],
        [0, 1, 3, 2],
        [1, 2, 2, 0],
    ]
    kernel = [
        [1, 0, -1],
        [1, 0, -1],
        [1, 0, -1],
    ]

    output = convolve2d(image, kernel)
    print("conv_output=")
    for row in output:
        print([round(value, 2) for value in row])


if __name__ == "__main__":
    main()
