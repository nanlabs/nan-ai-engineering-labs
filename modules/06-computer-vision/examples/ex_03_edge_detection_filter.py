"""Edge detection using Sobel-like kernels.

Run:
    python modules/06-computer-vision/examples/ex_03_edge_detection_filter.py
"""

from __future__ import annotations


def convolve2d(image: list[list[float]], kernel: list[list[float]]) -> list[list[float]]:
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
    """Compute horizontal and vertical edge maps."""
    image = [
        [10, 10, 10, 200],
        [10, 10, 20, 220],
        [10, 20, 30, 230],
        [5, 10, 15, 240],
    ]

    sobel_x = [[1, 0, -1], [2, 0, -2], [1, 0, -1]]
    sobel_y = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]

    grad_x = convolve2d(image, sobel_x)
    grad_y = convolve2d(image, sobel_y)

    print("grad_x=")
    for row in grad_x:
        print([round(value, 2) for value in row])

    print("grad_y=")
    for row in grad_y:
        print([round(value, 2) for value in row])


if __name__ == "__main__":
    main()
