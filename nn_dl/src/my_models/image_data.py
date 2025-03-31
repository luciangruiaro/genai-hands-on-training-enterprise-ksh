from dataclasses import dataclass
from typing import List


@dataclass
class ImageData:
    label: int
    image: List[List[int]]

    def print_image(self):
        print("Image:")
        for row in self.image:
            print(" ".join(str(pixel) for pixel in row))
        print("Label:", self.label)

    def to_double_array(self):
        return [[float(pixel) for pixel in row] for row in self.image]
