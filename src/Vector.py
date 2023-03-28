
import math

# Use the provided Vector class
class Vector:
    def __init__(self, coordinates):
        self.coordinates = coordinates
        self.length = len(coordinates)

    def __len__(self):
        return self.length

    def __repr__(self):
        return f"Vector {self.coordinates}"

    def __add__(self, other):
        return Vector([a + b for a, b in zip(self.coordinates, other.coordinates)])

    def __sub__(self, other):
        return Vector([a - b for a, b in zip(self.coordinates, other.coordinates)])

    def __mul__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return Vector([a * other for a in self.coordinates])
        else:
            return sum([a * b for a, b in zip(self.coordinates, other.coordinates)])

    def __rmul__(self, other):
        return self * other

    def dot(self, other):
        if self.length != other.length:
            raise ValueError("Vectors must be the same length.")
        return sum([self.coordinates[i] * other.coordinates[i] for i in range(self.length)])
    
    def magnitude(self):
        return math.sqrt(sum([self.coordinates[i] ** 2 for i in range(self.length)]))

    def distance(self, other):
        if self.length != other.length:
            raise ValueError("Vectors must be the same length.")
        return (self - other).magnitude()