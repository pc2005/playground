import unittest

def square(n):
    return n*n


def cube(n):
    return n**3

class Test(unittest.TestCase):
    def test_square_positive(self):
        self.assertEqual(square(2), 4)

    def test_square_negative(self):
        self.assertEqual(square(-2.5), 6.25)

    def test_cube(self):
        self.assertEqual(cube('abc'), 8)