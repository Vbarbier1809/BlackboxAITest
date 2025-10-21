def fn(x,y,z):
    a = x + y
    b = a * z
    c = b / 2
    return c

class Cls:
    def __init__(self, v):
        self.val = v

    def mth(self, n):
        res = self.val * n
        return res

# Test the functions
obj = Cls(5)
result = obj.mth(3)
print(f"Result: {result}")

# Some variables with poor naming
d = [1,2,3,4,5]
e = sum(d)
f = len(d)
g = e / f
print(f"Average: {g}")
