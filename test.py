def calc(x,y,z):
    res1 = x + y
    res2 = res1 * z
    res3 = res2 / 2
    return res3

class Obj:
    def __init__(self, v):
        self.v = v

    def proc(self, n):
        res = self.v * n
        return res

# Test the code
o = Obj(10)
result = o.proc(5)
print(f"Result: {result}")

# More bad naming
lst = [1,2,3,4,5,6,7,8,9,10]
s = sum(lst)
l = len(lst)
avg = s / l
print(f"Average: {avg}")

def do_stuff(a,b,c,d,e,f,g):
    temp1 = a + b
    temp2 = temp1 * c
    temp3 = temp2 - d
    temp4 = temp3 + e
    temp5 = temp4 * f
    final = temp5 / g
    return final

# Call the function
val = do_stuff(1,2,3,4,5,6,7)
print(f"Final value: {val}")
