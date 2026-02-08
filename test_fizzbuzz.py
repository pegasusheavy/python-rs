x = 10
y = 3
print(x + y)
print(x > y)

for i in range(1, 21):
    if i % 15 == 0:
        print("FizzBuzz")
    elif i % 3 == 0:
        print("Fizz")
    elif i % 5 == 0:
        print("Buzz")
    else:
        print(i)

def fib(n):
    a = 0
    b = 1
    for i in range(n):
        temp = a
        a = b
        b = temp + b
    return a

print(fib(10))
