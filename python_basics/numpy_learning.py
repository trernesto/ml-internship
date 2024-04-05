import numpy as np
from numpy import pi
from numpy import newaxis

if __name__ == "__main__":
    #np array from list
    some_list: list = [1, 2, 3]
    some_nparray: np.array = np.array(some_list)
    #print(some_list)
    #print(some_nparray, some_nparray.dtype)
    
    #2d array
    some_2dlist: list = [(1.5, 2, 5), (1.2, 3, 12)]
    some_2dnparray: np.array = np.array(some_2dlist)
    #print(some_2dlist)
    #print(some_2dnparray)
    
    #zeros, ones and empty arrays
    #!difference between zeroes and empty: empty just reserve memory for array; zeroes equate all memory to 0
    np.zeros((3, 4))
    np.ones((2, 3, 2))
    np.empty((3, 3))
    
    #arange like range, but arraylike
    #print(np.arange(10, 25, 5))
    #print(np.arange(10, 26, 5))
    #print(np.arange(0, 2, 0.1))
    
    #linspace from 0 to 10 in 10 pieces
    #print(np.linspace(0, 10, 10))
    #print(np.linspace(0, 10, 11))
    
    x = np.linspace(0, 2 * pi, 10)
    y = np.sin(x)
    #print(y)
    
    #reshape
    #print(x.reshape(10, 10))
    
    #basic operations
    A = np.array([2, 3, 4, 5])
    B = np.arange(4)
    #print(A - B)
    #print(A ** 2 <= 16)
    #print(10 * y)
    
    #matrix multiplication
    A = np.array([[1, 1],
                 [0, 1]])
    B = np.array([[2, 0],
                 [3, 4]])
    
    #print (A * B)   #Elementwise
    #print(A @ B)    #matrix
    #print(A.dot(B)) #same ^
    
    rg = np.random.default_rng(1)
    a = rg.random((2, 3))
    #print(a)
    #print("Max:", a.max(), "Min:", a.min(), "Sum:", a.sum())
    #print("Sum axis 0:", a.sum(axis=0), "Sum axis 1:", a.sum(axis=1))
    
    def f(x, y) -> int:
        return  10 * x + y
    
    b = np.fromfunction(f, (5, 4), dtype = int)
    #print(b)
    #print(b[0:5, 1])    #b[:, 1]
    #print(b[1, :])
    #print(b[-1])
    
    #a = np.floor(10 * rg.random((3, 4, 2)))
    #print(a)    
    #print(a.ravel())
    
    a = np.floor(10 * rg.random((2, 2)))
    b = np.floor(10 * rg.random((2, 2)))
    
    #print(a, "\n", b)
    
    #print(np.vstack((a, b)))
    #print(np.hstack((a, b)))
    #print(np.column_stack((a, b)))
    #a = np.floor(10 * rg.random((4, 1))) 
    #print(a[:, newaxis])
    