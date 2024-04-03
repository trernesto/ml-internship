#text type
text_type: str = "hello world"

#numeric types
int_type: int = 5
float_type: float = 3.74
complex_type_1: complex = complex(1, 3)
complex_type_2: complex = complex("5-10j")

#sequence types
list_type: list = ["apple", "orange", "cherry"]
tuple_type: tuple = ("table", "chair", "bookshelf")
range_type: range = range(3, 6)

#mapping types
dict_type: dict = {
    "name": "Josh",
    "age": 36,
    "id": 1623412
}

#set types
set_type: set = {
    "apple",
    "cherry",
    "banana",
    "orange",
    "kimumori"
}

frozenset_type: frozenset = frozenset(set_type)

#boolean type
bool_type: bool = True #False

#binary types
bytes_type: bytes = b"hello"
byte_array: bytearray = bytearray(5)
memoryview_type: memoryview = memoryview(bytes(10))

#None type
none_type: None = None

def list_manipulations() -> None:
    some_list: list = [1, 2, 3]
    print("Some list:", some_list)
    
    some_list.append(4)
    print("\nAfter append:", some_list)
    
    some_range: range = range(2, 5)
    print("\nRange:", some_range)
    some_list.extend(some_range)
    print("After extending range:", some_list)
    
    some_list.insert(2, 10)
    print("\nAfter insert:", some_list)
    
    some_list.remove(3)
    print("\nAfter remove:", some_list)

    pop_list: list = some_list.pop(2)
    print("\nAfter pop:", some_list)
    print("Also poped item:", pop_list)
    
    print("\nNumber of times 4 appeared:", some_list.count(4))
    
    some_list.sort()
    print("\nSorted list:", some_list)
    some_list.reverse()
    print("Reversed list:", some_list)
    
    some_copy_list: list = some_list.copy()
    some_second_list = some_list
    print("\nCopy of original list", some_copy_list)
    print("= list:", some_second_list)
    
    some_list.clear()
    print("\nAfter clear")
    print("\nCopy of original list", some_copy_list)
    print("= list:", some_second_list)
   
def dict_and_set_manipulations() -> None:
    some_dict: dict = {
        "name": "John",
        "age": 23,
        "id": 100052
    }
    
    print("\nDict:", some_dict)
    print("Get name from dict:", some_dict["name"])
    
    some_set: set = {
        "apple",
        "banana",
        "cherry",
        0, False,   # 0===False
        1, True     # 1===True
    }
    print(some_set)
   
def file_manipulations(in_filename: str) -> None:
    '''
    f.open(filename, mode) <- base
    
    file = open(in_filename, "r")
    print(file.read())  #string that contains all characters in file
    
    for each in file:
        print(each, end="")
    ''' 
    with open(in_filename) as new_file:
        data = new_file.readlines()
        for line in data:
            word = line.split()
            print(word)
            
    with open(in_filename, "w") as write_file:
        write_file.write("some new data")
        write_file.write("My name is not Er")
        
    
    with open(in_filename, "a") as append_file:
        append_file.write("1234")
        append_file.write("Abricos")
        
import os

def create_file(filename: str) -> None:
    try:
        with open(filename, 'w') as f:
            f.write('Hello world!\n')
        print("File " + filename + " created successfully!")
    except IOError:
        print("Error: could not create file " + filename)

def read_file(filename) -> None:
    try:
        with open(filename, 'r') as f:
            content: str = f.read()
            print(content)
    except IOError:
        print("Error: could not read file " + filename)
        
def append_file(filename, text) -> None:
    try:
        with open(filename, 'a') as f:
            f.write(text)
        print("Text appended to file " + filename + " success")
    except IOError:
        print("Error: could not append to file " + filename)
        
def rename_file(filename, new_filename) -> None:
    try:
        os.rename(filename, new_filename)
        print("File " + filename + " renamed to " + new_filename
              + "successfully")
    except IOError:
        print("Error: could not rename file " + filename)
        
def delete_file(filename) -> None:
    try:
        os.remove(filename)
        print("File " + filename + " deleted successfully")
    except IOError:
        print("Error could not delete file " + filename)

if __name__ == "__main__":
    filename = "example.txt"
    new_filename = "new_example.txt"
 
    create_file(filename)
    read_file(filename)
    append_file(filename, "This is some additional text.\n")
    read_file(filename)
    rename_file(filename, new_filename)
    read_file(new_filename)
    delete_file(new_filename)