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
    
if __name__ == "__main__":
    list_manipulations()