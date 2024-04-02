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

if __name__ == "__main__":
    print("Exevutable code")