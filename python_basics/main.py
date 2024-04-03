from file_handling import *

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