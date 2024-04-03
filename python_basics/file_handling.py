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
              + " successfully")
    except IOError:
        print("Error: could not rename file " + filename)
        
def delete_file(filename) -> None:
    try:
        os.remove(filename)
        print("File " + filename + " deleted successfully")
    except IOError:
        print("Error could not delete file " + filename)