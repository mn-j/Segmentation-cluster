
import csv 

# Load from CSV as list of tuples
def load_from_csv_as_list_of_tuples(file_path):
    data_list = []
    try:
        with open(file_path, mode='r', newline='') as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header
            for row in reader:
                data_list.append((row[0], row[1]))  # Each row is a tuple
    except FileNotFoundError:
        print(f"The file {file_path} does not exist.")
    return data_list





