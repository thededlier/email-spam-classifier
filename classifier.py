import os
import io
from pandas import DataFrame

def read_files(path):
    for dir_path, dir_names, file_names in os.walk(path):
        for file_name in file_names:
            path = os.path.join(dir_path, file_name)
            in_body = False
            lines = []
            f = io.open(path, 'r', encoding='latin1')
            for line in f:
                if in_body:
                    lines.append(line)
                elif line == '\n':
                    in_body = True
            f.close()
            message = '\n'.join(lines)
            yield path, message

def data_frame_from_directory(path, classification):
    rows = []
    index = []
    for file_name, message in read_files(path):
        rows.append({'message': message, 'class': classification})
        index.append(file_name)

    return DataFrame(rows, index=index)

data = DataFrame({'message': [], 'class': []})

data = data.append(data_frame_from_directory('./emails/spam', 'spam'))
data = data.append(data_frame_from_directory('e./emails/ham', 'ham'))

print("--- Data Summary ---")
print(data.head())