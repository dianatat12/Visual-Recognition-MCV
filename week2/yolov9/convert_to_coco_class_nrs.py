import os 

directory = '/ghome/group04/MCV-C5-G4/week2/optional_task/dataset/test/labels'
output_directory = '/ghome/group04/MCV-C5-G4/week2/optional_task/test'

files = sorted(os.listdir(directory))

for file in files:
    read_path = os.path.join(directory, file)
    with open(read_path, 'r') as fp:
        lines = fp.readlines()
    lines = [line.split(' ') for line in lines]
    for line in lines: 
        if line[0] == '0': 
            line[0] = '2'
        elif line[0] == '1':
            line[0] = '0'
    
    lines = [' '.join(line) for line in lines]
    
    write_string = "".join(lines)

    output_file_path = os.path.join(output_directory, file)

    with open(output_file_path, 'w') as fp:
        fp.write(write_string)

    
