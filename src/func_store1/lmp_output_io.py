import numpy as np

def find_column_numbers_in_text(file, column_num, continuous_row=10):
    # Initialize a list to store the indices of lines that meet the criteria
    valid_indices = []

    # Read the file line by line
    with open(file, 'r') as f:
        lines = f.readlines()

    # Iterate through the lines and check for valid lines
    for index, line in enumerate(lines):
        # Split the line by space to get columns
        columns = line.strip().split()    
        # Check if there are enough columns
        if len(columns) == column_num:
            try:
                # Try to convert the specified column to a float
                float(columns[column_num - 1])
                # If successful, add the index to the valid_indices list
                valid_indices.append(index)
            except ValueError:
                # If conversion to float fails, continue to the next line
                continue

    # Filter the valid_indices list to keep only continuous rows
    final_indices = []
    current_continuous_count = 1

    for i in range(1, len(valid_indices)):
        if valid_indices[i] == valid_indices[i - 1] + 1:
            current_continuous_count += 1
        else:
            current_continuous_count = 1

        if current_continuous_count == continuous_row:
            final_indices.extend(range(valid_indices[i] - current_continuous_count + 1, valid_indices[i]+1))
        elif current_continuous_count > continuous_row:
            final_indices.extend(range(valid_indices[i],valid_indices[i]+1))

    # Initialize a list to store the selected lines
    selected_lines = []

    # Iterate through the lines and select lines based on the provided indices
    for index, line in enumerate(lines):
        if index in final_indices:
            selected_lines.append(line)

    # Join the selected lines into a single string (you can modify this as needed)
    selected_text = ''.join(selected_lines)

    return final_indices, selected_text

def deform_seg_indices(strain):
    diff = strain[1:] - strain[:-1]
    next_changed_id = np.where(diff!=0)[0]
    index0 = np.concatenate(([0],next_changed_id+1))
    index1 = np.concatenate((next_changed_id+1,[len(strain)]))
    return index0, index1
