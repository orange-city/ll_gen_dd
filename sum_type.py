def count_lines_with_I(filename):
    count = 0
    with open(filename, 'r') as file:
        for line in file:
            if 'I' in line:
                count += 1
    return count

filename = 'traintag_fake7002.txt'
num_lines_with_I = count_lines_with_I(filename)
print("Number of lines containing 'I':", num_lines_with_I)
