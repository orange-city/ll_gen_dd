
import random
import re


def split_data(input_file, train_file, eval_file, train_lines):
    with open(input_file, 'r') as file:
        lines = file.readlines()
    lines = [line for line in lines if not re.match(r'^Speaker\w*', line)]
    lines = [line for line in lines if not re.match(r'^Speaker\w*', line)]

    random.shuffle(lines)

    train_lines = min(train_lines, len(lines))
    train_data = lines[:train_lines]
    eval_data = lines[train_lines:1100]

    with open(train_file, 'w') as file:
        file.writelines(train_data)

    with open(eval_file, 'w') as file:
        file.writelines(eval_data)


# 示例用法

# input_file = 'disfluency-detection/dps/swbd/pretrain.txt'
# output_file = 'disfluency-detection/dps/swbd/train_5000.dps'
# line_limit = 5000
#
# with open(input_file, 'r') as file:
#     lines = file.readlines()
#     random.shuffle(lines)
#
# filtered_lines = [line for line in lines if not line.startswith('Speaker')]
# selected_lines = filtered_lines[:line_limit]
#
# with open(output_file, 'w') as file:
#     file.writelines(selected_lines)
# exit(0)

input_file = "pretrain.txt"
train_file = "transformers/examples/fake_train_1000.txt"
eval_file = "transformers/examples/eval_1000.txt"
train_lines = 850#4257#4000#4293#4138#3620#4697#4137

split_data(input_file, train_file, eval_file, train_lines)

