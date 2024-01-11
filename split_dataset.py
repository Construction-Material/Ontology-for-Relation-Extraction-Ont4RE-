import random
import json
from collections import defaultdict

def split_dataset(input_file, train_file, valid_file, test_file, ratios=(7, 2, 1)):
    with open(input_file, 'r', encoding='utf-8') as f:
        instances = [json.loads(line.strip()) for line in f if line.strip()]

    # 打乱句子顺序
    random.shuffle(instances)

    # 将句子按关系类型分组
    grouped_instances = defaultdict(list)
    for instance in instances:
        relation_type = instance['relation']  # 假设关系类型是每行的第一个元素
        grouped_instances[relation_type].append(instance)

    # 对每个关系类型进行分层抽样
    train_instances, valid_instances, test_instances = [], [], []
    for _, group in grouped_instances.items():
        random.shuffle(group)
        total = len(group)
        train_size = int(total * ratios[0] / sum(ratios))
        valid_size = int(total * ratios[1] / sum(ratios))
        train_instances.extend(group[:train_size])
        valid_instances.extend(group[train_size:train_size + valid_size])
        test_instances.extend(group[train_size + valid_size:])

    random.shuffle(train_instances)
    random.shuffle(valid_instances)
    random.shuffle(test_instances)

    # 将数据集写入文件
    with open(train_file, 'w', encoding='utf-8') as f:
        for instance in train_instances:
            f.write(json.dumps(instance, ensure_ascii=False) + '\n')

    with open(valid_file, 'w', encoding='utf-8') as f:
        for instance in valid_instances:
            f.write(json.dumps(instance, ensure_ascii=False) + '\n')

    with open(test_file, 'w', encoding='utf-8') as f:
        for instance in test_instances:
            f.write(json.dumps(instance, ensure_ascii=False) + '\n')

if __name__ == '__main__':

    # 拆分数据集
    input_file = r'dataset.txt'
    train_file = r'benchmark/CONSD/train/train.txt'
    valid_file = r'benchmark/CONSD/val/val.txt'
    test_file = r'benchmark/CONSD/test/test.txt'
    split_dataset(input_file, train_file, valid_file, test_file)
