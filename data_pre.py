import json
import re
from mapping import mapping

'''transfer ontology triples (热塑底油 has-attribute 粘度) to this format like: {'h': {'name': '热塑底油'}, 'relation': 'has-attribute', 't': {'name': '粘度'}}'''
triples = []

def ontology2text(ontology_path):
    with open(ontology_path, 'r', encoding='gbk') as f:
        lines = f.readlines()

    for line in lines:
        h_name, relation, t_name = line.strip().split('\t')
        triples.append({
            'h': {'name': h_name},
            'relation': relation,
            't': {'name': t_name}
        })

    return triples

'''为每一个三元组分配实体id，例如{'h': {'name': '热塑底油', 'id': '7689'}, 'relation': 'has-attribute', 't': {'name': '粘度', 'id': '4871'}}'''
def entity_add_id(word2id_path):
    # 加载词典
    with open(word2id_path, 'r', encoding='utf-8') as f:
        word2id = json.load(f)

    # word2id字典为实体分配ID
    for triple in triples:
        for entity in ['h', 't']:
            name = triple[entity]['name']
            triple[entity]['id'] = str(word2id[name])
        # print(triple)

    return triples

'''基于规则的entity2映射过程中的转换'''
def extract_entity2(text, special_chars_to_entity):
    for special_char, entity in special_chars_to_entity.items():
        # 将字符串转换为原始字符串
        pattern = r'(\d+(\.\d+)?\s*)?' + re.escape(special_char)
        match = re.search(pattern, text)
        if match:
            return special_char, entity

    return None, None

'''在corpus中查找每个实体的起始位置并添加到pos字段(远程监督标注关系)'''
def add_pos(corpus_path):
    # 读取语料集每一条instance
    with open(corpus_path, 'r', encoding='utf-8') as f:
        texts = f.read().split('\n')

    # 获取特殊字符到实体的映射
    special_chars_to_entity = mapping()

    # 将结果输出保存至dataset_result_path
    with open(dataset_result_path, 'w', encoding='utf-8') as f:
        first_entry = True
        for text in texts:
            found = False  # 添加一个标志来跟踪是否找到了匹配的三元组
            last_h_name = last_t_name = None      # 保存已经找到的头实体
            for triple in triples:
                if found and triple['h']['name'] != last_h_name and triple['t']['name'] != last_t_name:
                    found = False
                h_name = triple['h']['name']
                t_name = triple['t']['name']
                start_pos, end_pos = None, None  # Initialize the variables here

                # 第一种情况：entity1与entity2均在corpus中
                if h_name in text and t_name in text:
                    found = True        # 设置标志为True
                    last_h_name = h_name
                    last_t_name = t_name  # 更新头实体
                    new_triple = {
                        'text': text,
                        'relation': triple['relation'],
                        'h': {
                            'id': triple['h']['id'],
                            'name': h_name,
                            'pos': [text.find(h_name), text.find(h_name) + len(h_name)]
                        },
                        't': {
                            'id': triple['t']['id'],
                            'name': t_name,
                            'pos': [text.find(t_name), text.find(t_name) + len(t_name)]
                        }
                    }
                    if first_entry:
                        first_entry = False
                    else:
                        f.write("\n")
                    f.write(json.dumps(new_triple, ensure_ascii=False, indent=None))

                elif h_name not in text:
                    continue

                # 第二种情况需要进行规则转换：the entity1 is in the text while the entity2 is not.
                # 例如，text:双钢轮压路机需要以6km/h进行双向反复碾压，该text中有entity1双钢轮压路机，但没有直接给出entity2，需要进行规则转换，如text中有6km/h特殊单位符号km/h则可以根据此将entity2转换为速度，然后根据三元组可知relation为has-speed
                elif h_name in text and t_name not in text:
                    special_char, entity2 = extract_entity2(text, special_chars_to_entity)
                    if special_char == None and entity2 == None:
                        continue

                    elif special_char in text:
                        start_pos = text.find(special_char)
                        end_pos = start_pos + len(special_char)

                        if entity2 == t_name:
                            found = True
                            last_h_name = h_name
                            last_t_name = t_name  # 更新头实体
                            relation = triple['relation']
                            new_triple = {
                                'text': text,
                                'relation': relation,
                                'h': {
                                    'id': triple['h']['id'],
                                    'name': h_name,
                                    'pos': [text.find(h_name), text.find(h_name) + len(h_name)]
                                },
                                't': {
                                    'id': triple['t']['id'],
                                    'name': t_name,
                                    'pos': [start_pos, end_pos]
                                }
                            }
                            if first_entry:
                                first_entry = False
                            else:
                                f.write("\n")
                            f.write(json.dumps(new_triple, ensure_ascii=False, indent=None))

                        elif entity2 != t_name:
                            found = True
                            last_h_name = None
                            last_t_name = None  # 更新头实体
                            relation = 'NA'
                            new_triple = {
                                'text': text,
                                'relation': relation,
                                'h': {
                                    'id': triple['h']['id'],
                                    'name': h_name,
                                    'pos': [text.find(h_name), text.find(h_name) + len(h_name)]
                                },
                                't': {
                                    'id': triple['t']['id'],
                                    'name': t_name,
                                    'pos': [start_pos, end_pos]
                                }
                            }
                            if first_entry:
                                first_entry = False
                            else:
                                f.write("\n")
                            f.write(json.dumps(new_triple, ensure_ascii=False, indent=None))



'''将生成的instance进行去重操作'''
def remove_duplicates(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        instances = f.read().split('\n')

    unique_instances = set(instances)

    with open(output_file, 'w', encoding='utf-8') as f:
        for i, instance in enumerate(unique_instances):
            if i != 0:
                f.write("\n")
            f.write(instance)

if __name__ == '__main__':
    ontology_path = r'benchmark/CONSD/ontology triples_test.txt'
    word2id_path = r'benchmark/CONSD/corpus/glove_cn_50d_vocab.json'
    corpus_path = r'benchmark/CONSD/corpus/corpus.txt'
    dataset_result_path = r'triples_with_id_and_pos.txt'
    dataset_saved_path = r'final_dataset.txt'
    ontology2text(ontology_path)
    entity_add_id(word2id_path)
    add_pos(corpus_path)
    remove_duplicates(dataset_result_path, dataset_saved_path)