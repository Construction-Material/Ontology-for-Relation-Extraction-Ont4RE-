from sentence_transformers import SentenceTransformer, models
from transformers import BertModel, BertTokenizer
import numpy as np
import jieba
import torch
import json
from typing import List, Dict, Optional, Tuple

# Function to compute cosine similarity between two embeddings
def cosine_similarity(embedding1, embedding2):
    return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

def ontology2text(ontology_path: str) -> List[Dict]:
    with open(ontology_path, 'r', encoding='gbk') as f:
        lines = f.readlines()

    triples = []
    for line in lines:
        h_name, relation, t_name = line.strip().split('\t')
        triples.append({
            'h': {'name': h_name},
            'relation': relation,
            't': {'name': t_name}
        })

    return triples

def entity_add_id(triples: List[Dict],
                  word2id_path: str) -> List[Dict]:
    # 加载词典
    with open(word2id_path, 'r', encoding='utf-8') as f:
        word2id = json.load(f)

    # word2id字典为实体分配ID
    for triple in triples:
        for entity in ['h', 't']:
            name = triple[entity]['name']
            triple[entity]['id'] = str(word2id[name])

    return triples

def pre_triple_embeddings(triples: List[Dict],
                          model: SentenceTransformer,
                          device: str) -> Dict:

    embeddings = {}

    for triple in triples:
        h_name = triple['h']['name']
        t_name = triple['t']['name']

        h_embedding = model.encode(h_name, convert_to_tensor=True, device=device)
        t_embedding = model.encode(t_name, convert_to_tensor=True, device=device)

        embeddings[h_name] = h_embedding
        embeddings[t_name] = t_embedding

    return embeddings

def read_corpus(corpus_path):
    with open(corpus_path, 'r', encoding='utf-8') as f:
        sentences = f.read().split('\n')

    return sentences

def add_terms_to_jieba(terms_path):
    with open(terms_path, 'r', encoding='utf-8') as f:
        for line in f:
            word = line.strip()  # Remove whitespace or newline characters from each line
            jieba.add_word(word)

def semantic_similarity(triples: List[Dict],
                        sentences_batch: List[str],
                        model: SentenceTransformer,
                        pre_triple_embeddings: Dict,
                        threshold: float,
                        device: str
) -> List[Optional[Dict]]:

    '''
    # Get embeddings for the head and tail classes of the triple
    h_embedding = model.encode(triple['h']['name'], convert_to_tensor=True, device=device)
    t_embedding = model.encode(triple['t']['name'], convert_to_tensor=True, device=device)
    '''

    batch_results = []

    sentences_embeddings = model.encode(sentences_batch, convert_to_tensor=True, device=device)

    for idx, sentence_embedding in enumerate(sentences_embeddings):
        sentence = sentences_batch[idx]
        # Getting embeddings for each word in the sentence
        # words = word_embedding_model.tokenizer.tokenize(sentence) #原word_embedding是以wordpiece为单位进行分词，我们需要以词为单位进行分词
        words = list(jieba.cut(sentence))
        if len(words) == 0:
            continue
        word_embeddings = model.encode(words, convert_to_tensor=True, device=device)

        best_similarity = -float('inf')
        best_result = None

        for triple in triples:
            # Get precomputed embeddings for the head and tail classes of the triple
            h_embedding = pre_triple_embeddings[triple['h']['name']]
            t_embedding = pre_triple_embeddings[triple['t']['name']]

            # Calculating similarities
            # h_similarities = [torch.nn.functional.cosine_similarity(h_embedding, word_emb, dim=0).item() for word_emb in word_embeddings]
            # t_similarities = [torch.nn.functional.cosine_similarity(t_embedding, word_emb, dim=0).item() for word_emb in word_embeddings]

            # Computing cosine similarity using matrix multiplication
            # Instead of iterating over each word, we use batch matrix multiplication to compute the similarities
            h_similarities = torch.nn.functional.cosine_similarity(word_embeddings, h_embedding.unsqueeze(0))
            t_similarities = torch.nn.functional.cosine_similarity(word_embeddings, t_embedding.unsqueeze(0))

            '''
            # Getting the most similar words for head and tail in the sentence
            h_max_similarity, h_max_index = max((val, idx) for (idx, val) in enumerate(h_similarities))
            t_max_similarity, t_max_index = max((val, idx) for (idx, val) in enumerate(t_similarities))
            '''

            # Finding the most similar words to the head and tail entities
            h_max_index = h_similarities.argmax().item()
            t_max_index = t_similarities.argmax().item()

            h_pos_start = sentence.find(words[h_max_index])
            h_pos_end = h_pos_start + len(words[h_max_index])

            t_pos_start = sentence.find(words[t_max_index])
            t_pos_end = t_pos_start + len(words[t_max_index])

            h_max_similarity = h_similarities[h_max_index].item()
            t_max_similarity = t_similarities[t_max_index].item()
            avg_similarity = (h_max_similarity + t_max_similarity) / 2

            # If both similarities are above threshold, return the relation and positions, else None
            if h_max_similarity > threshold and t_max_similarity > threshold:

                if avg_similarity > best_similarity:
                    best_similarity = avg_similarity
                    best_result = {
                        'text': sentence,
                        'relation': triple['relation'],
                        'h': {
                            'id': triple['h']['id'],
                            'name': triple['h']['name'],
                            # 'most_similar_word': words[h_max_index],
                            'pos': [h_pos_start, h_pos_end]
                        },
                        't': {
                            'id': triple['t']['id'],
                            'name': triple['t']['name'],
                            # 'most_similar_word': words[t_max_index],
                            'pos': [t_pos_start, t_pos_end]
                        }
                        #'h_similarity': round(h_max_similarity, 2),
                        #'t_similarity': round(t_max_similarity, 2)
                    }

            elif h_max_similarity > threshold and t_max_similarity < threshold:

                if avg_similarity > best_similarity:
                    best_similarity = avg_similarity
                    best_result = {
                        'text': sentence,
                        'relation': 'NA',
                        'h': {'id': triple['h']['id'],
                              'name': triple['h']['name'],
                              # 'most_similar_word': words[h_max_index],
                              'pos': [h_pos_start, h_pos_end]
                              },
                        't': {
                            'id': triple['t']['id'],
                            'name': triple['t']['name'],
                            # 'most_similar_word': words[t_max_index],
                            'pos': [t_pos_start, t_pos_end]
                        }
                        # 'h_similarity': round(h_max_similarity, 2),
                        # 't_similarity': round(t_max_similarity, 2)
                    }

        batch_results.append(best_result)

    return batch_results

# 加载模型和tokenizer
pretrain_path = 'model/bert-base-chinese'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
word_embedding_model = models.Transformer(pretrain_path)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode_mean_tokens=True)
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
model.to(device=device)
BATCH_SIZE = 512

# 依赖文件
ontology_path = 'benchmark/CONSD/CEMO_triples.txt'
corpus_path = 'benchmark/CONSD/corpus/corpus.txt'
word2id_path = r'benchmark/CONSD/corpus/vocab.json'
terms_path = 'benchmark/CONSD/corpus/terms.txt'
dataset_result_path = r'dataset.txt'
triples = ontology2text(ontology_path)
triples = entity_add_id(triples, word2id_path)
sentences = read_corpus(corpus_path)
add_terms_to_jieba(terms_path)
pre_triple_embeddings = pre_triple_embeddings(triples, model, device)

with open(dataset_result_path, 'w', encoding='utf-8') as f:
    first_entry = True

    for i in range(0, len(sentences), BATCH_SIZE):
        sentences_batch = sentences[i:i + BATCH_SIZE]
        batch_results = semantic_similarity(triples, sentences_batch, model, pre_triple_embeddings,
                                                  threshold=0.85, device=device)
        for result in batch_results:
            if result:
                dataset = {
                    'text': result['text'],
                    'relation': result['relation'],
                    'h': {
                        'id': result['h']['id'],
                        'name': result['h']['name'],
                        # 'most_similar_word': result['h']['most_similar_word'],
                        'pos': result['h']['pos']
                    },
                    't': {
                        'id': result['t']['id'],
                        'name': result['t']['name'],
                        # 'most_similar_word': result['t']['most_similar_word'],
                        'pos': result['t']['pos']
                    },
                    # 'h_similarity': result['h_similarity'],
                    # 't_similarity': result['t_similarity']
                }

                if first_entry:
                    first_entry = False
                else:
                    f.write("\n")

                f.write(json.dumps(dataset, ensure_ascii=False, indent=None))