def extract_terms_from_triples(ontology_path: str, terms_output_path: str):
    # Read ontology triples
    with open(ontology_path, 'r', encoding='gbk') as f:
        lines = f.readlines()

    # Extract unique classes
    terms = set()
    for line in lines:
        h_name, _, t_name = line.strip().split('\t')
        terms.add(h_name)
        terms.add(t_name)

    # Write the terms to the output file
    with open(terms_output_path, 'w', encoding='utf-8') as f:
        for term in terms:
            f.write(term + '\n')

    return terms

ontology_path = 'benchmark/CONSD/ontology triples_test.txt'
terms_output_path = 'benchmark/CONSD/corpus/terms.txt'
extracted_terms = extract_terms_from_triples(ontology_path, terms_output_path)