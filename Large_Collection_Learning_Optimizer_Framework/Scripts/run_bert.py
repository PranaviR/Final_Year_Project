from flair.data import Sentence

def bert(input_file, document_embeddings):
    embeddings = []
    with open(input_file) as f:
        trainList = f.read().splitlines()
    sentences = [Sentence(tweet) for tweet in trainList]
    for sentence in sentences:
        document_embeddings.embed(sentence)
        sentence_embedding = sentence.get_embedding().detach().numpy().reshape(-1)
        embeddings.append(sentence_embedding)
    return embeddings