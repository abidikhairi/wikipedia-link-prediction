import argparse
import pandas as pd
import numpy as np

from tqdm import tqdm
from nltk.tokenize import RegexpTokenizer
from gensim.downloader import load as gensim_loader


def cli_main(args):
    google_news_vectors = gensim_loader('word2vec-google-news-300')
    tknzr = RegexpTokenizer(r'\w+')
    
    nodes_df = pd.read_csv(args.input_file, sep='\t')
    nodes_df.sort_values(by="id", inplace=True, ascending=True)

    print("num_nodes: ", len(nodes_df))
    node_features = []

    for row in tqdm(nodes_df.itertuples(), total=len(nodes_df)):
        text = str(row.text)
        tokens = tknzr.tokenize(text)
        
        text_embedding = np.zeros(300)
        
        for token in tokens:
            if token in google_news_vectors:
                text_embedding += google_news_vectors[token]
        
        node_features.append(text_embedding.tolist())
    
    node_features = np.array(node_features, dtype=np.float32)

    np.save(args.output_file, node_features)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input-file', required=True, help="Path to nodes.tsv file")
    parser.add_argument('--output-file', required=True, help="Path to store node features")

    args = parser.parse_args()
    
    cli_main(args)
