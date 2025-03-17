import os
from miditok import REMI, TokenizerConfig
import networkx as nx
from util.maestro_train_dev_test_splits import train_split

basic_remi_tokenizer = REMI()

start_token_type = "START"
end_token_type = "END"

start_node_index = -1
end_node_index = lambda remi_tokens: len(remi_tokens)

# Creates a score graph that is just a single path from START to END.
# A linked list is a monoidal structure.
def load_as_monoidal_remi_score_graph(midi_file_path):
    # Create the file's REMI token sequence. It already sorts tokens in a canonical order, monotonically by time and pitch.
    remi_tokens = basic_remi_tokenizer.encode(midi_file_path)[0].tokens

    # Create a directed graph containing each remi_token i as node i with type remi_tokens[i] and with edges (i, i+1) for each i.
    G = nx.DiGraph()

    # Add the start node.
    G.add_node(start_node_index, token_type=start_token_type)

    for i, token_type in enumerate(remi_tokens):
        G.add_node(i, token_type=token_type)
        G.add_edge(i-1, i)

    # Add the end node.
    G.add_node(len(remi_tokens), token_type=end_token_type)
    G.add_edge(end_node_index(remi_tokens) - 1, end_node_index(remi_tokens))

    return G