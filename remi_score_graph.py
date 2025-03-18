import os
from miditok import REMI, TokenizerConfig
import networkx as nx
from util.baroque_midi_train_dev_test_splits import train_split
import sys
sys.setrecursionlimit(30000)  # for large graphs

basic_remi_tokenizer = REMI()

start_token_type = "START"
end_token_type = "END"

start_node_index = -1
end_node_index = lambda remi_tokens: len(remi_tokens)


# Given a list of REMI tokens, merges them into a single token type with parentheses.
def merge_remi_tokens_to_single_token_type(tokens):
    return f"({' '.join(tokens)})"


# Merges two token types into a single token type.
def merge_token_types(type_a, type_b):
    a_subtypes = parse_remi_tokens_from_token_type(type_a)
    b_subtypes = parse_remi_tokens_from_token_type(type_b)
    return merge_remi_tokens_to_single_token_type(a_subtypes + b_subtypes)


# Returns a list of the REMI tokens that are contained within the token type.
# Merged tokens are sequences of REMI tokens like (token1 token2 token3 ...).
# The token type can also be a single REMI token without parentheses.
def parse_remi_tokens_from_token_type(token_type):
    is_merged_token = token_type[0] == "(" and token_type[-1] == ")"
    if is_merged_token:
        parsed_remi_token_types = token_type[1:-1].split(" ")
        return parsed_remi_token_types
    return [token_type]  # The token type was just a single REMI token without parentheses.


# Helper methods to interpret REMI tokens.
is_bar = lambda remi_token: "Bar_" in remi_token
is_position = lambda remi_token: "Position_" in remi_token
parse_position = lambda remi_token: int(remi_token.split("_")[1])
is_pitch = lambda remi_token: "Pitch_" in remi_token
parse_pitch = lambda remi_token: int(remi_token.split("_")[1])
is_velocity = lambda remi_token: "Velocity_" in remi_token
is_duration = lambda remi_token: "Duration_" in remi_token
is_start = lambda remi_token: start_token_type == remi_token
is_end = lambda remi_token: end_token_type == remi_token

# Helper method for applying a sort order to graph nodes.
# If the tokenization has split after a position but before its pitch, we will want to scan ahead
# to find its pitch to ensure canonical sort order in the 1D serialization.
def lookahead_for_pitch(G, same_node_subsequent_remi_tokens, next_node_index=None):
    for token in same_node_subsequent_remi_tokens:
        if is_pitch(token):
            return parse_pitch(token)
    if next_node_index is None:
        return 0
    next_node_remi_tokens = parse_remi_tokens_from_token_type(G.nodes[next_node_index]["token_type"])
    successors = [succ for succ in G.successors(next_node_index)]
    return lookahead_for_pitch(G, next_node_remi_tokens, successors[0] if len(successors) == 1 else None)

# Adds a sort order to graph nodes.
# Recursively annotate each node with its depth, to provide a sort order for canonical 1D serialization.
# The tokens are read akin to a sort of modified uniform cost search based on this sort order.
def annotate_nodes_with_depths(G, start_node_index=-1, bar=0, position=0, pitch=0, seen=set()):
    # print("0")
    if start_node_index in seen:
        # print("1")
        return
    # print("2")
    seen.add(start_node_index)

    # print("A")
    remi_tokens = parse_remi_tokens_from_token_type(G.nodes[start_node_index]["token_type"])
    current_token_depth = None
    successors = [succ for succ in G.successors(start_node_index)]

    # print("B")
    for i, token in enumerate(remi_tokens):
        if is_bar(token):
            bar += 1
            position = 0
            pitch = 0
        elif is_position(token):
            position = parse_position(token)
            pitch = lookahead_for_pitch(G, remi_tokens[i + 1 :], successors[0] if len(successors) == 1 else None)
        elif is_pitch(token):
            pitch = parse_pitch(token)

        # The first iteration is based on the first REMI token. A merged token should be
        # sorted based on the depth of its first constituent REMI token.
        if current_token_depth is None:
            current_token_depth = (bar, position, pitch)

    # print("C")
    G.nodes[start_node_index]["depth"] = (
        current_token_depth if current_token_depth is not None else (bar, position, pitch)
    )

    # print("D")
    for succ in successors:
        # print(succ)
        if succ not in seen:
            annotate_nodes_with_depths(G, succ, bar, position, pitch, seen)

# Creates a score graph that is just a linked list from START to END.
def load_as_monoidal_remi_score_graph(midi_file_path):
    # Create the file's REMI token sequence. It already sorts tokens in a canonical order, monotonically by time and pitch.
    remi_tokens = basic_remi_tokenizer.encode(midi_file_path)[0].tokens

    # print("loaded remi tokens:", remi_tokens)

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

def load_with_vertical_and_horizontal_topology(midi_file_path):
    print(midi_file_path)
    G = load_as_monoidal_remi_score_graph(midi_file_path)

    # print('a')
    # Annotate each node with its depth, to provide a sort order for canonical 1D serialization.
    annotate_nodes_with_depths(G, start_node_index=-1, bar=0, position=0, pitch=0, seen=set())

    # print('b')
    # Link barlines to the every position token (every note event begins with one) that is simultanesous with the barline.
    bar_nodes = [node for node in G.nodes if is_bar(G.nodes[node]["token_type"])]
    for bar in bar_nodes:
        (bar_depth, position_depth, pitch_depth) = G.nodes[bar]["depth"]
        successor_positions = [
            succ for succ in G.successors(bar)
            if is_position(G.nodes[succ]["token_type"])
            and G.nodes[succ]["depth"][0] == bar_depth
            and G.nodes[succ]["depth"][1] == position_depth
        ]
        for successor_position in successor_positions:
            if not G.has_edge(bar, successor_position):
                G.add_edge(bar, successor_position)

    # print('c')
    # Link every duration (the last token of a note event) to the positions of notes that are simultaneous with it but sort after it.
    # These conditions prevent cycles, and the sort order is designed to place bass notes first.
    duration_nodes = [node for node in G.nodes if is_duration(G.nodes[node]["token_type"])]
    for duration in duration_nodes:
        (duration_bar_depth, duration_position_depth, duration_pitch_depth) = G.nodes[duration]["depth"]
        simultaneous_successor_position_tokens = [
            succ for succ in G.successors(duration)
            if is_position(G.nodes[succ]["token_type"])
            and duration_bar_depth == G.nodes[succ]["depth"][0]
            and duration_position_depth == G.nodes[succ]["depth"][1]
            and duration_pitch_depth < G.nodes[succ]["depth"][2]
        ]
        for successor_position in simultaneous_successor_position_tokens:
            if not G.has_edge(duration, successor_position):
                G.add_edge(duration, successor_position)

    # print('d')
    # Link each duration node to each pitch that occurs after it without first passing a position or bar.
    # Similar to the case above. Applies when the graph does not repeat positions for each simultaneous note event.
    for duration in duration_nodes:
        (duration_bar_depth, duration_position_depth, duration_pitch_depth) = G.nodes[duration]["depth"]
        successors = [succ for succ in G.successors(duration)]
        if len(successors) != 1:
            break
        succ = successors[0]
        while is_pitch(G.nodes[succ]["token_type"]) or is_velocity(G.nodes[succ]["token_type"]) or is_duration(G.nodes[succ]["token_type"]):
            if is_pitch(G.nodes[succ]["token_type"]) and not G.has_edge(duration, succ):
                G.add_edge(duration, succ)
            successors = [succ for succ in G.successors(succ)]
            if len(successors) != 1:
                break
            succ = successors[0]
    # print('e')
    return G