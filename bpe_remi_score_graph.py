import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
from remi_score_graph import (
    load_as_monoidal_remi_score_graph,
    load_with_vertical_and_horizontal_topology,
    start_token_type,
    end_token_type,
    annotate_nodes_with_depths,
    parse_remi_tokens_from_token_type,
    merge_token_types,
)
from util.baroque_midi_train_dev_test_splits import train_split
import MidiTok.src.miditok as miditok

import sys
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

sys.setrecursionlimit(30000)  # for large graphs

basic_remi_tokenizer = miditok.REMI()


def remi_token_type_sequence_to_vocab_ids(remi_token_type_sequence, tokenizer=basic_remi_tokenizer):
    token_id_sequence = []
    for token_type in remi_token_type_sequence:
        if token_type in tokenizer.vocab:
            token_id = tokenizer.vocab[token_type]
            token_id_sequence.append(token_id)
        else:
            print(f"Token '{token_type}' not in tokenizer vocabulary")
    return token_id_sequence


def remi_token_type_sequence_to_midi(remi_token_type_sequence, tokenizer=basic_remi_tokenizer):
    token_ids = remi_token_type_sequence_to_vocab_ids(remi_token_type_sequence, tokenizer=tokenizer)
    score = tokenizer.decode([token_ids])
    score.dump_midi("./output2.mid")


# Render the graph using matplotlib.
def draw_graph(G, title):
    plt.figure(figsize=(8, 5))
    pos = nx.spring_layout(G, seed=7)
    labels = {
        node: f"{node}\n{data['token_type']}\n{data['depth']}" if "depth" in data else f"{node}\n{data['token_type']}"
        for node, data in G.nodes(data=True)
    }
    nx.draw(
        G, pos, with_labels=True, labels=labels, node_color="lightblue", edge_color="gray", node_size=100, font_size=8
    )
    plt.title(title)
    plt.show()


class REMIGraphBPE:
    def __init__(self, target_learned_vocab_size, graph_corpus):
        self.target_learned_vocab_size = target_learned_vocab_size
        self.vocabulary = {}  # Stores the learned vocabulary.
        self.original_graphs = [G.copy() for G in graph_corpus]  # Graphs to learn from.
        self.graphs_with_merges = [
            G.copy() for G in graph_corpus
        ]  # Copies of the original_graphs that have been modified by REMI.
        # Index of pairs that cannot be merged due to connectivity issues in previous merge attempts.
        self.per_graph_blocked_pairs = [set() for _ in range(len(graph_corpus))]

        # Special tokens for 1D serialization.
        self.continue_sequence_token = 7777
        self.continue_sequence_token_type = "->"
        self.simultaneous_start_token = 8888

        # Say we have a path of a complete measure followed by another such path.
        # To collate them if they actually begin at the same time step, we wrap them
        # in a simultaneous section.
        self.simultaneous_start_token_type = "["
        self.simultaneous_end_token = 9999
        self.simultaneous_end_token_type = "]"

        # At the limit, the Janus token approaches halving the number of tokens for simultaneous sections.
        # We just use one Janus token where we would have both an end and a beginning token consecutively.
        self.simultaneous_end_begin_janus_token = 10000
        self.simultaneous_end_begin_janus_token_type = "]["

    def log_status_update(self, label, type_a, type_b):
        print(label)
        print(f"Learned vocab size is {len(self.vocabulary)} of target {self.target_learned_vocab_size}")
        pair_counts = self.count_type_pairs_of_potentially_mergable_edges(
            self.graphs_with_merges, self.per_graph_blocked_pairs
        )
        total_candidate_pairs = sum(pair_counts.values())
        blocked_pairs_count = sum([len(blocked_pairs) for blocked_pairs in self.per_graph_blocked_pairs])
        status_update = f"{pair_counts[(type_a, type_b)]} out of {total_candidate_pairs} are ({type_a}, {type_b}) and there are {blocked_pairs_count} blocked pairs."
        print(status_update)

    def count_type_pairs_of_potentially_mergable_edges(self, graphs, per_graph_blocked_pairs):
        # Count token type pairs of potentially mergable edges A -> B.
        return Counter(
            (G.nodes[a]["token_type"], G.nodes[b]["token_type"])
            for (G, blocked_pairs) in zip(graphs, per_graph_blocked_pairs)
            for (a, b) in G.edges()
            if G.nodes[a]["token_type"] != start_token_type  # Don't merge from the start token.
            and G.nodes[b]["token_type"] != end_token_type  # Don't merge into the end token.
            and (a, b) not in blocked_pairs  # Edges are blocked if earlier merge attempts broke connetivity.
        )

    # Heuristic to prevent obvious connectivity issues. Returns True if the merge passes, False otherwise.
    def _candidate_merge_passes_heuristic(self, G, a, b):
        # If we complete the merge, then will drop the edges from A to its successors. This might
        # disconnect the graph, so we make sure that all of A's other successors have an alternate
        # predecessor. This heuristic can fail when A is the only entrance to a cycle, so we still
        # need to check the graph's connectivity after the merge.
        other_successors_of_a = [succ for succ in G.successors(a) if succ != b]
        alternate_predecessors_of_other_successors_of_a = [
            pred for succ in other_successors_of_a for pred in G.predecessors(succ) if pred != a
        ]
        a_has_other_successors = len(other_successors_of_a) != 0
        other_successors_of_a_have_other_predecessors = len(alternate_predecessors_of_other_successors_of_a) != 0
        if a_has_other_successors and not other_successors_of_a_have_other_predecessors:
            return False
        return True

    def _apply_merge_in_place(self, G, a, b, merged_type=None):
        # Node a will absorb node b in the merge.
        G.nodes[a]["token_type"] = (
            merged_type
            if merged_type is not None
            else merge_token_types(G.nodes[a]["token_type"], G.nodes[b]["token_type"])
        )

        # Remove old successors of a.
        for succ in list(G.successors(a)):
            G.remove_edge(a, succ)

        # Remove old predecessors of b.
        for pred in list(G.predecessors(b)):
            G.remove_edge(pred, b)

        # Migrate successors of b to the merge node.
        for succ in list(G.successors(b)):
            if succ != a:
                G.add_edge(a, succ)
            G.remove_edge(b, succ)

        # Remove b.
        G.remove_node(b)
        return G

    def all_nodes_are_reachable_from_start(self, G, start_node_index=-1):
        return len(nx.descendants(G, start_node_index)) + 1 == len(G.nodes)

    # Merges token types A -> B in a single DAG while ensuring REMI-recoverability for monoids.
    # If finicky, checks connectivity after every merge before deciding to keep that merge.
    # Otherwise, performs all merges of the type and checks connectivity at the end.
    # If non-finicky fails, falls back to finicky, ensuring that the result is a connected graph,
    # and updating the blocked pairs accordingly.
    def _merge_directed_type_pair_edges_single_dag(self, G, type_a, type_b, blocked_pairs, merged_type, finicky=False):
        result_G = G.copy()

        candidate_edges = [
            (a, b)
            for (a, b) in result_G.edges()
            if result_G.nodes[a]["token_type"] == type_a
            and result_G.nodes[b]["token_type"] == type_b
            and result_G.nodes[a]["token_type"] != start_token_type  # Don't merge from the start token.
            and result_G.nodes[b]["token_type"] != end_token_type  # Don't merge into the end token.
            and (a, b) not in blocked_pairs  # Blocked edges caused connectivity issues in earlier merge attempts.
        ]

        for a, b in candidate_edges:
            if not result_G.has_edge(a, b):
                continue  # The edge existed when candidate_edges was created, but a merge has since removed it.

            if not self._candidate_merge_passes_heuristic(result_G, a, b):
                blocked_pairs.add((a, b))
                continue

            candidate_G = None

            if finicky:
                # The finicky case is slower, but guarantees that all nodes remain reachable from START.
                # Each candidate merge is applied to a candidate_G copy of the result_G, and we keep that updated
                # candidate only if it passes a connectivity check; otherwise we skip that specific merge.
                candidate_G = result_G.copy()
            else:
                # We will operate on the result_G in-place and check connectivity at the end, falling back to a
                # finicky merge if START no longer reaches all nodes.
                candidate_G = result_G

            # Node a will absorb node b in the merge and its type will be merged_type.
            # This operation mutates candidate_G in-place, to enhance performance in the non-finicky case.
            self._apply_merge_in_place(candidate_G, a, b, merged_type=merged_type)

            if finicky:
                # Check if all nodes are still reachable from the start of the score. If not, then this merge is not allowed.
                if self.all_nodes_are_reachable_from_start(candidate_G):
                    result_G = candidate_G
                else:
                    blocked_pairs.add((a, b))
            else:
                # candidate_G is just a reference to result_G in the non-finicky case, so this isn't really necessary.
                result_G = candidate_G

        # The finicky case already checked the last merge (and all others), so this check applies only to the non-finicky case.
        if not finicky and not self.all_nodes_are_reachable_from_start(result_G):
            # If the non-finicky attempt failed, we fall back to finicky to apply just the merges that we can.
            return self._merge_directed_type_pair_edges_single_dag(
                G, type_a, type_b, blocked_pairs, merged_type, finicky=True
            )

        return result_G

    def merge_directed_type_pair_edges(self, graphs, per_graph_blocked_pairs, type_a, type_b, merged_type):
        results = []
        for G, blocked_pairs, graph_index in zip(graphs, per_graph_blocked_pairs, range(len(graphs))):
            # print(f"Merging {type_a} and {type_b} in graph {graph_index}")
            results.append(
                self._merge_directed_type_pair_edges_single_dag(
                    G, type_a, type_b, blocked_pairs, merged_type, finicky=False
                )
            )
        return results

    def learn_merged_token(self, type_a, type_b):
        merged_type = merge_token_types(type_a, type_b)
        self.vocabulary[merged_type] = {"components": (type_a, type_b)}
        return merged_type

    # Iteratively applies REMIGraphBPE on the score graphs until the vocabulary reaches the target size.
    def learn_and_apply_vocabulary(self, graphs=None):
        graphs = self.original_graphs if graphs is None else graphs

        while len(self.vocabulary) < self.target_learned_vocab_size:
            print(f"Learned vocab size is {len(self.vocabulary)} of target {self.target_learned_vocab_size}")

            pair_counts = self.count_type_pairs_of_potentially_mergable_edges(
                self.graphs_with_merges, self.per_graph_blocked_pairs
            )

            total_candidate_pairs = sum(pair_counts.values())
            if total_candidate_pairs == 0:
                print(f"No more pairs to attempt merging. Stopping with vocabulary size: {len(self.vocabulary)}")
                break

            (type_a, type_b), _ = pair_counts.most_common(1)[0]

            merged_type = self.learn_merged_token(type_a, type_b)

            # self.log_status_update("Pre merge", type_a, type_b)

            self.graphs_with_merges = self.merge_directed_type_pair_edges(
                self.graphs_with_merges, self.per_graph_blocked_pairs, type_a, type_b, merged_type
            )

            # self.log_status_update("After merge", type_a, type_b)

        print(f"REMIGraphBPE completed with {len(self.vocabulary)} vocabulary learned.")
        return self.graphs_with_merges

    # Applies the learned vocabulary to the given graphs.
    def compress_graphs(self, graphs):
        results = [G.copy() for G in graphs]
        per_graph_blocked_pairs = [set() for _ in range(len(graphs))]

        # This loop implicitly relies on the Python feature that the entries of dictionaries are ordered by insertion order,
        # to apply the learned merges to the given graphs in the order that they were learned from the training corpus.
        for merged_type, details in remi_graph_bpe.vocabulary.items():
            type_a, type_b = details["components"]
            results = self.merge_directed_type_pair_edges(results, per_graph_blocked_pairs, type_a, type_b, merged_type)
        return results

    def serialize_to_1d_token_sequence(self, G):

        # Confirm that the graph is a DAG.
        if not nx.is_directed_acyclic_graph(G):
            raise ValueError("The input graph must be a directed acyclic graph (DAG).")

        def serialize_graph_nodes(G, drop_start_end=True):

            # Zip and sort the nodes with their depths.
            annotate_nodes_with_depths(G)
            depth_node_tuples = zip([data["depth"] for _, data in G.nodes(data=True)], G.nodes)
            sorted_depth_node_tuples = sorted(depth_node_tuples)

            result_token_ids = []
            result_token_types = []
            seen_nodes = set()
            within_simultaneous = False
            previous_depth = sorted_depth_node_tuples[0][0] if len(sorted_depth_node_tuples) > 0 else None

            for i, (current_depth, node) in enumerate(sorted_depth_node_tuples):
                # Don't process nodes twice. There is some logic that sometimes skips ahead, and in those cases the nodes are marked as seen.
                if node in seen_nodes:
                    continue
                seen_nodes.add(node)

                # Detect whether the current node is at a different depth than the previous one.
                if current_depth != previous_depth and within_simultaneous:
                    result_token_ids.append(self.simultaneous_end_token)
                    result_token_types.append(self.simultaneous_end_token_type)
                    within_simultaneous = False
                previous_depth = current_depth

                # Detect whether we are entering a section of simultaneous nodes.
                next_depth = sorted_depth_node_tuples[i + 1][0] if i + 1 < len(sorted_depth_node_tuples) else None
                next_node_has_same_depth = next_depth == current_depth
                if next_node_has_same_depth and not within_simultaneous:
                    if len(result_token_ids) > 0 and result_token_ids[-1] == self.simultaneous_end_token:
                        result_token_ids[-1] = self.simultaneous_end_begin_janus_token
                        result_token_types[-1] = self.simultaneous_end_begin_janus_token_type
                    else:
                        result_token_ids.append(self.simultaneous_start_token)
                        result_token_types.append(self.simultaneous_start_token_type)
                    within_simultaneous = True

                result_token_ids.append(node)
                result_token_types.append(G.nodes[node]["token_type"])

                # If the node has a monoid of successors of the same depth, consume that path so that
                # these related nodes are captured together. E.g., position -> pitch -> velocity -> duration
                # are the same depth and shouldn't have edges interfering between their monoid and the broader graph.
                iterate_from_node = node
                while True:
                    successors = [succ for succ in G.successors(iterate_from_node)]
                    if len(successors) != 1:
                        break
                    succ = successors[0]
                    if succ in seen_nodes:
                        break
                    if G.nodes[succ]["depth"] != current_depth:
                        break
                    seen_nodes.add(succ)
                    result_token_ids.append(self.continue_sequence_token)
                    result_token_ids.append(succ)
                    result_token_types.append(self.continue_sequence_token_type)
                    result_token_types.append(G.nodes[succ]["token_type"])
                    iterate_from_node = succ

            if within_simultaneous:
                result_token_ids.append(self.simultaneous_end_token)
                result_token_types.append(self.simultaneous_end_token_type)

            # The START and END tokens are just to assist in manipulating the graph.
            # They are not essential to the 1d token sequence, so we can drop them.
            if drop_start_end:
                return result_token_ids[1:-1], result_token_types[1:-1]

            return result_token_ids, result_token_types

        special_non_remi_token_types = set(
            [
                start_token_type,
                end_token_type,
                self.continue_sequence_token_type,
                self.simultaneous_start_token_type,
                self.simultaneous_end_token_type,
                self.simultaneous_end_begin_janus_token_type,
            ]
        )

        # is_remi = lambda token: token not in special_non_remi_token_types
        def node_sequence_to_remi_tokens(node_tokens_types_1d_sequence):
            non_special_tokens = [
                token for token in node_tokens_types_1d_sequence if token not in special_non_remi_token_types
            ]
            return [
                token
                for remi_tokens in [parse_remi_tokens_from_token_type(token) for token in non_special_tokens]
                for token in remi_tokens
            ]

        node_tokens_ids_sequence, node_tokens_types_sequence = serialize_graph_nodes(G)
        # print("node_tokens_ids_sequence", node_tokens_ids_sequence)
        # print("node_tokens_types_sequence", node_tokens_types_sequence)
        extracted_remi = node_sequence_to_remi_tokens(node_tokens_types_sequence)
        # print("\nextracted_remi", extracted_remi)
        return node_tokens_ids_sequence, node_tokens_types_sequence, extracted_remi

    # Expand compressed score graph. Note that the REMIGraphBPE compression loses edges that are
    # not essential for recovering the underlying score.


# Demo DAG corpus.
def create_dag_corpus():
    # G1 = nx.DiGraph()
    # G1.add_node(-1, token_type=start_token_type)
    # G1.add_node(0, token_type="Bar_1")
    # G1.add_node(1, token_type="Position_0")
    # G1.add_node(2, token_type="Pitch_C4")
    # G1.add_node(3, token_type="Duration_Quarter")
    # G1.add_node(4, token_type="Position_12")
    # G1.add_node(5, token_type="Pitch_E4")
    # G1.add_node(6, token_type="Duration_Quarter")
    # G1.add_node(7, token_type=end_token_type)
    # G1.add_edges_from([(-1, 0), (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (6, 7)])

    G2 = nx.DiGraph()
    G2.add_node(-1, token_type=start_token_type)
    G2.add_node(0, token_type="Bar_2")
    G2.add_node(1, token_type="Position_0")
    G2.add_node(2, token_type="Pitch_50")
    G2.add_node(3, token_type="Duration_Eighth")
    G2.add_node(4, token_type="Position_6")
    G2.add_node(5, token_type="Pitch_60")
    G2.add_node(6, token_type="Duration_Eighth")
    G2.add_node(7, token_type="Position_0")
    G2.add_node(8, token_type="Pitch_50")
    G2.add_node(9, token_type="Position_0")
    G2.add_node(10, token_type="Pitch_50")
    G2.add_node(11, token_type=end_token_type)
    G2.add_edges_from(
        [
            (-1, 0),
            (0, 1),
            (1, 2),
            (2, 3),
            (0, 4),
            (4, 5),
            (5, 6),
            (0, 7),
            (7, 8),
            (0, 9),
            (9, 10),
            (2, 7),
            (2, 9),
            (6, 11),
            (10, 11),
        ]
    )

    # G3 = nx.DiGraph()
    # G3.add_node(0, token_type="Bar_3")
    # G3.add_node(1, token_type="Position_0")
    # G3.add_node(2, token_type="Pitch_C4")
    # G3.add_node(3, token_type="Duration_Eighth")
    # G3.add_node(4, token_type="Position_6")
    # G3.add_node(5, token_type="Pitch_G4")
    # G3.add_node(6, token_type="Duration_Eighth")
    # G3.add_edges_from([(0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6)])

    # return [G1, G2, G3]


    # G = load_with_vertical_and_horizontal_topology("./twinkle.mid")
    # return [G]
    # G = load_as_monoidal_remi_score_graph("./hot-cross-buns.mid")
    # G = load_as_monoidal_remi_score_graph("./data/maestro-v3.0.0/2013/ORIG-MIDI_03_7_8_13_Group__MID--AUDIO_18_R2_2013_wav--3.midi")

    graphs = [load_as_monoidal_remi_score_graph(midi) for midi in train_split[10:30]]
    # G = graphs[0]
    # draw_graph(G , "title")
    # print(G)
    return graphs
    # return [G] + [load_as_monoidal_remi_score_graph(midi) for midi in train_split[:1]]


# Run REMIGraphBPE on the small corpus.
dag_corpus = create_dag_corpus()
remi_graph_bpe = REMIGraphBPE(target_learned_vocab_size=50, graph_corpus=dag_corpus)

compressed_graphs = remi_graph_bpe.learn_and_apply_vocabulary()

# Print the learned vocabulary.
print("\nLearned REMIGraphBPE Vocabulary:")
for token, details in remi_graph_bpe.vocabulary.items():
    source_types = f"({details['components'][0]}, {details['components'][1]})"
    print(f"{token} <- {source_types}")


print(f"Nodes prior to REMIGraphBPE: {len(dag_corpus[0].nodes)}")
print(f"Nodes after REMIGraphBPE: {len(compressed_graphs[0].nodes)}")
draw_graph(dag_corpus[0], "Transformed DAG (Before REMIGraphBPE)")

node_tokens_ids_sequence, node_tokens_types_sequence, extracted_remi = remi_graph_bpe.serialize_to_1d_token_sequence(
    compressed_graphs[0]
)

# print("compressed token types sequence", node_tokens_types_sequence)

# print("extracted_remi", extracted_remi)
# remi_token_type_sequence_to_midi(extracted_remi)
print("done")
draw_graph(compressed_graphs[0], "Transformed DAG (After REMIGraphBPE)")

# print(train_split[10:11])

# additional_graphs = [load_as_monoidal_remi_score_graph("./hot-cross-buns.mid")]
# additional_graphs = [load_as_monoidal_remi_score_graph(midi) for midi in train_split[:2]]
# for additional in additional_graphs:
#     print(f"Nodes before REMIGraphBPE: {len(additional.nodes)}")
# additional_compressed_graphs = remi_graph_bpe.compress_graphs(additional_graphs)
# for additional in additional_compressed_graphs:
#     print(f"Nodes after REMIGraphBPE: {len(additional.nodes)}")
# draw_graph(additional, "Transformed DAG (After REMIGraphBPE)")
# draw_graph(dag_corpus[1], "Transformed DAG (Before REMIGraphBPE)")
# draw_graph(compressed_graphs[1], "Transformed DAG (After REMIGraphBPE)")
# draw_graph(dag_corpus[2], "Transformed DAG (Before REMIGraphBPE)")
# draw_graph(compressed_graphs[2], "Transformed DAG (After REMIGraphBPE)")
