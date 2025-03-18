import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
from remi_score_graph import (
    load_as_monoidal_remi_score_graph,
    start_token_type,
    end_token_type,
)
from util.maestro_train_dev_test_splits import train_split


# Visualize one of the transformed DAGs.
def draw_graph(G, title):
    plt.figure(figsize=(8, 5))
    pos = nx.spring_layout(G, seed=42)
    labels = {node: f"{node}\n{data['token_type']}" for node, data in G.nodes(data=True)}
    nx.draw(
        G,
        pos,
        with_labels=True,
        labels=labels,
        node_color="lightblue",
        edge_color="gray",
        node_size=2000,
        font_size=10,
    )
    plt.title(title)
    plt.show()


def split_type_identifier(type_id):
    if type_id.startswith("(") and type_id.endswith(")"):
        return type_id[1:-1].split(" ")
    return [type_id]


def merged_type_identifier(type_a, type_b):
    a_subtypes = split_type_identifier(type_a)
    b_subtypes = split_type_identifier(type_b)
    return f"({' '.join(a_subtypes)} {' '.join(b_subtypes)})"


class GraphBPE:
    def __init__(self, target_vocab_size, graph_corpus):
        self.target_vocab_size = target_vocab_size
        self.vocabulary = {}  # Stores the learned vocabulary.
        self.original_graphs = [G.copy() for G in graph_corpus]
        self.graphs_with_merges = [G.copy() for G in graph_corpus]
        # When a merge attempt causes connectivity issues, we block that pair from being merged again.
        self.per_graph_blocked_pairs = [set() for _ in range(len(graph_corpus))]

    def count_type_pairs_of_potentially_mergable_edges(self, graphs, per_graph_blocked_pairs):
        print(graphs, per_graph_blocked_pairs)
        for a in zip(graphs, per_graph_blocked_pairs):
            print(a)
        return Counter(
            (G.nodes[a]["token_type"], G.nodes[b]["token_type"])
            for (G, blocked_pairs) in zip(graphs, per_graph_blocked_pairs)
            for (a, b) in G.edges()
            # blocked edges caused connectivity issues in an earlier merge attempt
            # We also prevent merges with the start node, in case its special role is valuable
            if (a, b) not in blocked_pairs
            and G.nodes[a]["token_type"] != start_token_type
            and G.nodes[b]["token_type"] != end_token_type
        )

    # Merges token types A -> B in a single DAG while ensuring REMI-recoverability for monoids.
    # If finicky, checks connectivity after every merge before deciding to keep that merge.
    # Otherwise, performs all merges of the type and checks connectivity at the end.
    # If non-finicky fails, falls back to finicky, ensuring that the result is a connected graph,
    # and updating the blocked pairs accordingly.
    def merge_directed_type_pair_edges_single_dag(self, G, type_a, type_b, blocked_pairs, finicky=False):
        result = G.copy()

        merged_type = merged_type_identifier(type_a, type_b)

        # Skip the start and end nodes when merging, to preserve them.
        nodes_a = [
            a
            for a in list(G.nodes)
            if G.nodes[a]["token_type"] == type_a
            and a not in blocked_pairs
            and G.nodes[a]["token_type"] != start_token_type
        ]
        nodes_b = [
            b
            for b in list(G.nodes)
            if G.nodes[b]["token_type"] == type_b
            and b not in blocked_pairs
            and G.nodes[b]["token_type"] != end_token_type
        ]
        candidate_edges = [(a, b) for a in nodes_a for b in nodes_b if G.has_edge(a, b) and (a, b) not in blocked_pairs]

        for a, b in candidate_edges:
            if not result.has_edge(a, b):
                continue

            print(f"Finicky: {finicky}")
            print(f"Current Graph, about to merge {a} and {b}")
            # draw_graph(result, f"Current Graph, about to merge {a} and {b}")

            # Heuristic to prevent obvious connectivity issues.
            # We would drop edges from A during the merge. If A is the only predecessor of one of its successors (besides B),
            # then removing the edges from A will disconnect that successor and make the graph disconnected.
            other_successors_of_a = [succ for succ in result.successors(a) if succ != b]
            other_predecessors_of_other_successors_of_a = [
                pred for succ in other_successors_of_a for pred in result.predecessors(succ) if pred != a
            ]
            a_has_other_successors = len(other_successors_of_a) != 0
            other_successors_of_a_have_other_predecessors = len(other_predecessors_of_other_successors_of_a) != 0
            if a_has_other_successors and not other_successors_of_a_have_other_predecessors:
                blocked_pairs.add((a, b))
                continue

            # # Heuristic to prevent obvious connectivity issues.
            # alternate_predecessors_of_b = [
            #     pred for pred in result.predecessors(b) if pred != a
            # ]
            # successors_of_alternate_predecessors_of_b = [
            #     succ
            #     for pred in alternate_predecessors_of_b
            #     for succ in result.successors(pred)
            #     if succ != a
            # ]
            # alternate_successors_of_alternate_predecessors_of_b = [
            #     succ for succ in successors_of_alternate_predecessors_of_b if succ != b
            # ]
            # if (
            #     len(alternate_predecessors_of_b) != 0
            #     and len(alternate_successors_of_alternate_predecessors_of_b) == 0
            # ):
            #     blocked_pairs.add((a, b))
            #     continue

            # The heuristics pass, but we will still need to ensure that the graph remains connected after the merge.

            candidate = None

            if finicky:
                # We will make the changes on a copy and check connectivity after this specific merge before committing them to the result.
                # This guarantees that the result is a connected graph, by skipping specific bad merges, but it is slower.
                candidate = result.copy()
            else:
                # We will operate on the result in-place and check connectivity at the end, falling back to a finicky merge if the graph has become disconnected.
                candidate = result

            # Node a will absorb node b in the merge.
            candidate.nodes[a]["token_type"] = merged_type

            # Add edges from predecessors of a to the merge node.
            # Nothing to do since a is absorbing b and a already has all of its predecessors linked.

            # Remove old successors of a.
            for succ in list(candidate.successors(a)):
                candidate.remove_edge(a, succ)

            # Remove old predecessors of b.
            for pred in list(candidate.predecessors(b)):
                candidate.remove_edge(pred, b)

            # Migrate successors of b to the merge node.
            for succ in list(candidate.successors(b)):
                if succ != a:
                    candidate.add_edge(a, succ)
                candidate.remove_edge(b, succ)

            # Remove b.
            candidate.remove_node(b)

            if finicky:
                # Check if the graph is still weakly connected. If it isn't, then this merge is not allowed.
                if len(nx.descendants(candidate, -1)) + 1 == len(candidate.nodes):
                    result = candidate
                else:
                    print(
                        "Finicky solved connectivity issue by skipping merge of nodes:",
                        (a, b),
                    )
                    blocked_pairs.add((a, b))
            else:
                # Not really necessary since candidate is just a reference to result in the non-finicky case.
                result = candidate

        if not finicky and not len(nx.descendants(result, -1)) + 1 == len(result.nodes):
            print("Non-finicky failed, falling back to finicky.")
            # One or more of the merges caused the graph to become disconnected.
            # If this were a finicky merge, we would have already skipped the invalid merge.
            return self.merge_directed_type_pair_edges_single_dag(G, type_a, type_b, blocked_pairs, finicky=True)

        return result

    def merge_directed_type_pair_edges(self, graphs, type_a, type_b):
        merged_type = merged_type_identifier(type_a, type_b)
        self.vocabulary[merged_type] = {"components": (type_a, type_b)}
        results = []
        for G, blocked_pairs, graph_index in zip(graphs, self.per_graph_blocked_pairs, range(len(graphs))):
            print(f"Merging {type_a} and {type_b} in graph {graph_index}")
            results.append(
                self.merge_directed_type_pair_edges_single_dag(G, type_a, type_b, blocked_pairs, finicky=False)
            )
        return results

    # Iteratively applies GraphBPE until the vocabulary reaches the target size.
    def apply_graphbpe(self, graphs=None):
        graphs = self.original_graphs if graphs is None else graphs

        while len(self.vocabulary) < self.target_vocab_size:
            print(
                "Current vocabulary size:",
                len(self.vocabulary),
                "Target vocabulary size:",
                self.target_vocab_size,
            )

            pair_counts = self.count_type_pairs_of_potentially_mergable_edges(
                self.graphs_with_merges, self.per_graph_blocked_pairs
            )

            total_candidate_pairs = sum(pair_counts.values())
            if total_candidate_pairs == 0:
                print(f"No more pairs to attempt merging. Stopping with vocabulary size: {len(self.vocabulary)}")
                break

            (type_a, type_b), _ = pair_counts.most_common(1)[0]

            pre_merge_blocked_pairs_count = sum([len(blocked_pairs) for blocked_pairs in self.per_graph_blocked_pairs])

            print(
                f"{pair_counts[(type_a, type_b)]} out of {total_candidate_pairs} are ({type_a}, {type_b}) and there are {pre_merge_blocked_pairs_count} blocked pairs."
            )
            print(f"Merging {type_a} and {type_b} into {merged_type_identifier(type_a, type_b)}")

            print("before")
            print(self.graphs_with_merges)
            self.graphs_with_merges = self.merge_directed_type_pair_edges(self.graphs_with_merges, type_a, type_b)
            print("after")
            print(self.graphs_with_merges)

            after_merge_pair_count = self.count_type_pairs_of_potentially_mergable_edges(
                self.graphs_with_merges, self.per_graph_blocked_pairs
            )[(type_a, type_b)]

            post_merge_blocked_pairs_count = sum([len(blocked_pairs) for blocked_pairs in self.per_graph_blocked_pairs])

            print(
                f"Post-merge, {after_merge_pair_count} candidate pairs are ({type_a}, {type_b}) and there are {post_merge_blocked_pairs_count} blocked pairs."
            )

        print(f"GraphBPE completed with {len(self.vocabulary)} vocabulary learned.")
        return self.graphs_with_merges


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

    # G2 = nx.DiGraph()
    # G2.add_node(-1, token_type=start_token_type)
    # G2.add_node(0, token_type="Bar_2")
    # G2.add_node(1, token_type="Position_0")
    # G2.add_node(2, token_type="Pitch_D4")
    # G2.add_node(3, token_type="Duration_Eighth")
    # G2.add_node(4, token_type="Position_6")
    # G2.add_node(5, token_type="Pitch_G4")
    # G2.add_node(6, token_type="Duration_Eighth")
    # G2.add_node(7, token_type="Position_0")
    # G2.add_node(8, token_type="Pitch_D4")
    # G2.add_node(9, token_type="Position_0")
    # G2.add_node(10, token_type="Pitch_D4")
    # G2.add_node(11, token_type=end_token_type)
    # G2.add_edges_from(
    #     [
    #         (-1, 0),
    #         (0, 1),
    #         (1, 2),
    #         (2, 3),
    #         (0, 4),
    #         (4, 5),
    #         (5, 6),
    #         (0, 7),
    #         (7, 8),
    #         (0, 9),
    #         (9, 10),
    #         (2, 7),
    #         (2, 9),
    #         (6, 11),
    #         (10, 11),
    #     ]
    # )

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

    # return [G1]

    G = load_as_monoidal_remi_score_graph("./hot-cross-buns.mid")
    return [G]
    # print([G] + [load_as_monoidal_remi_score_graph(midi) for midi in train_split[:1]])
    # return [G] + [load_as_monoidal_remi_score_graph(midi) for midi in train_split[:5]]


# Run GraphBPE on the small corpus.
dag_corpus = create_dag_corpus()
graphbpe = GraphBPE(target_vocab_size=19, graph_corpus=dag_corpus)

compressed_graphs = graphbpe.apply_graphbpe()

# Print the learned vocabulary.
print("\nLearned GraphBPE Vocabulary:")
for token, details in graphbpe.vocabulary.items():
    source_types = f"({details['components'][0]} {details['components'][1]})"
    print(f"{token} <- {source_types}")


print(f"Nodes prior to GraphBPE: {len(dag_corpus[0].nodes)}")
print(f"Nodes after GraphBPE: {len(compressed_graphs[0].nodes)}")
draw_graph(dag_corpus[0], "Transformed DAG (Before GraphBPE)")
draw_graph(compressed_graphs[0], "Transformed DAG (After GraphBPE)")
# draw_graph(dag_corpus[1], "Transformed DAG (Before GraphBPE)")
# draw_graph(compressed_graphs[1], "Transformed DAG (After GraphBPE)")
# draw_graph(dag_corpus[2], "Transformed DAG (Before GraphBPE)")
# draw_graph(compressed_graphs[2], "Transformed DAG (After GraphBPE)")
