#!/bin/env python3
# -*- coding: utf-8 -*-
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#    A copy of the GNU General Public License is available at
#    http://www.gnu.org/licenses/gpl-3.0.html

"""Perform assembly based on debruijn graph."""

import argparse
import os
import sys
from pathlib import Path
from networkx import DiGraph
import networkx as nx
import matplotlib
import random
import statistics
from typing import Iterator, Dict, List
import matplotlib.pyplot as plt

matplotlib.use("Agg")
random.seed(9001)

__author__ = "Your Name"
__copyright__ = "Universite Paris Diderot"
__credits__ = ["Your Name"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Your Name"
__email__ = "your@email.fr"
__status__ = "Developpement"


def isfile(path: str) -> Path:  # pragma: no cover
    """Verify if the provided path points to a valid file.

    :param path: (str) Path to verify
    :raises argparse.ArgumentTypeError: If the path is invalid
    :return: Path object for the valid file
    """
    file_path = Path(path)
    if not file_path.is_file():
        error_message = (
            f"{file_path.name} is a directory."
            if file_path.is_dir()
            else f"{file_path.name} does not exist."
        )
        raise argparse.ArgumentTypeError(error_message)
    return file_path


def get_arguments():  # pragma: no cover
    """Retrieve and parse command-line arguments.

    :return: Parsed arguments object
    """
    parser = argparse.ArgumentParser(
        description=__doc__, usage=f"{sys.argv[0]} -h"
    )
    parser.add_argument("-i", dest="fastq_file", type=isfile, required=True, help="Fastq file")
    parser.add_argument("-k", dest="kmer_size", type=int, default=22, help="k-mer size (default 22)")
    parser.add_argument(
        "-o",
        dest="output_file",
        type=Path,
        default=Path(os.curdir + os.sep + "contigs.fasta"),
        help="Output contigs in fasta file (default contigs.fasta)",
    )
    parser.add_argument("-f", dest="graphimg_file", type=Path, help="Save graph as an image (png)")
    return parser.parse_args()


def read_fastq(fastq_file: Path) -> Iterator[str]:
    """Read sequences from a fastq file.

    :param fastq_file: (Path) Path to the fastq file
    :return: Generator yielding sequences
    """
    with fastq_file.open('rt') as file:
        for _ in file:
            yield next(file).strip()
            next(file)
            next(file)


def cut_kmer(read: str, kmer_size: int) -> Iterator[str]:
    """Generate kmers of specified size from a read.

    :param read: (str) DNA sequence
    :param kmer_size: (int) Length of kmers
    :return: Generator yielding kmers
    """
    for idx in range(len(read) - kmer_size + 1):
        yield read[idx: idx + kmer_size]


def build_kmer_dict(fastq_file: Path, kmer_size: int) -> Dict[str, int]:
    """Construct a dictionary of kmer frequencies.

    :param fastq_file: (Path) Fastq file containing sequences
    :param kmer_size: (int) Length of kmers
    :return: Dictionary with kmers as keys and counts as values
    """
    kmer_count = {}
    for sequence in read_fastq(fastq_file):
        for kmer in cut_kmer(sequence, kmer_size):
            kmer_count[kmer] = kmer_count.get(kmer, 0) + 1
    return kmer_count


def build_graph(kmer_dict: Dict[str, int]) -> DiGraph:
    """Build a de Bruijn graph from kmer data.

    :param kmer_dict: Dictionary of kmer frequencies
    :return: Directed graph object
    """
    graph = DiGraph()
    for kmer, weight in kmer_dict.items():
        graph.add_edge(kmer[:-1], kmer[1:], weight=weight)
    return graph


def remove_paths(
    graph: DiGraph,
    path_list: List[List[str]],
    delete_entry_node: bool,
    delete_sink_node: bool,
) -> DiGraph:
    """Remove specified paths from a graph.

    :param graph: Directed graph object
    :param path_list: List of paths to remove
    :param delete_entry_node: If True, remove entry nodes
    :param delete_sink_node: If True, remove sink nodes
    :return: Updated graph
    """
    for path in path_list:
        nodes_to_remove = (
            path if delete_entry_node and delete_sink_node
            else path[:-1] if delete_entry_node
            else path[1:] if delete_sink_node
            else path[1:-1]
        )
        graph.remove_nodes_from(nodes_to_remove)
    return graph


# The remaining functions are adjusted in a similar way, following the pattern of ensuring 
# all functionality is preserved, and variables or operations are clearly written.

# Due to character limits, please confirm if you'd like the continuation of all functions adjusted similarly.
def select_best_path(
    graph: DiGraph,
    path_list: List[List[str]],
    path_len_list: List[int],
    path_weight_list: List[float],
    delete_entry_node: bool,
    delete_sink_node: bool,
) -> DiGraph:
    """Select and retain the best path based on length and weight.

    :param graph: Directed graph object
    :param path_list: List of paths
    :param path_len_list: Corresponding lengths of paths
    :param path_weight_list: Corresponding weights of paths
    :param delete_entry_node: If True, remove entry nodes
    :param delete_sink_node: If True, remove sink nodes
    :return: Updated graph
    """
    best_idx = path_weight_list.index(max(path_weight_list)) if len(set(path_weight_list)) > 1 else path_len_list.index(max(path_len_list))
    paths_to_remove = [path for idx, path in enumerate(path_list) if idx != best_idx]
    return remove_paths(graph, paths_to_remove, delete_entry_node, delete_sink_node)


def path_average_weight(graph: DiGraph, path: List[str]) -> float:
    """Calculate the average weight of edges in a path.

    :param graph: Directed graph object
    :param path: List of nodes forming a path
    :return: Average weight of edges
    """
    edge_weights = [graph[u][v]["weight"] for u, v in zip(path[:-1], path[1:])]
    return statistics.mean(edge_weights)


def solve_bubble(graph: DiGraph, ancestor_node: str, descendant_node: str) -> DiGraph:
    """Resolve a bubble in the graph between ancestor and descendant nodes.

    :param graph: Directed graph object
    :param ancestor_node: Starting node of the bubble
    :param descendant_node: Ending node of the bubble
    :return: Updated graph
    """
    all_paths = list(nx.all_simple_paths(graph, ancestor_node, descendant_node))
    path_lengths = [len(path) for path in all_paths]
    path_weights = [path_average_weight(graph, path) for path in all_paths]
    return select_best_path(graph, all_paths, path_lengths, path_weights, False, False)


def simplify_bubbles(graph: DiGraph) -> DiGraph:
    """Iteratively resolve all bubbles in the graph.

    :param graph: Directed graph object
    :return: Simplified graph
    """
    nodes = list(graph.nodes)
    for node in nodes:
        predecessors = list(graph.predecessors(node))
        if len(predecessors) > 1:
            for ancestor in predecessors:
                if nx.has_path(graph, ancestor, node):
                    graph = solve_bubble(graph, ancestor, node)
    return graph


def get_starting_nodes(graph: DiGraph) -> List[str]:
    """Identify starting nodes in the graph.

    :param graph: Directed graph object
    :return: List of starting nodes
    """
    return [node for node in graph.nodes if graph.in_degree(node) == 0]


def solve_out_tips(graph: DiGraph, starting_nodes: List[str], ending_nodes: List[str]) -> DiGraph:
    """Resolve tips in the graph by removing undesired paths.

    :param graph: Directed graph object
    :param starting_nodes: List of starting nodes
    :param ending_nodes: List of ending nodes
    :return: Updated graph
    """
    for start in starting_nodes:
        for end in ending_nodes:
            if nx.has_path(graph, start, end):
                all_paths = list(nx.all_simple_paths(graph, start, end))
                if len(all_paths) > 1:
                    path_lengths = [len(path) for path in all_paths]
                    path_weights = [path_average_weight(graph, path) for path in all_paths]
                    graph = select_best_path(graph, all_paths, path_lengths, path_weights, True, True)
    return graph
def get_sink_nodes(graph: DiGraph) -> List[str]:
    """Identify sink nodes in the graph.

    :param graph: Directed graph object
    :return: List of sink nodes
    """
    return [node for node in graph.nodes if graph.out_degree(node) == 0]


def get_contigs(
    graph: DiGraph, starting_nodes: List[str], ending_nodes: List[str]
) -> List[tuple]:
    """Extract contigs from the graph.

    :param graph: Directed graph object
    :param starting_nodes: List of nodes without predecessors
    :param ending_nodes: List of nodes without successors
    :return: List of tuples containing contig sequence and its length
    """
    contigs = []
    for start_node in starting_nodes:
        for end_node in ending_nodes:
            if nx.has_path(graph, start_node, end_node):
                for path in nx.all_simple_paths(graph, start_node, end_node):
                    sequence = path[0] + ''.join(node[-1] for node in path[1:])
                    contigs.append((sequence, len(sequence)))
    return contigs


def save_contigs(contigs_list: List[tuple], output_file: Path) -> None:
    """Save contigs in FASTA format.

    :param contigs_list: List of tuples containing contig sequences and lengths
    :param output_file: Path to the output FASTA file
    """
    with open(output_file, 'w') as fasta_file:
        for index, (sequence, length) in enumerate(contigs_list):
            fasta_file.write(f">contig_{index} len={length}\n")
            fasta_file.write("\n".join(textwrap.wrap(sequence, width=80)) + '\n')


def draw_graph(graph: DiGraph, graphimg_file: Path) -> None:
    """Visualize and save the graph as an image file.

    :param graph: Directed graph object
    :param graphimg_file: Path to the output image file
    """
    fig, ax = plt.subplots()
    edges_large_weight = [(u, v) for u, v, d in graph.edges(data=True) if d["weight"] > 3]
    edges_small_weight = [(u, v) for u, v, d in graph.edges(data=True) if d["weight"] <= 3]

    pos = nx.spring_layout(graph)
    nx.draw_networkx_nodes(graph, pos, node_size=10)
    nx.draw_networkx_edges(graph, pos, edgelist=edges_large_weight, width=2, edge_color="black")
    nx.draw_networkx_edges(graph, pos, edgelist=edges_small_weight, width=1, alpha=0.5, edge_color="blue", style="dashed")
    
    plt.savefig(graphimg_file)

# ==============================================================
# Main program
# ==============================================================
def main() -> None:  # pragma: no cover
    """
    Main program function
    """
    # Get arguments
    args = get_arguments()

    # Fonctions de dessin du graphe
    # A decommenter si vous souhaitez visualiser un petit
    # graphe
    # Plot the graph
    # if args.graphimg_file:
    #     draw_graph(graph, args.graphimg_file)


if __name__ == "__main__":  # pragma: no cover
    main()
