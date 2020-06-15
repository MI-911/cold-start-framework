import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
import pickle

if __name__ == '__main__':
    with open('../../entrypoints/dec_tree_graph.pickle', 'rb') as fp:
        dec_tree_graph = pickle.load(fp)
        graph = dec_tree_graph['graph']
        edge_labels = dec_tree_graph['edge_labels']
        node_labels = dec_tree_graph['node_labels']

        print(f'Calculating positions of {graph.number_of_nodes()} nodes...')
        # Possible layouts: twopi, neato, sfdp,
        pos = graphviz_layout(graph, prog='twopi')
        print(f'Drawing...')
        nx.draw(graph, pos=pos, with_labels=False, node_size=750, node_color='lightgrey', node_labels=node_labels)
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
        nx.draw_networkx_labels(graph, pos, labels=node_labels)
        print(f'Showing...')
        plt.savefig('dec_tree_graph.pdf')
        plt.show()
