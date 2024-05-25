import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, roc_auc_score
import dgl
import torch
import networkx as nx


# Assuming you have extracted node embeddings from your GNN model
# and stored them in a NumPy array named 'node_embeddings'
# Each row of 'node_embeddings' corresponds to the embedding of a single node

def conf_matrix(y_true, y_pred, dataset_name):
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Visualize the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=np.unique(y_true),
                yticklabels=np.unique(y_true))
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title("Confusion Matrix for Predictions of " + dataset_name + " Dataset")
    plt.show()


def vis_embeddings(node_embeddings, node_labels, dataset_name):
    # Apply t-SNE to reduce dimensionality
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embeddings_2d = tsne.fit_transform(node_embeddings)
    # Create a color map for labels
    unique_labels = np.unique(node_labels)
    num_labels = len(unique_labels)
    color_map = plt.get_cmap('tab10')  # Use a predefined colormap, e.g., 'tab10'
    colors = [color_map(i / num_labels) for i in range(num_labels)]

    # Plot t-SNE visualization with colored nodes
    plt.figure(figsize=(10, 8))
    for i, label in enumerate(unique_labels):
        indices = np.where(node_labels == label)[0]
        plt.scatter(embeddings_2d[indices, 0], embeddings_2d[indices, 1], color=colors[i], label=label, s=10)
    plt.title('t-SNE Visualization of Node Embeddings of ' + dataset_name + ' Dataset')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend()
    plt.grid(True)
    plt.show()


def vis_graph(graph):
    # Convert DGL graph to NetworkX graph
    nx_graph = graph.to_networkx(node_attrs=['label'])
    # Plot the graph
    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(nx_graph, seed=42)  # Layout for positioning nodes
    labels = nx.get_node_attributes(nx_graph, 'label')
    nx.draw(nx_graph, pos, node_color=list(labels.values()), cmap=plt.get_cmap('Set3'), node_size=100,
            font_size=10)
    plt.show()


def vis_graphs(graph):

    G = graph.to_networkx(node_attrs=['label'])
    # Create a sample graph
    # G = nx.karate_club_graph()

    # Compute the line graph of G
    L = nx.line_graph(G)

    # Plot the original graph G and its line graph L
    plt.figure(figsize=(12, 6))

    # Positioning for the original graph G
    pos_G = nx.spring_layout(G, seed=42, k=0.5)  # Adjust k for larger distances

    # Positioning for the line graph L
    pos_L = nx.spring_layout(L, seed=42, k=0.5)  # Adjust k for larger distances

    # Draw the original graph G
    plt.subplot(121)
    nx.draw(G, pos_G, node_color='blue', edge_color='lightblue', node_size=100, font_size=8)
    plt.title("Original Graph")

    # Draw the line graph L
    plt.subplot(122)
    nx.draw(L, pos_L, node_color='green', edge_color='lightgreen', node_size=100, font_size=8)
    plt.title("Line Graph")

    plt.show()


def plot_precision_recall(y_true, y_pred, num_classes):
    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    f1 = f1_score(y_true, y_pred, average=None)

    plt.figure()
    for i in range(num_classes):
        plt.plot(recall[i], precision[i], marker='.', label=f'Class {i}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.show()


def plot_macro_micro_metrics(y_true, y_pred, dataset_name):
    # Compute macro-average precision, recall, and F1-score
    macro_precision = precision_score(y_true, y_pred, average='macro')
    macro_recall = recall_score(y_true, y_pred, average='macro')
    macro_f1 = f1_score(y_true, y_pred, average='macro')

    # Compute micro-average precision, recall, and F1-score
    micro_precision = precision_score(y_true, y_pred, average='micro')
    micro_recall = recall_score(y_true, y_pred, average='micro')
    micro_f1 = f1_score(y_true, y_pred, average='micro')

    plt.figure()
    metrics = ['Precision', 'Recall', 'F1-score']
    macro_metrics = [macro_precision, macro_recall, macro_f1]
    micro_metrics = [micro_precision, micro_recall, micro_f1]
    for i, metric in enumerate(metrics):
        plt.bar([i - 0.2 for i in range(3)], macro_metrics[i], width=0.4, label='Macro')
        plt.bar([i + 0.2 for i in range(3)], micro_metrics[i], width=0.4, label='Micro')
    plt.xticks([0, 1, 2], ['Precision', 'Recall', 'F1-score'])
    plt.ylabel('Score')
    plt.title('Macro vs Micro Metrics')
    plt.legend()
    plt.show()


def plot_roc_curve(set_labels, y_pred_proba, num_classes, dataset_name):
    y_true_one_hot = label_binarize(set_labels, classes=range(num_classes))
    fpr = dict()
    tpr = dict()
    auc_roc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_one_hot[:, i], y_pred_proba[:, i])
        auc_roc[i] = roc_auc_score(y_true_one_hot[:, i], y_pred_proba[:, i])
    plt.figure()
    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i], marker='.', label=f'Class {i} (AUC = {auc_roc[i]:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve / ' + dataset_name)
    plt.legend()
    plt.show()


def plot_auc_roc_micro(set_labels, y_pred_proba, num_classes, dataset_name):
    y_true_one_hot = label_binarize(set_labels, classes=range(num_classes))
    fpr = dict()
    tpr = dict()
    auc_roc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_one_hot[:, i], y_pred_proba[:, i])
        auc_roc[i] = roc_auc_score(y_true_one_hot[:, i], y_pred_proba[:, i])

    fpr_micro, tpr_micro, _ = roc_curve(y_true_one_hot.ravel(), y_pred_proba.ravel())
    auc_roc_micro = roc_auc_score(y_true_one_hot, y_pred_proba, average='micro')
    plt.figure()
    plt.plot(fpr_micro, tpr_micro, marker='.')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Micro-average ROC Curve (AUC = {auc_roc_micro:.2f}) / ' + dataset_name)
    plt.show()


def generate_dataset(n_nodes, n_edges, feat_dim, num_classes):
    # Define the number of nodes and edges
    n_nodes = 200
    n_edges = 250

    # Create a graph with 200 nodes
    graph = dgl.DGLGraph()
    graph.add_nodes(n_nodes)

    # Generate 250 random edges
    src_nodes = np.random.choice(n_nodes, n_edges)
    dst_nodes = np.random.choice(n_nodes, n_edges)
    graph.add_edges(src_nodes, dst_nodes)

    # Set node features
    feat_dim = 3703
    node_features = np.random.rand(n_nodes, feat_dim)  # Random node features
    node_features_tensor = torch.tensor(node_features, dtype=torch.float32)  # Convert to PyTorch tensor
    graph.ndata['feat'] = node_features_tensor

    # Randomly label nodes into one of six classes
    n_classes = 6
    node_labels = np.random.randint(0, n_classes, size=n_nodes)
    node_labels_tensor = torch.tensor(node_labels, dtype=torch.long)  # Convert to PyTorch tensor
    graph.ndata['label'] = node_labels_tensor

    # Visualize the graph (optional)
    # dgl.plot(graph, node_labels=node_labels)

    # Print information about the graph
    print("Number of nodes:", graph.number_of_nodes())
    print("Number of edges:", graph.number_of_edges())
    print("Node feature shape:", graph.ndata['feat'].shape)
    print("Node label shape:", graph.ndata['label'].shape)

    return node_features, node_labels

