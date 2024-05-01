
import pandas as pd
from ogb.nodeproppred import PygNodePropPredDataset
import numpy as np
# Load the ogbn-mag dataset
dataset = PygNodePropPredDataset(name='ogbn-mag')
# Get the graph object

def process_ogbn_mag_dataset(dataset):
    data = dataset[0]
    year = data.node_year['paper'].numpy()
    train_mask = year < 2018
    edge = data.edge_index_dict['paper', 'cites', 'paper'].numpy()
    directed = np.concatenate([edge, edge[[1,0]]], axis=1)
    num_nodes = data.num_nodes_dict['paper']
    def process_sparse_idx(rel, length, base):
        sp_idx = [[] for i in range(num_nodes)]
        for i, j in rel.T:
            sp_idx[i].append(j)
        for i in range(num_nodes):
            if len(sp_idx[i]) > length:
                sp_idx[i] = sp_idx[i][0:length]
            while len(sp_idx[i]) < length:
                sp_idx[i].append(-1)
        sp_idx = np.array(sp_idx)
        sp_idx += (base + 1)
        return sp_idx

    node_id = np.arange(num_nodes).reshape(-1, 1)
    field = data.edge_index_dict[('paper', 'has_topic', 'field_of_study')].numpy()
    paper_field = process_sparse_idx(field, 10, num_nodes)
    idx_max = num_nodes + data.num_nodes_dict['field_of_study'] + 1
    author = data.edge_index_dict[('author', 'writes', 'paper')].numpy()
    paper_author = process_sparse_idx(author[[1, 0]], 10, idx_max)
    idx_max += data.num_nodes_dict['author'] + 1
    x=np.concatenate([paper_field, paper_author ,node_id], axis=1),
    y=data.y_dict["paper"].numpy().squeeze(),
    edge_index=directed,
    num_classes=dataset.num_classes
    return x,y,edge_index,num_classes


x,y,edge_index,num_classes = process_ogbn_mag_dataset(dataset)
# Get node features
node_features = x

# Get node labels
node_labels = y

# Convert node features and labels to DataFrame
node_df = pd.DataFrame(columns=["id", "feature", "class"])


print(edge_index[0].shape)

# Convert edge indices to DataFrame
edge_df = pd.DataFrame(edge_index[0].T,columns=['source_id', 'end_id'])
print(edge_df.head(5))
# Save edge DataFrame to CSV
edge_df.to_csv('ogbn_mag_edges.csv', index=False)
