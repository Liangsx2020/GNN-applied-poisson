import torch
import torch_scatter
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import MetaLayer
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.data import Data
import numpy as np
import scipy
import scipy.sparse as sp

def get_model():
    """Gets the TrainableJacobiGNN model"""
    return MetaLayer(None, VertexUpdate(edge_to_vertex_aggregation), None)

def edge_to_vertex_aggregation(edge_index, edge_attr, num_nodes):
    """
    Aggregate edge information for use in the vertex update.

    In this case, we return the min, mean, sum, and max of all edge_attr with the same
    sending vertex.
    """
    # edge_index: [2, # edges] with max entry (# nodes - 1)
    # edge_attr: [# edges, # edge attrib]
    # num_nodes: total number of nodes - needed for allocating memory
    # output should be [# nodes, # aggregated attrib]

    agg_min  = torch_scatter.scatter(edge_attr, edge_index[0], dim=0, dim_size=num_nodes, reduce="min")
    agg_mean = torch_scatter.scatter(edge_attr, edge_index[0], dim=0, dim_size=num_nodes, reduce="mean")
    agg_sum  = torch_scatter.scatter(edge_attr, edge_index[0], dim=0, dim_size=num_nodes, reduce="sum")
    agg_max  = torch_scatter.scatter(edge_attr, edge_index[0], dim=0, dim_size=num_nodes, reduce="max")

    return torch.cat((agg_min, agg_mean, agg_sum, agg_max), 1)

class VertexUpdate(torch.nn.Module):
    """
    The vertex update - includes the edge aggregation logic

    The edge aggregation function is passed in an argument
    """

    # Could add arg for network or network structure
    def __init__(self, edge_agg):
        super().__init__()
        self.edge_agg = edge_agg
        self.vertex_update = Sequential(Linear(5,50),
                                        ReLU(),
                                        Linear(50,20),
                                        ReLU(),
                                        Linear(20,1))
        self.vertex_update.apply(init_weights)

    def forward(self, vertex_attr, edge_index, edge_attr, g, batch):
        # vertex_attr   : [#vertices, #vertexAttributes]. vertex_attr[k,:] are attributes at vertex k
        # edge_index   : [2, #edges (i.e., #matrixNnzs)] with entries between 0 and n_vertices-1
        # edge_attr     : [#edges, #edgeAttributes]. edge_attr[k,:] are attributes defined at edge k
        # g             : [#graphs, #globalAttributes] - not used in this case
        # batch         : [#vertices] with max entry (#graphs - 1)

        # Aggregate edge info
        # edge_agg should output [# nodes, # aggregated attrib]
        agg_edges = self.edge_agg(edge_index, edge_attr, vertex_attr.size(0))

        # Aggregate node info, aggr edge info, and global info together
        update_input = torch.cat([vertex_attr, agg_edges], dim=1)

        # Update node features
        return self.vertex_update(update_input)

def init_weights(m):
    """Initialize weights to uniform random values in [0,1) and all biases to 0.01"""
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.uniform_(m.weight, a=0.0, b=1.0)
        m.bias.data.fill_(0.01)

if __name__ == '__main__':
    # Test out the matrix conversion code:
    A = torch.tensor([[ 10.,- 1.,  2.,  0.],
                      [- 1., 11.,- 1.,  3.],
                      [  2.,- 1., 10., -1.],
                      [  0.,  3.,- 1.,  8.]])

    def TorchDenseMatrixToGraph(A):
        edge_index = []
        edge_attr = []
        vertex_attr = [0 for _ in range(A.shape[0])]
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                if i == j:
                    vertex_attr[i] = A[i,i]
                elif i != j:
                    edge_index.append([i,j])
                    edge_attr.append(A[i,j].item())

        edge_index = torch.t(torch.tensor(edge_index))
        edge_attr = torch.reshape(torch.tensor(edge_attr), (-1,1))
        vertex_attr = torch.reshape(torch.tensor(vertex_attr), (-1,1))
        return vertex_attr, edge_index, edge_attr

    vertex_attr, edge_index, edge_attr = TorchDenseMatrixToGraph(A)

    # Print out the gnn inputs
    print('vertex_attr')
    print(vertex_attr)
    print('edge_index')
    print(edge_index)
    print('edge_attr')
    print(edge_attr)

    # Need a batch indication - only have 1 graph so all zeros
    batch = torch.zeros(vertex_attr.size(0))

    # Build the graphnet
    gnn = get_model()

    # Run the graphnet to update node and edge attr
    vertex_attr, edge_attr, _ = gnn(vertex_attr, edge_index, edge_attr, batch=batch)

    # Print the updated features
    print('New values')
    print('vertex_attr')
    print(vertex_attr)
    print('edge_index')
    print(edge_index)
    print('edge_attr')
    print(edge_attr)

