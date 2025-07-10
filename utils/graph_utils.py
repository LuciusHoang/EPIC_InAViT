import torch


def build_temporal_graph(pose_tensor, edges=None):
    """
    Constructs a temporal graph from a pose sequence tensor.

    Args:
        pose_tensor (Tensor): Shape [T, J, D], where
            T = time steps,
            J = number of joints,
            D = joint feature dimension (usually 3 for xyz)
        edges (list of tuple): Optional list of intra-frame joint connections.

    Returns:
        graph_tensor (Tensor): Shape [T, J, J], where each [J, J] is an adjacency matrix.
    """
    T, J, D = pose_tensor.shape
    graph_tensor = torch.zeros((T, J, J), dtype=torch.float32)

    if edges is None:
        # Fully connected graph (intra-frame)
        for i in range(J):
            for j in range(J):
                if i != j:
                    graph_tensor[:, i, j] = 1.0
    else:
        for (i, j) in edges:
            graph_tensor[:, i, j] = 1.0
            graph_tensor[:, j, i] = 1.0  # symmetric

    # Normalize adjacency matrices if needed (optional for some GNNs)
    return graph_tensor
