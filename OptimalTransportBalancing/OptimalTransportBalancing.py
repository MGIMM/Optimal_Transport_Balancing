import torch

class OptimalTransportBalancing():
    def __init__(self, metric = None):
        """
        Attributes
        ----------
        metric: function (x,y)
            Compute distance between two elements x and y, that are both assumed to be tensors.
        """
        if metric:
            self.metric = metric
        else:
            self.metric = lambda x,y : torch.linalg.norm(x-y)
    def get_nearest_neighbor_index (self, x,Y):
        n = Y.shape[0]
        distance_list = torch.zeros(n)
        for i in range(n):
            distance_list[i] = self.metric(x, Y[i])
        return torch.argmin(distance_list).item() 
    def get_weights(self, source, target, source_weights = None, target_weights = None):
        n = len(source)
        m = len(target)
        if source_weights:
            w = source_weights/n
        else:
            w = torch.ones(n)/n
        if target_weights:
            w_ring = target_weights/m
        else:
            w_ring = torch.ones(m)/m
        sum_nn_w_ring = torch.zeros(n)
        nearest_neighbor_index = torch.zeros(m, dtype = int)
        for j in range(m):
            nearest_neighbor_index[j] = self.get_nearest_neighbor_index(target[j], source)
        for i in range(n):
            for j in range(m):
                if int(nearest_neighbor_index[j].item()) == i:
                    sum_nn_w_ring[i] += w_ring[j]
        eta_ring = w * sum_nn_w_ring
        return eta_ring * n
