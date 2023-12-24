import torch
import torch_scatter


def shift_voxel_grids(voxel_grids, shift_tensor, grid_size, cur_device):
    """
    Shift voxel grids by shift tensor
    :param voxel_grids: [N, 3], where N represents the number of sparse voxels
    :param shift_tensor: [27, 3]
    :param grid_size: [3]
    :param cur_device: current device
    :return: shifted_voxel_grids: [27, N, 3]
    """
    shifted_voxel_tensors = []
    bs_grid_size = torch.tensor([1] + grid_size, device=cur_device)
    for shift in shift_tensor:
        shifted_voxel_grids = (voxel_grids + torch.tensor(shift, device=cur_device)) % bs_grid_size
        shifted_voxel_tensors.append(shifted_voxel_grids)
    return shifted_voxel_tensors



def return_tensor_index(value, t):
    """
    Prerequisite: each value in tensor t is unique. That is, each value has at most 1 time appearance in t.
    Check if value is in tensor t. If True, return the index of value in tensor t; else, return -1
    :param value: Tensor(M,) a one-dimension vector of M elements
    :param t: Tensor(N,M) a two-dimension tensor of N vectors, each vector has M elements
    :return: scalar, the index of value in t at the first dimension
    """
    
    value = value.unsqueeze(0) # (1, M)
    condition = torch.all(t == value, dim=1) 
    indices = torch.nonzero(condition)
    if len(indices) == 0:
        return -1
    return indices.squeeze()

def return_tensor_index_v2(value, t):
    """
    Prerequisite: each value in tensor t is unique. That is, each value has at most 1 time appearance in t.
    Check if value is in tensor t. If True, return the index of value in tensor t; else, return -1
    :param value: Tensor(Q,M) a Q-dimension vector of M elements
    :param t: Tensor(N,M) a two-dimension tensor of N vectors, each vector has M elements
    :return: Tensor(Q,) a one-dimension vector of Q elements, each element is the  index of value in t 
    at the first dimension
    """
    Q = value.shape[0]
    N = t.shape[0]
    t_exp = t.unsqueeze(0).expand(Q, t.shape[0], t.shape[1]) # (Q, N, M)
    value = value.unsqueeze(1) # (Q, 1, M)
    res = torch.all(t_exp == value, dim=2) # (Q, N)
    ones = torch.ones((Q, 1), dtype=torch.bool, device=value.device)
    res_with_ones = torch.concat((res, ones), dim=1) # (Q, N+1)
    res_index = torch.nonzero(res_with_ones) # (Q, 2)
    unq_res_index, unq_inv_res_index =torch.unique(res_index[:,0], return_inverse=True, dim=0)
    res_index = torch.masked_fill(res_index, res_index == N, -1) # (Q, 2)
    select_ind, _ = torch_scatter.scatter_max(res_index[:, 1], unq_inv_res_index, dim=0) 
    
    return select_ind
    
    
    
