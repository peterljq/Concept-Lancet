import spams
import torch
import numpy as np
import algorithms  


def dim_reduction_embed_old(X, method = 'all', param = None):
    # X : array-like, shape (n_dim, n_samples)
    n_dim, n_samples = X.shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gram_matrix = X.T @ X
    assert isinstance(X, torch.Tensor), 'current implementation accepts only torch tensor for X'

    L, Q = torch.linalg.eigh(gram_matrix)
    L = torch.max(L, torch.zeros_like(L))
    assert torch.all(L[:-1] <= L[1:]), 'L is supposed to be in ascending order'


    if method == 'all':
        return torch.diag_embed(torch.sqrt(L)) @ Q.T
    elif method ==  'top_d':
        assert isinstance(param, int), 'when method is top, param must be an integer meaning the dimension'
        return torch.diag_embed(torch.sqrt(L[-param:])) @ Q.T[-param:]
    elif method ==  'top_energy':
        assert isinstance(param, float), 'when method is top_energy, param must be a float'
        L_flipped = torch.flip(L, [0])
        L_cumsum = torch.cumsum(L_flipped, 0)
        L_cumsum = L_cumsum / L_cumsum[-1]
        n_dim = torch.sum(L_cumsum < param)
        return torch.diag_embed(torch.sqrt(L[-n_dim:])) @ Q.T[-n_dim:]


def dim_reduction_embed(X, method = 'all', param = None):
    # X : array-like, shape (n_dim, n_samples)
    n_dim, n_samples = X.shape
    print(f"n_dim: {n_dim}, n_samples: {n_samples}")

    U, S, V = torch.svd_lowrank(X.float(), q=min(n_dim, n_samples))

    if method == 'all':
        return (V * S).T 
    elif method == 'top_d':
        assert isinstance(param, int), 'when method is top_d, param must be an integer meaning the dimension'
        return (V[:, :param] * S[:param]).T
    elif method == 'top_energy':
        assert isinstance(param, float), 'when method is top_energy, param must be a float'
        S_cumsum = torch.cumsum(S**2, 0)
        S_total = S_cumsum[-1]
        energy_threshold = S_total * param
        n_dim_reduced = torch.sum(S_cumsum < energy_threshold) + 1  # +1 to include the dimension meeting the threshold
        return (V[:, :n_dim_reduced] * S[:n_dim_reduced]).T
    else:
        raise ValueError("Method not recognized. Use 'all', 'top_d', or 'top_energy'.")


def decompose(target, dl_dict, tau = 0.95, alpha = 0.05, print_level = 3, normalize = True, return_rectr_err = False):

    if print_level >= 2:
        print('shape of target is {}'.format(target.shape))
        print('length of dl_dict is {}'.format(len(dl_dict)))

    data = []
    data.append(target.view(-1))
    for atom in dl_dict:
        atom = atom.view(-1)
        data.append(atom)

    data = torch.stack(data, dim=0)
    print('shape of data is {}'.format(data.shape))

    if print_level >= 2:
        print(f"Successfully loaded {data.shape[0]} emotions, each in {data.shape[1]} dimension.")
        print("data dimension: ", data.shape)
        print("norm of each row of data: ", torch.norm(data, dim=1))

    data_new = dim_reduction_embed(data.T.float()).T
    data_new = data_new.detach().cpu().numpy()

    if not np.all(np.isfinite(data_new)):
        print('data_new contains nan or inf')
        import pdb; pdb.set_trace()

    id_target = 0
    y = data_new[id_target]
    y = np.copy(y.reshape(1, -1))

    if normalize:
        norm_y = np.linalg.norm(y) / 1
        y = y / norm_y
        norm_data_new = np.linalg.norm(data_new, axis=1) / 1
        data_new = data_new / norm_data_new[:, None]

    data_new[id_target] = 0    

    c_spams = spams.lasso(np.asfortranarray(y.T), D=np.asfortranarray(data_new.T), 
                                        lambda1=tau * alpha, lambda2=(1.0-tau) * alpha, mode=2)
    c_spams = np.asarray(c_spams.todense()).T[0]

    if print_level >= 2:
        print(c_spams)

    c_spams = c_spams.reshape(1, -1)

    def obj_elastic_net(X, y, c, alpha, tau):
        reconstruction_error = np.linalg.norm(y.T - X.T @ c.T) 
        l1_reg = np.sum(np.abs(c))
        l2_reg = np.sum(c ** 2)
        total_error = 3 * 0.5 * (reconstruction_error ** 2) + alpha * (tau * l1_reg + 0.5 * (1.0 - tau) * l2_reg)
        return reconstruction_error, l1_reg, l2_reg, total_error


    if print_level >= 1:
        reconstruction_error, l1_reg, l2_reg, total_error = obj_elastic_net(data_new, y, c_spams, alpha, tau)
        print('===elastic net===')
        print(f'recstr = {reconstruction_error:.4f}, l1_reg = {l1_reg:.4f}, l2_reg = {l2_reg:.4f}, total = {total_error:.4f}')

        c_lstsq = np.linalg.lstsq(data_new.T, y.T, rcond=None)[0].T

        reconstruction_error, l1_reg, l2_reg, total_error = obj_elastic_net(data_new, y, c_lstsq, alpha, tau)
        print('===least squares===')
        print(f'recstr = {reconstruction_error:.4f}, l1_reg = {l1_reg:.4f}, l2_reg = {l2_reg:.4f}, total = {total_error:.4f}')
    
    c_spams = c_spams[0]

    if normalize:
        c_spams = c_spams / norm_data_new * norm_y

    control_coefficient = torch.tensor(c_spams[1:])
    
    if print_level >= 1:
        print('l1 norm of control_coefficient: ', torch.sum(torch.abs(control_coefficient)))

    if return_rectr_err:
        return control_coefficient, reconstruction_error
    
    return control_coefficient


def ip_omp(target, dictionary, num_iterations, device='cuda'):


    with torch.no_grad():

        log = algorithms.ip(
            Phi=dictionary,
            y=target.unsqueeze(2),
            num_iterations=num_iterations,
            device=device,
            compute_xhat=False,
        )

        coeffs = algorithms.estimate_x_lsq(
            dictionary,
            target.unsqueeze(2),
            log["columns"],
            device=device,
        )  

    return coeffs.squeeze()