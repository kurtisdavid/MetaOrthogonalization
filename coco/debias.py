import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as f
def debias_loss(embeddings, v_g, t=0, unreduced = False, norm='l2', mean=False):
    embeddings = f.normalize(embeddings,p=2, dim=1)
    if unreduced:
        return (embeddings @ (v_g / v_g.norm()) - t)
    else:
        if norm == 'l2':
            l = ( (embeddings @ (v_g / v_g.norm()) - t) ** 2 ).sum()
        elif norm == 'l1':
            l = ( (embeddings @ (v_g / v_g.norm()) - t).abs()  ).sum()
        if mean:
            return l / (embeddings.shape[0])
        else:
            return l
    # return (embeddings @ (v_g / v_g.norm()) - t) ** 2


# given embeddings, subtract out a component of v_g s.t.
#    v^*_embedding \cdot v_g = t
def partial_orthogonalization(embeddings, v_g, t=1e-6):

    m = torch.norm(embeddings, dim=1) # provides norms for each embedding
    n = v_g.norm() # norm of bias
    alpha = embeddings @ (v_g / (v_g.norm() ** 2))
    dot = embeddings @ v_g
    z = 1 - (t / dot)
    z = alpha * z * (np.abs(t) ** np.abs(t)) # correction term to get somewhat close to desired t
    z = z.reshape(-1,1) 
    correction = z @ (v_g.reshape(1,-1)) # want outer product so that each row is z_1 * v_g
    new_embeddings = embeddings - correction
    # now we want to rescale new embeddings to hopefully have good initialization still
    m_star = torch.norm(new_embeddings, dim=1)
    scaling = m / m_star
    return scaling.unsqueeze(1) * new_embeddings

'''
    "imitates" the optim.step() function
        - creates computational graph rather than inplace operations
        - makes sure not to actually update the optimizer
        - should work in conjunction with LRSchedulers
'''
def imitate_SGD_momentum(n2v_optim):
    # we know net2vec is first param group + params
    group = n2v_optim.param_groups[0]
    weight_decay = group['weight_decay']
    momentum = group['momentum']
    dampening = group['dampening']
    nesterov = group['nesterov']
    p = group['params'][0]
    # make sure we've already called backwards on the actual loss
    assert p.grad is not None

    # start building the graph
    d_p = p.grad
    ans = d_p
    if weight_decay != 0:
        ans = ans + weight_decay * p
    if momentum != 0:
        param_state = n2v_optim.state[p]
        if 'momentum_buffer' not in param_state:
            # not needed, but just kept for parallel with SGD
            buf = torch.clone(d_p).detach()
            ans = d_p
        else:
            # extract buffer that is supposed to be a constant in graph
            #   don't want multiple steps affecting...
            buf = param_state['momentum_buffer']
            # momentum update
            tmp = buf * (momentum) + (1-dampening) * d_p
            ans = tmp
        if nesterov:
            # TODO, need to check
            raise Exception("Nesterov not implemented")
            ans = ans + momentum * ans
    return group['lr']*ans
