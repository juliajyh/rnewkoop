import torch.nn as nn
import torch
import numpy as np

def dadat(A, diag_mat):
    diag_ext = diag_mat.unsqueeze(1)
    first_prod = A * A * diag_ext
    return torch.sum(first_prod, 2)


def dadbt(A, diag_mat, B):
    diag_ext = diag_mat.unsqueeze(1)
    first_prod = A * B * diag_ext
    return torch.sum(first_prod, 2)


# Pack and Unpack functions

def pack_state(mean, covar):
    return torch.cat([mean] + covar, 1)


def unpack_state(state):
    lod = int(state.size(1) / 5)
    mean = state[:, :2 * lod]
    covar_upper = state[:, 2 * lod: 3 * lod]
    covar_lower = state[:, 3 * lod: 4 * lod]
    covar_side = state[:, 4*lod:]
    return mean, [covar_upper, covar_lower, covar_side]


def pack_input(obs_mean, obs_covar):
    # if not obs_valid.dtype == tf.float32:
        # obs_valid = tf.cast(obs_valid, tf.float32)
    return torch.cat((obs_mean, obs_covar), 1)


def unpack_input(input_as_vector):
    lod = int(input_as_vector.size(1) / 2)
    obs_mean = input_as_vector[:, :lod]
    obs_covar = input_as_vector[:, lod:]
    return obs_mean, obs_covar


class RKNSingleCell(nn.Module):
    def __init__(self,
                 latent_state_dim,
                 latent_obs_dim,
                 org_dim,
                 number_of_basis,
                 bandwidth,
                 device,
                 initial_trans_covar):
        super(SingleCell, self).__init__()
        assert latent_state_dim == 2 * latent_obs_dim, "Currently only 2 * m = n supported"

        self._lsd = latent_state_dim
        self._lod = latent_obs_dim
        self._num_basis = number_of_basis
        self._initial_trans_covar = initial_trans_covar
        self._bandwidth = bandwidth
        self.device = device

        self.dense_mean = nn.Linear(org_dim, latent_obs_dim)  # Remember to set bias
        self.dense_covar = nn.Linear(org_dim, latent_obs_dim)
        self.mean_norm = nn.Linear(latent_obs_dim, latent_obs_dim)
        self.activate_mean = lambda x: nn.ELU()(x) + 1
        self.activate_covar = lambda x: x / x.norm(1, keepdim=True)

        tm_11_init = np.tile(np.expand_dims(np.eye(self._lod, dtype=np.float32), 0), [self._num_basis, 1, 1])
        tm_12_init = 0.2 * np.tile(np.expand_dims(np.eye(self._lod, dtype=np.float32), 0), [self._num_basis, 1, 1])
        tm_21_init = -0.2 * np.tile(np.expand_dims(np.eye(self._lod, dtype=np.float32), 0), [self._num_basis, 1, 1])
        tm_22_init = np.tile(np.expand_dims(np.eye(self._lod, dtype=np.float32), 0), [self._num_basis, 1, 1])

        tm_11_full = nn.Parameter(torch.from_numpy(tm_11_init), requires_grad=True)
        tm_12_full = nn.Parameter(torch.from_numpy(tm_12_init), requires_grad=True)
        tm_21_full = nn.Parameter(torch.from_numpy(tm_21_init), requires_grad=True)
        tm_22_full = nn.Parameter(torch.from_numpy(tm_22_init), requires_grad=True)
        self.tm_full = [tm_11_full, tm_12_full, tm_21_full, tm_22_full]

        self._coefficient_net = nn.Sequential(nn.Linear(self._lsd, self._num_basis), nn.Softmax())
        # weights = self._coefficient_net.weights()
        elup1_inv = lambda x: (np.log(x) if x < 1.0 else (x - 1.0))
        self.log_transition_covar = nn.Parameter(
            torch.from_numpy(np.array([elup1_inv(self._initial_trans_covar)] * self._lsd)), requires_grad=True)

    def forward(self, input, state):
        obs_mean = self.activate_mean(self.mean_norm(self.dense_mean(input)))
        obs_covar = self.activate_covar(self.dense_covar(input))
        state_mean, state_covar = unpack_state(state)
        pred_res = self._predict(state_mean, state_covar)
        prior_mean, prior_covar = pred_res
        update_res = self._update(prior_mean, prior_covar, obs_mean, obs_covar)
        state_mean, state_covar = update_res[:2]
        return state_mean, pack_state(state_mean, state_covar)

    def _predict(self, post_mean, post_covar):
        # compute state dependent transition matrix
        coefficients = self._coefficient_net(post_mean) # nn.Linear 改一下， 保存gradient
        tm_full = [tm.to(self.device) for tm in self.tm_full]
        tm_11, tm_12, tm_21, tm_22 = (torch.triu(x, diagonal=-self._bandwidth) - torch.triu(x, diagonal=self._bandwidth) for x in
                                      tm_full)
        basis_matrices = torch.cat([torch.cat([tm_11, tm_12], -1),
                                          torch.cat([tm_21, tm_22], -1)], -2)
        basis_matrices = basis_matrices.unsqueeze(0)
        scaled_matrices = coefficients.view(-1, self._num_basis, 1, 1) * basis_matrices
        transition_matrix = torch.sum(scaled_matrices, 1)

        # predict next prior mean
        expanded_state_mean = post_mean.unsqueeze(-1)
        new_mean = torch.matmul(transition_matrix, expanded_state_mean).squeeze(-1)

        # predict next prior covariance (eq 10 - 12 in paper supplement)
        b11 = transition_matrix[:, :self._lod, :self._lod]
        b12 = transition_matrix[:, :self._lod, self._lod:]
        b21 = transition_matrix[:, self._lod:, :self._lod]
        b22 = transition_matrix[:, self._lod:, self._lod:]

        covar_upper, covar_lower, covar_side = post_covar

        trans_covar = nn.ELU()(self.log_transition_covar) + 1.
        trans_covar = trans_covar
        trans_covar_upper = trans_covar[:self._lod]
        trans_covar_lower = trans_covar[self._lod:]

        new_covar_upper = dadat(b11, covar_upper) + 2 * dadbt(b11, covar_side, b12) + dadat(b12, covar_lower) \
                          + trans_covar_upper
        new_covar_lower = dadat(b21, covar_upper) + 2 * dadbt(b21, covar_side, b22) + dadat(b22, covar_lower) \
                          + trans_covar_lower
        new_covar_side = dadbt(b21, covar_upper, b11) + dadbt(b22, covar_side, b11) \
                         + dadbt(b21, covar_side, b12) + dadbt(b22, covar_lower, b12)
        return new_mean, [new_covar_upper, new_covar_lower, new_covar_side]


    def _update(self, prior_mean, prior_covar, obs_mean, obs_covar):
        covar_upper, covar_lower, covar_side = prior_covar

        # compute kalman gain (eq 2 and 3 in paper)
        denominator = covar_upper + obs_covar
        q_upper = covar_upper / denominator
        q_lower = covar_side / denominator

        # update mean (eq 4 in paper)
        residual = obs_mean - prior_mean[:, :self._lod]
        new_mean = prior_mean + torch.cat((torch.mul(q_upper, residual), torch.mul(q_lower, residual)), -1)
        # update covariance (eq 5 -7 in paper)
        covar_factor = 1 - q_upper
        new_covar_upper = covar_factor * covar_upper
        new_covar_lower = covar_lower - q_lower * covar_side
        new_covar_side = covar_factor * covar_side
        return new_mean.float(), [new_covar_upper.float(), new_covar_lower.float(), new_covar_side.float()]