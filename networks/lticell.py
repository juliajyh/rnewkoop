import torch.nn as nn
import torch
import numpy as np

class LTICell(nn.Module):
    def __init__(self, latent_state_dim, latent_obs_dim,
                 number_of_basis, bandwidth, device, num_states, initial_trans_covar, z_factor):
        # z factor should be finetuned.

        super(LTICell, self).__init__()
        self._lsd = latent_state_dim
        self._lod = latent_obs_dim
        self._num_basis = number_of_basis
        self._initial_trans_covar = initial_trans_covar
        self._bandwidth = bandwidth
        self.device = device
        self._num_states = num_states
        self._z = z_factor

        self.activate = lambda x: nn.ELU()(x) + 1
        # self.activate_covar = lambda x: x / x.norm(1, keepdim=True)

        tm_11_init = np.tile(np.expand_dims(np.eye(self._lod * max(1, self._num_states), dtype=np.float32), 0), [self._num_basis, 1, 1])
        tm_12_init = 0.2 * np.tile(np.expand_dims(np.eye(self._lod * max(1, self._num_states), dtype=np.float32), 0), [self._num_basis, 1, 1])
        tm_21_init = -0.2 * np.tile(np.expand_dims(np.eye(self._lod * max(1, self._num_states), dtype=np.float32), 0), [self._num_basis, 1, 1])
        tm_22_init = np.tile(np.expand_dims(np.eye(self._lod * max(1, self._num_states), dtype=np.float32), 0), [self._num_basis, 1, 1])

        tm_11_full = nn.Parameter(torch.from_numpy(tm_11_init), requires_grad=True)
        tm_12_full = nn.Parameter(torch.from_numpy(tm_12_init), requires_grad=True)
        tm_21_full = nn.Parameter(torch.from_numpy(tm_21_init), requires_grad=True)
        tm_22_full = nn.Parameter(torch.from_numpy(tm_22_init), requires_grad=True)
        self.tm_full = [tm_11_full, tm_12_full, tm_21_full, tm_22_full]

        if num_states > 0:
            tm = np.tile(np.expand_dims(np.eye(self._lsd * num_states, dtype=np.float32), 0), [self._num_basis * num_states, 1, 1])
            self.tm = nn.Parameter(torch.from_numpy(tm), requires_grad=True)
            identity = np.eye(self._lsd * num_states, dtype=np.float32)
            self.identity = nn.Parameter(torch.from_numpy(identity), requires_grad=False)
            self._disturbance_net = nn.Sequential(nn.Linear(self._lsd * num_states, self._num_basis * num_states),
                                                  nn.Softmax())

        base_identity = np.eye(self._lsd * max(1, self._num_states), dtype=np.float32)
        self._base_identity = nn.Parameter(torch.from_numpy(base_identity), requires_grad=False)

        self._coefficient_net = nn.Sequential(nn.Linear(self._lsd, self._num_basis), nn.Softmax())
        elup1_inv = lambda x: (np.log(x) if x < 1.0 else (x - 1.0))
        self.log_transition_covar = nn.Parameter(
            torch.from_numpy(np.array([elup1_inv(self._initial_trans_covar)] * self._lsd)), requires_grad=True)


    def forward(self, observations, states, prev_transition=None):
        # parameter: observations(o_t-T,,,,, o_t)
        # parameter: states(x_t-T,,,,, x_t)
        # predict step: A_t-T......A_t are the transition matrix.
        # update step: takes (x, o, A) get the correct state with the observation.
        # update step returns x_t-T+1..........x_t+1 replace inplace.

        transition_matrix = None
        for i in range(self._num_states):
            state = states[:, i]
            coefficients = self._coefficient_net(self.activate(state))

            tm_full = [tm.to(self.device) for tm in self.tm_full]
            tm_11, tm_12, tm_21, tm_22 = (torch.triu(x, diagonal=-self._bandwidth) - torch.triu(x, diagonal=self._bandwidth)
                                          for x in
                                          tm_full)
            basis_matrices = torch.cat([torch.cat([tm_11, tm_12], -1),
                                        torch.cat([tm_21, tm_22], -1)], -2)
            basis_matrices = tm_11 + tm_12 + tm_21 + tm_22
            basis_matrices = basis_matrices.unsqueeze(0)
            scaled_matrices = coefficients.view(-1, self._num_basis, 1, 1) * basis_matrices
            if transition_matrix is None:
                transition_matrix = torch.sum(scaled_matrices, 1)
            else:
                transition_matrix = torch.cat([transition_matrix, torch.sum(scaled_matrices, 1)], 1)

        if self._num_states == 0:
            state = states.squeeze()
            coefficients = self._coefficient_net(self.activate(state))

            tm_full = [tm.to(self.device) for tm in self.tm_full]
            tm_11, tm_12, tm_21, tm_22 = (torch.triu(x, diagonal=-self._bandwidth) - torch.triu(x, diagonal=self._bandwidth)
                            for x in tm_full)
            # basis_matrices = torch.cat([torch.cat([tm_11, tm_12], -1),
            #                             torch.cat([tm_21, tm_22], -1)], -2)
            basis_matrices = tm_11 + tm_12 + tm_21 + tm_22
            basis_matrices = basis_matrices.unsqueeze(0)
            scaled_matrices = coefficients.view(-1, self._num_basis, 1, 1) * basis_matrices
            transition_matrix = torch.sum(scaled_matrices, 1)

            expanded_state_mean = state.unsqueeze(-1)
            prior_state = torch.matmul(transition_matrix, expanded_state_mean).squeeze(-1)
            return prior_state

        post_state = self.update(states, observations, transition_matrix)
        return post_state

    def update(self, states, observations, transitions):
        # x' = R x (o - x)
        # (zI - A) x R = I + delta (z is the z factor) R square matrix.
        # delta and R is an upper triangular matrix
        # R[0, 1] = (zI - A)^-1 x (I + delta)[0,1] iterate all upper index.
        # update step returns x_t-T+1..........x_t+1 replace inplace.
        disturbances = self._disturbance_net(states.flatten(start_dim = 1))
        basis_matrice = torch.triu(self.tm)
        scaled_matrix = disturbances.view(-1, self._num_states * self._num_basis, 1, 1) * basis_matrice
        identity_matrix = torch.sum(scaled_matrix, 1) + self.identity
        output_state = None
        for i in range(self._num_states):
            state_vec = torch.zeros(states.size(0), self._lsd * self._num_states).to(states.device)
            identity = identity_matrix[:, self._lsd * i: self._lsd * (i + 1)]
            for ii in range(i, self._num_states):
                transition = transitions[:, self._num_states - i - 1]
                multiplier = torch.matmul(identity, torch.inverse(self._z * torch.unsqueeze(self._base_identity, dim=0).repeat(transition.shape[0], 1, 1) - transition[0]))
                error = observations[:, self._num_states - i - 1].flatten(start_dim = 1) - states[:, self._num_states - i - 1]
                state_vec += torch.matmul(error.unsqueeze(-1).transpose(-1, -2), multiplier).squeeze() #torch.matmul(multiplier, error)
            if output_state is None:
                output_state = state_vec.unsqueeze(1)
            else:
                output_state = torch.cat([state_vec.unsqueeze(1), output_state], 1)

        return output_state[..., :self._lsd] # definitely not the best solution we have, but should be able to fix through slicing the matrices above -- zitong

    def _predict(self, post_mean):
        # predict next prior covariance (eq 10 - 12 in paper supplement)
        # b11 = transition_matrix[:, :self._lod, :self._lod]
        # b12 = transition_matrix[:, :self._lod, self._lod:]
        # b21 = transition_matrix[:, self._lod:, :self._lod]
        # b22 = transition_matrix[:, self._lod:, self._lod:]

        # covar_upper, covar_lower, covar_side = post_covar
        #
        # trans_covar = nn.ELU()(self.log_transition_covar) + 1.
        # trans_covar = trans_covar
        # trans_covar_upper = trans_covar[:self._lod]
        # trans_covar_lower = trans_covar[self._lod:]
        #
        # new_covar_upper = dadat(b11, covar_upper) + 2 * dadbt(b11, covar_side, b12) + dadat(b12, covar_lower) \
        #                   + trans_covar_upper
        # new_covar_lower = dadat(b21, covar_upper) + 2 * dadbt(b21, covar_side, b22) + dadat(b22, covar_lower) \
        #                   + trans_covar_lower
        # new_covar_side = dadbt(b21, covar_upper, b11) + dadbt(b22, covar_side, b11) \
        #                  + dadbt(b21, covar_side, b12) + dadbt(b22, covar_lower, b12)

        return new_mean

        return None
