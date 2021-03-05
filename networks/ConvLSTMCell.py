import torch.nn as nn
import torch
import numpy as np
from .koopcell import IresnetBlock
from .lticell import LTICell

def compute_operator(prev_block, block):
    U, S, V = torch.svd(prev_block)
    # print(S.size()) [b, 52]
    # print(U.size()) [b, 52, 52]
    # print(V.size()) [b, 512, 52]
    eigen_matrix = torch.stack([torch.diag(1. / S[i]) for i in range(S.size(0))], 0)
    A = torch.matmul(torch.matmul(U.transpose(1, 2), block), torch.matmul(V, eigen_matrix))
    Ws = []
    for i in range(block.size(0)):
        _, W = torch.eig(A[i], eigenvectors=True)
        Ws.append(W)
    W = torch.stack(Ws, 0)
    projection = torch.matmul(block, V), torch.matmul(eigen_matrix, W)
    return projection

class ConvLSTMCell(nn.Module):

    def __init__(self,
                 latent_state_dim,
                 latent_obs_dim,
                 org_dim,
                 number_of_basis,
                 bandwidth,
                 device,
                 seq_len, z,
                 T = 3, h = 4, w = 13,
                 initial_trans_covar=0.1):
        super(ConvLSTMCell, self).__init__()

        self._lsd = latent_state_dim
        self._lod = latent_obs_dim
        self._seq = seq_len
        self.T = T

        self.iresnet = IresnetBlock((512, h, w), h * w)
        self.h = h
        self.w = w
        lti_cells = []
        for i in range(seq_len):
            num_states = i if i < T else T
            lti_cells.append(LTICell(latent_state_dim, latent_obs_dim, number_of_basis,
                                            bandwidth, device, num_states, initial_trans_covar, z))
        self.lti_cells = nn.ModuleList(lti_cells)


    def forward(self, inputs, state):
        frames = inputs.size(1) / self._seq
        projections = []
        for i in range(int(frames)-1):
            # include o_t-T: t and o_t-T+1: t+1
            prev_block = inputs[:, i * self._seq: (i+1) * self._seq]
            block = inputs[:, (i+1) * self._seq: (i+2) * self._seq]
            # proj = compute_operator(prev_block, block)
            # projections.append(proj)

        concat_output = None
        for i in range(int(frames)):
            # if i == 0:
            #     proj = projections[0]
            # else:
            #     proj = projections[i-1]
            input = inputs[:, i * self._seq: (i+1) * self._seq]
            b, _, c = input.size()
            # k input low dim, input high dim.
            input = input.view(b, c, self.h, self.w)
            k_input, _ = self.iresnet(input) # [b, c_low, h, w]
            state = self.init_hidden(state.shape[0])
            concat_state = None
            concat_koutput = []
            for ii in range(self._seq):
                observation = k_input[:, :ii] if ii < self.T else k_input[:, ii-self.T: ii]
                # def forward(self, observations, states, prev_error=None, prev_transition=None):
                # state: x_t, x_(t+1) observation: o_(t+1)
                # predict step: x_(t+1)^- = Ax_t (A is transition matrix)
                # correct step x_(t+1)^+ = F(x_(t+1)^-, o_(t+1))
                state = self.lti_cells[ii](observation, state) # state dimension will not change
                if len(state.shape) == 2:
                    state = state.unsqueeze(1)
                # print(ii, state.size())
                concat_koutput.append(state[:, -1].unsqueeze(-1))

                # stack the state with previous states and feed to the next lti cell
                if concat_state is None:
                    concat_state = state
                elif ii < self.T:
                    concat_state = torch.cat([concat_state, state[:, -1].unsqueeze(1)], dim=1)
                    state = concat_state
            # fixed: print(concat_koutput.size()) [b, 52, 104] dimension is 104, 52 is the sequence length.
            k_output = torch.cat(concat_koutput, -1).view(b, -1, self.h, self.w)
            output = self.iresnet.inverse(k_output, input)
            if concat_output is None:
                concat_output = output.unsqueeze(1)
            else:
                concat_output = torch.cat([concat_output, output.unsqueeze(1)], 1)

        return concat_output

    def init_hidden(self, batch_size):
        device = next(self.parameters()).device
        initial_mean = torch.zeros([batch_size, 1, self._lod], dtype=torch.float, device=device)
        # initial_covar_diag = 10 * torch.ones([batch_size, 2 * self._lod], dtype=torch.float, device=device)
        # initial_covar_side = torch.ones([batch_size, 1 * self._lod], dtype=torch.float, device=device)
        return initial_mean