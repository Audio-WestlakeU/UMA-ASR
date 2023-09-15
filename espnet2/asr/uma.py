
'''
Author: FnoY fangying@westlake.edu.cn
LastEditTime: 2023-09-15 14:31:51
FilePath: /espnet/espnet2/asr/uma.py
Notes: If the feature dimension changes from 256 to 512, just modify 'output_size: int = 256' to 'output_size: int = 512'.
'''
# """Unimodal aggregation definition."""
import logging
from typing import Optional, Tuple
import torch
from typeguard import check_argument_types


class UMA(torch.nn.Module):
    """UMA module.

    """

    def __init__(
        self,
        input_size: int = 512,
        output_size: int = 256,
    ):
        assert check_argument_types()
        super().__init__()
        self._output_size = output_size
        input_size = output_size

        self.linear_sigmoid = torch.nn.Sequential(
            torch.nn.Linear(input_size, 1),
            torch.nn.Sigmoid(),
        )

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        xs_pad: torch.Tensor,
        olens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Calculate forward propagation.

        Args:
            xs_pad (torch.Tensor): Input tensor (#batch, L, input_size).
            olens (torch.Tensor): Input length (#batch).
            prev_states (torch.Tensor): Not to be used now.
        Returns:
            torch.Tensor: Output tensor (#batch, I, output_size).
            torch.Tensor: Output length (#batch).
            torch.Tensor: Not to be used now.
        """

        batch, length, _ = xs_pad.size()
        # Use Linear-Sigmoid to generate unimodal aggregation weights
        # uma_weights: (#batch, L, 1)
        uma_weights = self.linear_sigmoid(xs_pad)

        # Unimodal Detection
        scalar_before = uma_weights[:,:-1,:].detach() # (#batch, L-1, 1)
        scalar_after = uma_weights[:,1:,:].detach() # (#batch, L-1, 1)
        scalar_before = torch.nn.functional.pad(scalar_before,(0,0,1,0))    # (#batch, L, 1)
        scalar_after = torch.nn.functional.pad(scalar_after,(0,0,0,1))  # (#batch, L, 1)

        mask = (uma_weights.lt(scalar_before)) & (uma_weights.lt(scalar_after)) # bool tensor (#batch, L, 1)
        mask = mask.reshape(uma_weights.shape[0], -1) # bool tensor (#batch, L)
        mask[:,0] = True
        # mask.nonzero() is [[0,0],[0,3],[0,7],...,[1,0],[1,2],...,[2,0],[2,4],...,[#batch-1,0],...]
        # mask.nonzero() : (K,2); K is the total number of valleys in this batch
        batch_index = mask.nonzero()[:,0] # (k,1); [0,0,0,...,1,1,...,2,2,...,#batch-1,...]
        valley_index_start = mask.nonzero()[:,1] # (k,1); [0,3,7,...,0,2,...,0,4,...,0,...]
        mask[:,0] = False
        mask[:,-1] = True
        valley_index_end = mask.nonzero()[:,1] + 2 
        # (k,1); [5,9,...,4,...,6,...]
        valley_index_end = torch.where(valley_index_end > (length) * torch.ones_like(valley_index_end), 
                                       (length) * torch.ones_like(valley_index_end), valley_index_end)

        _,counts = torch.unique(batch_index, return_counts = True) # (#batch, 1); the number of valleys in each sample
        max_counts = (torch.max(counts)).item() 

        utri_mat1 = torch.tril(torch.ones(max_counts+1,max_counts),-1).to(xs_pad.device)
        batch_index_mask = utri_mat1[counts]
        batch_index_mask = batch_index_mask.reshape(-1,1)
        batch_index_mask = batch_index_mask.nonzero()[:, 0]

        valleys = torch.zeros(batch * max_counts, 2).type_as(valley_index_start)
        valleys[batch_index_mask] = torch.cat((valley_index_start.unsqueeze(1), valley_index_end.unsqueeze(1)),1)
        # logging.info(str(valleys))
        
        # utri_mat = torch.tril(torch.cuda.FloatTensor(length+1,length).fill_(1),-1)
        utri_mat = torch.tril(torch.ones(length+1,length),-1).to(xs_pad.device)
        output_mask = (utri_mat[valleys[:,1]]-utri_mat[valleys[:,0]]).reshape(batch, max_counts, length)
        output_mask = output_mask.detach()

        # Aggregation
        alpha_h = torch.mul(uma_weights, xs_pad)
        xs_pad = torch.bmm(output_mask, alpha_h) / torch.bmm(output_mask, uma_weights).clamp_(1e-6)
 
        olens = (olens / olens[0] * xs_pad.shape[1]).type_as(olens)
        # olens = counts
        
        # return xs_pad, olens, uma_weights
        return xs_pad, olens, None
