import torch

def proposal_pairs_fb(grouped_indices, instance_mask):
    """

    :param grouped_indices: grouped_seed_indices, [16, 256, 16]
    :param instance_mask: pts instance mask, list[16, tensor(20000)]
    :return:
    """
    bs = grouped_indices.shape[0]
    group_num = grouped_indices.shape[1]
    proposal_instance_mask = []
    for b in range(bs):
        batch_mask = instance_mask[b]
        batch_proposal = grouped_indices[b]
        batch_instance_mask = []
        for g in range(group_num):
            proposal_indices = batch_proposal[g]
            pts_mask = batch_mask[proposal_indices]
            # unique_proposal_indice = torch.unique(proposal_indices)
            batch_instance_mask.append(pts_mask.unsqueeze(0))
        batch_instance_mask = torch.cat(batch_instance_mask, dim=0)
        proposal_instance_mask.append(batch_instance_mask.unsqueeze(0))
    proposal_instance_mask = torch.cat(proposal_instance_mask, dim=0)

    return proposal_instance_mask
