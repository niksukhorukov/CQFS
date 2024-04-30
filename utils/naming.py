def get_experiment_id(alpha_inv, rank_inv, deg_inv, p=None):
    expID = f'a{alpha_inv}r{rank_inv}d{deg_inv}'

    if p is not None:
        expID = f'{expID}p{p:03d}'

    expID = expID.replace('.', '')
    return expID
