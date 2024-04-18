import torch


def forward_looking_detailed_balance_loss(P_F, P_B, F, R, traj_lengths, transition_rs):
    cumul_lens = torch.cumsum(torch.cat([torch.zeros(1, device=traj_lengths.device), traj_lengths]), 0).long()

    total_loss = torch.zeros(1, device=traj_lengths.device)
    for ep in range(traj_lengths.shape[0]):
        offset = cumul_lens[ep]
        T = int(traj_lengths[ep])
        for i in range(T):
            flag = float(i + 1 < T)

            curr_PF = P_F[offset + i]
            curr_PB = P_B[offset + i]
            curr_F = F[offset + i]
            curr_F_next = F[offset + min(i + 1, T - 1)]
            curr_r = transition_rs[offset + i]
            acc = curr_F + curr_PF - curr_F_next - curr_PB - curr_r

            total_loss += acc.pow(2)

    return total_loss