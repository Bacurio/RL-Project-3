import torch

def lambda_return(rewards, values, masks, gamma=0.99, lam=0.95):
    """
    Compute lambda-returns (G^λ) for a batch of trajectories.
    
    Args:
        rewards (Tensor): shape [T, B] – rewards at each timestep.
        values (Tensor): shape [T + 1, B] – estimated state values (from Q-network or critic).
        masks (Tensor): shape [T, B] – 0 if done at timestep, 1 otherwise.
        gamma (float): discount factor.
        lam (float): lambda parameter ∈ [0, 1].
        
    Returns:
        lambda_returns (Tensor): shape [T, B] – computed λ-returns.
    """
    T = rewards.size(0)   # number of timesteps
    lambda_returns = torch.zeros_like(rewards) # shape [T, B]
    next_value = values[-1]  # shape [B]

    for t in reversed(range(T)): 
        td_error = rewards[t] + gamma * values[t + 1] * masks[t] - values[t]  
        next_value = td_error * lam + values[t]  # G^λ_t = δ_t + γλ G^λ_{t+1}
        lambda_returns[t] = next_value

    return lambda_returns


