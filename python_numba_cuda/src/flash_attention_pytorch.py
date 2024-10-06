import torch
import math
import torch.nn.functional as F
eps = 1e-6
print("eps: ", eps)

def flash_attention_2_row_block(Q_i, K, V, BcDim):
    BrDim, dim = Q_i.shape  # Q_i: [BrDim, dim]
    N = K.shape[0]  # K: [N, dim]
    
    # Compute the number of column blocks
    Tc = math.ceil(N / BcDim)
    
    # Initialize on-chip variables
    O_i = torch.zeros((BrDim, dim), device=Q_i.device)  # O_i: [BrDim, dim]
    l_i = torch.zeros(BrDim, device=Q_i.device)  # l_i: [BrDim]
    m_i = torch.full((BrDim,), float('-inf'), device=Q_i.device)  # m_i: [BrDim]
    
    for j in range(Tc):
        # Load K_j and V_j
        start_col = j * BcDim
        end_col = min((j + 1) * BcDim, N)
        K_j = K[start_col:end_col, :]  # K_j: [BcDim, dim]
        V_j = V[start_col:end_col, :]  # V_j: [BcDim, dim]
        
        # Compute S_i_j
        S_i_j = torch.matmul(Q_i, K_j.transpose(-2, -1)) / math.sqrt(dim)  # S_i_j: [BrDim, BcDim]
        
        # Compute statistics
        m_i_new = torch.maximum(m_i, S_i_j.max(dim=-1)[0])  # m_i_new: [BrDim]
        exp_max_delta = torch.clamp(torch.exp(m_i - m_i_new), min=eps)
        P_tilde_i_j = torch.exp(S_i_j - m_i_new.unsqueeze(-1))  # P_tilde_i_j: [BrDim, BcDim]
        l_i_new = exp_max_delta * l_i + P_tilde_i_j.sum(dim=-1)  # l_i_new: [BrDim]
        l_i_new = torch.clamp(l_i_new, min=eps)
        
        # Update O_i
        O_i = exp_max_delta.unsqueeze(-1) * O_i + torch.matmul(P_tilde_i_j, V_j)  # O_i: [BrDim, dim]
        
        # Update m_i and l_i
        m_i, l_i = m_i_new, l_i_new  # m_i, l_i: [BrDim]
    
    # Compute final O_i and L_i
    O_i = O_i / l_i.unsqueeze(-1)  # O_i: [BrDim, dim]
    L_i = m_i + torch.log(l_i)  # L_i: [BrDim]
    
    return O_i, L_i  # O_i: [BrDim, dim], L_i: [BrDim]

def flash_attention_2(Q, K, V, BrDim, BcDim):
    Bs, Nh, N, dim = Q.shape  # Q: [Bs, Nh, N, dim]
    
    # Compute the number of row blocks
    Tr = math.ceil(N / BrDim)
    
    # Initialize output and logsumexp
    O = torch.zeros_like(Q)  # O: [Bs, Nh, N, dim]
    L = torch.zeros((Bs, Nh, N), device=Q.device)  # L: [Bs, Nh, N]
    
    for b in range(Bs):
        for h in range(Nh):
            for i in range(Tr):
                # Compute attention for this row block
                start_row = i * BrDim
                end_row = min((i + 1) * BrDim, N)
                Q_i = Q[b, h, start_row:end_row, :]  # Q_i: [BrDim, dim]
                O_i, L_i = flash_attention_2_row_block(Q_i, K[b, h], V[b, h], BcDim)
                
                # Write O_i and L_i to output
                O[b, h, start_row:end_row, :] = O_i  # O: [Bs, Nh, N, dim]
                L[b, h, start_row:end_row] = L_i  # L: [Bs, Nh, N]
    
    return O, L  # O: [Bs, Nh, N, dim], L: [Bs, Nh, N]

def standard_attention(Q, K, V):
    d = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d ** 0.5)
    attn = F.softmax(scores, dim=-1)
    output = torch.matmul(attn, V)
    return output, scores

def compare_attention_implementations(batch_size, num_heads, seq_len, dim, device='cuda'):
    # Generate random input tensors
    torch.manual_seed(123456778998)
    Q = torch.randn(batch_size, num_heads, seq_len, dim, device=device)
    K = torch.randn(batch_size, num_heads, seq_len, dim, device=device)
    V = torch.randn(batch_size, num_heads, seq_len, dim, device=device)

    # Compute attention using standard implementation
    standard_output, _ = standard_attention(Q, K, V)

    # Compute attention using FlashAttention-2
    flash_output, _ = flash_attention_2(Q, K, V, BrDim=32, BcDim=64)

    # Compute attention using torch.nn.functional.scaled_dot_product_attention
    torch_output = F.scaled_dot_product_attention(Q, K, V, dropout_p=0.0, is_causal=False)

    # Compare outputs
    max_diff_std = torch.max(torch.abs(standard_output - flash_output))
    print(f"Maximum difference with standard output: {max_diff_std.item()}")

    max_diff_torch = torch.max(torch.abs(torch_output - flash_output))
    print(f"Maximum difference with torch.nn.functional.scaled_dot_product_attention output: {max_diff_torch.item()}")

    # Check if outputs are close
    is_close_std = torch.allclose(standard_output, flash_output, rtol=1e-5, atol=1e-5)
    print(f"Outputs are close (Standard): {is_close_std}")

    is_close_torch = torch.allclose(torch_output, flash_output, rtol=1e-5, atol=1e-5)
    print(f"Outputs are close (torch.nn.functional.scaled_dot_product_attention): {is_close_torch}")

# Run comparison
compare_attention_implementations(batch_size=2, num_heads=2, seq_len=1024, dim=32)