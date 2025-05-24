import torch
import torch.nn as nn
import torch.nn.functional as F
def perplexity(logits, labels):
    """
    Calculate the perplexity of the model's predictions.
    
    Args:
        logits: The model's output logits.
        labels: The true labels.
        
    Returns:
        Perplexity value.
    """
    # Convert logits to probabilities
    # probs = torch.softmax(logits, dim=-1)
    
    # # Calculate the negative log likelihood
    # nll = -torch.log(probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1))
    # print("logits shape:", logits.shape, "labels shape:", labels.shape)
    logits = logits.view(-1, logits.size(-1))
    labels = labels.view(-1)
    loss = F.cross_entropy(logits, labels)
    perplexity  = torch.exp(loss)
    # Calculate the perplexity
    # perplexity = torch.exp(nll.mean())
    
    return perplexity