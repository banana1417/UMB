import torch
import torch.nn.functional as F


def wasserstein2_distance(mu_a, var_a, mu_b, var_b, eps=1e-6):
    """Squared W2 distance for diagonal Gaussian distributions."""
    std_a = torch.sqrt(var_a.clamp_min(eps))
    std_b = torch.sqrt(var_b.clamp_min(eps))
    return (mu_a - mu_b).pow(2).sum(dim=-1) + (std_a - std_b).pow(2).sum(dim=-1)


def pair_distribution_contrastive_loss(dist_a, dist_b, temperature=0.3, eps=1e-6):
    """InfoNCE over all cross-sample pairs in a minibatch."""
    mu_a, var_a = dist_a
    mu_b, var_b = dist_b
    bsz = mu_a.size(0)
    distances = wasserstein2_distance(
        mu_a[:, None, :], var_a[:, None, :],
        mu_b[None, :, :], var_b[None, :, :], eps=eps,
    )
    logits = -distances / temperature
    labels = torch.arange(bsz, device=mu_a.device)
    return F.cross_entropy(logits, labels)


def interaction_contrastive_loss(outputs, temperature=0.3, eps=1e-6):
    title_image = pair_distribution_contrastive_loss(
        (outputs["mu_title"], outputs["var_title"]),
        (outputs["mu_image"], outputs["var_image"]),
        temperature, eps,
    )
    true_false = pair_distribution_contrastive_loss(
        (outputs["mu_true_reason"], outputs["var_true_reason"]),
        (outputs["mu_false_reason"], outputs["var_false_reason"]),
        temperature, eps,
    )
    return 0.5 * (title_image + true_false)


def monte_carlo_classification_loss(outputs, labels, num_samples=7, eps=1e-6):
    """Equation (15): BCE on the average of sigmoid samples."""
    mu = outputs["logit_mu"]
    std = torch.sqrt(outputs["pred_var"].clamp_min(eps))
    noise = torch.randn(num_samples, *mu.shape, device=mu.device, dtype=mu.dtype)
    sampled_logits = mu.unsqueeze(0) + std.unsqueeze(0) * noise
    mean_probability = torch.sigmoid(sampled_logits).mean(dim=0)
    labels = labels.float()
    loss = -(
        labels * torch.log(mean_probability.clamp_min(eps))
        + (1.0 - labels) * torch.log((1.0 - mean_probability).clamp_min(eps))
    ).mean()
    return loss, mean_probability


def total_loss(outputs, labels, contrastive_weight=0.10, temperature=0.3, samples=7, eps=1e-6):
    mc_loss, mc_probability = monte_carlo_classification_loss(
        outputs, labels, num_samples=samples, eps=eps
    )
    contrastive = interaction_contrastive_loss(outputs, temperature, eps)
    return {
        "loss": mc_loss + contrastive_weight * contrastive,
        "loss_mc": mc_loss,
        "loss_contrastive": contrastive,
        "mc_probability": mc_probability.detach(),
    }
