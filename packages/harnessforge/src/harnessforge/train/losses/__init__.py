"""Custom Tinker losses that run inside the miles-tinker trainer (`--tinker-custom-loss NAME=module:function`,
`gateway.custom_losses` in a run config).

Each is a per-datum function with miles-tinker's custom-loss signature:

    fn(*, logprobs, sampling_logprobs, advantages, loss_mask, config) -> (per_token_loss, metrics)

per_token_loss has the shape of `logprobs` and the trainer sums it (raw sum, as the built-in losses); metrics
({name: number or 0-dim tensor}) are summed over a request's datums into its forward_backward metrics as
`<name>:sum`. The modules here import only torch (and the stdlib): the trainer imports them from hf_src.

    dppo    TMax's DPPO (binary TV/KL trust region against the sampler, ratio cap)
"""
