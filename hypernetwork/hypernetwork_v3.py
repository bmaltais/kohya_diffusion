# v3: blocks for each resolution in up/down, blocks for every context

import torch


class HypernetworkModule(torch.nn.Module):
  def __init__(self, dim, mlp_ratio, multiplier=1.0, checkpoint=False):
    super().__init__()
    self.multiplier = multiplier
    self.checkpoint = checkpoint

    def net():
      linear1 = torch.nn.Linear(dim, dim * mlp_ratio)
      torch.nn.init.kaiming_normal_(linear1.weight)
      with torch.no_grad():
        linear1.weight *= 0.1
      linear1.bias.data.zero_()

      linear2 = torch.nn.Linear(dim * mlp_ratio, dim)
      torch.nn.init.xavier_uniform_(linear2.weight)
      with torch.no_grad():
        linear2.weight *= 0.1
      linear2.bias.data.zero_()

      norm = torch.nn.LayerNorm(dim)
      relu = torch.nn.ReLU(inplace=not checkpoint)
      return torch.nn.Sequential(norm, linear1, relu, linear2)
    self.layers1 = net()
    self.layers2 = net()

  def forward(self, x, context):
    if self.checkpoint:
      return context + torch.utils.checkpoint.checkpoint_sequential(self.layers1, 4, context) * self.multiplier, \
             context + torch.utils.checkpoint.checkpoint_sequential(self.layers2, 4, context) * self.multiplier
    return context + self.layers1(context) * self.multiplier, context + self.layers2(context) * self.multiplier


class Hypernetwork(torch.nn.Module):
  enable_dims = [320, 640, 1280]
  mlp_ratios = [1, 1, 1]
  num_conditioning_attns = 16
  conditioning_dim = 768
  conditioning_mlp_ratio = 1
  # return self.modules[Hypernetwork.enable_sizes.index(dim)]

  def __init__(self, multiplier=1.0) -> None:
    super().__init__()
    self.down_modules = []
    self.up_modules = []
    self.cc_modules = []

    # down blocks
    for dim, mlp_ratio in zip(Hypernetwork.enable_dims, Hypernetwork.mlp_ratios):
      self.down_modules.append(HypernetworkModule(dim, mlp_ratio, multiplier))

    # middle block
    dim = Hypernetwork.enable_dims[-1]
    mlp_ratio = Hypernetwork.mlp_ratios[-1]
    self.middle_modules = [HypernetworkModule(dim, mlp_ratio, multiplier)]

    # up blocks
    for dim, mlp_ratio in zip(Hypernetwork.enable_dims, Hypernetwork.mlp_ratios):
      self.up_modules.append(HypernetworkModule(dim, mlp_ratio, multiplier))

    # conditioning
    dim = Hypernetwork.conditioning_dim
    for _ in range(Hypernetwork.num_conditioning_attns):
      self.cc_modules.append(HypernetworkModule(dim, Hypernetwork.conditioning_mlp_ratio, multiplier))

    for i, module in enumerate(self.down_modules + self.middle_modules + self.up_modules + self.cc_modules):
      self.register_module(f"m{i}", module)

  def apply_to_stable_diffusion(self, text_encoder, vae, unet):
    return self.apply_to_unet_common(True, unet.input_blocks, unet.middle_block, unet.output_blocks)

  def apply_to_diffusers(self, text_encoder, vae, unet):
    return self.apply_to_unet_common(False, unet.down_blocks, unet.mid_block, unet.up_blocks)

  def apply_to_unet_common(self, is_sd, input_blocks, middle_block, output_blocks):
    def apply_to_blocks(blocks, dims, modules):
      i = 0
      for block in blocks:
        if is_sd:
          subblks = block
        else:
          if not hasattr(block, 'attentions'):
            continue
          subblks = block.attentions

        for subblk in subblks:
          if 'SpatialTransformer' in str(type(subblk)) or 'Transformer2DModel' in str(type(subblk)):      # 0.6.0 and 0.7~
            for tf_block in subblk.transformer_blocks:
              for attn in [tf_block.attn1, tf_block.attn2]:
                dim = attn.context_dim if is_sd else attn.to_k.in_features
                if dim in dims:
                  # print("apply hyp", dim, i)
                  attn.hypernetwork = modules[i]
                  i += 1

    modules = [self.down_modules[0], self.down_modules[0],
               self.down_modules[1], self.down_modules[1],
               self.down_modules[2], self.down_modules[2]]
    apply_to_blocks(input_blocks, Hypernetwork.enable_dims, modules)

    apply_to_blocks([middle_block], [Hypernetwork.enable_dims[-1]], self.middle_modules)

    modules = [self.up_modules[2], self.up_modules[2], self.up_modules[2],
               self.up_modules[1], self.up_modules[1], self.up_modules[1],
               self.up_modules[0], self.up_modules[0], self.up_modules[0]]
    apply_to_blocks(output_blocks, Hypernetwork.enable_dims, modules)

    apply_to_blocks(input_blocks + [middle_block] + output_blocks, [Hypernetwork.conditioning_dim], self.cc_modules)

    return True       # TODO error checking

  # never called
  def forward(self, x, context, module):
    return module.forward(x, context)

  def load_from_state_dict(self, state_dict):
    self.load_state_dict(state_dict)
    return True

  def get_state_dict(self):
    state_dict = self.state_dict()
    return state_dict
