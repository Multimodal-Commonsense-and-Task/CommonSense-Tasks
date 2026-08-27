import torch

# process DDBM checkpoints

state_dict = torch.load("assets/ckpts/e2h_ema_0.9999_420000.pt", map_location="cpu")

module_list = []

for i in range(5, 16):
    if i == 8 or i == 12:
        continue
    module_list.append(f"input_blocks.{i}.1.qkv.weight")
    module_list.append(f"input_blocks.{i}.1.proj_out.weight")

module_list.append("middle_block.1.qkv.weight")
module_list.append("middle_block.1.proj_out.weight")

for i in range(0, 12):
    module_list.append(f"output_blocks.{i}.1.qkv.weight")
    module_list.append(f"output_blocks.{i}.1.proj_out.weight")

for name in module_list:
    state_dict[name] = state_dict[name].squeeze(-1)

torch.save(state_dict, "assets/ckpts/e2h_ema_0.9999_420000_adapted.pt")

state_dict = torch.load("assets/ckpts/diode_ema_0.9999_440000.pt", map_location="cpu")

module_list = []

for i in range(10, 18):
    if i == 12 or i == 15:
        continue
    module_list.append(f"input_blocks.{i}.1.qkv.weight")
    module_list.append(f"input_blocks.{i}.1.proj_out.weight")

module_list.append("middle_block.1.qkv.weight")
module_list.append("middle_block.1.proj_out.weight")

for i in range(0, 9):
    module_list.append(f"output_blocks.{i}.1.qkv.weight")
    module_list.append(f"output_blocks.{i}.1.proj_out.weight")

for name in module_list:
    state_dict[name] = state_dict[name].squeeze(-1)

torch.save(state_dict, "assets/ckpts/diode_ema_0.9999_440000_adapted.pt")
