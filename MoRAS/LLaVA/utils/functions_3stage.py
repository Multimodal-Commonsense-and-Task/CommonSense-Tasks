import torch
from utils.utils import *

def sigmoid(x, x0, k=1, sigmoid_inverse=False):
    
    if sigmoid_inverse:
        return 1.0 / (1.0 + np.exp(-k * (-x + x0)))
    else:
        return 1.0 / (1.0 + np.exp(-k * (x - x0)))
    

def get_sigmoid_parameters(eval_model):
    
    if eval_model == "llava_v1_5_7b":
        return 0.712, 15.955
        
    elif eval_model == "llava_v1_5_13b":
        return 0.741, 17.811
    
    
def generate_rephrased_question(args, model, conv, tokenizer, image_tensor, image_size, instruction):

    inp = get_prompt1(instruction)

    if model.config.mm_use_im_start_end:
        inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
    else:
        inp = DEFAULT_IMAGE_TOKEN + '\n' + inp

    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()


    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)


    with torch.inference_mode():
        output = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=[image_size],
            **get_generation_kwargs(args),
            use_cache=False,
            max_new_tokens=40,
            )


    generated_ids = output

    outputs = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()

    return outputs



def evaluate_rephrased_question(args, model, conv, tokenizer, image_tensor, image_size, instruction, n_token=3, first_token=True):

    inp = get_prompt2(instruction)
    
    if model.config.mm_use_im_start_end:
        inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
    else:
        inp = DEFAULT_IMAGE_TOKEN + '\n' + inp

    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()


    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)

    with torch.inference_mode():
        output = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=[image_size],
            **get_generation_kwargs(args),
            max_new_tokens=n_token,
            use_cache=True,
            output_hidden_states=True,
            return_dict_in_generate=True,
            )


    generated_ids = output.sequences
    hidden_states = output.hidden_states

    outputs = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()

    token_states = []
    for i in range(len(hidden_states)):
        temp = []
        for j in range(len(hidden_states[i])):
            temp.append(hidden_states[i][j][0, -1].detach().cpu().numpy())
        token_states.append(temp)

    token_states = np.array(token_states)

    if first_token:
        return outputs, token_states[0]
    else:
        return outputs, token_states


def generate_final_output(args, model, conv, tokenizer, image_tensor, image_size, instruction, sigmoid_x, sigmoid_k, shifting_coefficient=0):

    inp = instruction

    if model.config.mm_use_im_start_end:
        inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
    else:
        inp = DEFAULT_IMAGE_TOKEN + '\n' + inp

    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()


    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
    
    if args.mode == "ras_25e-2":
        shifting_coefficient = 0.25
    elif args.mode == "ras_50e-2":
        shifting_coefficient = 0.5
    elif args.mode == "ras_75e-2":
        shifting_coefficient = 0.75
    

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

    with torch.inference_mode():
        output = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=[image_size],
            **get_generation_kwargs(args),
            max_new_tokens=args.max_new_tokens,
            use_cache=False,
            shifting_params=[sigmoid_x, sigmoid_k, shifting_coefficient, input_ids.shape[1]+575, resolve_unsafe_states_file(args)]
            )


    generated_ids = output

    outputs = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()

    return outputs


# 3 token shifting coefficient (aggregated probabilities)
def get_shifting_coefficient(args, model, base_state1, sigmoid_x, sigmoid_k, n_token=3, gamma=0.3):
    
    unsafe_states_file = resolve_unsafe_states_file(args)
    if unsafe_states_file is not None:
        unsafe_states_mean = torch.load(unsafe_states_file)
    elif '7b' in args.eval_model:
        unsafe_states_mean = torch.load('./parameters/unsafe_states_prompt_7b.pt')
    elif '13b' in args.eval_model:
        unsafe_states_mean = torch.load('./parameters/unsafe_states_prompt_13b.pt')
        
    
    target_layer = -1
    
    p_list = []
    q_list = []

    
    for token_index in range(min(n_token, len(base_state1))):

        target_state = torch.from_numpy(base_state1[token_index, target_layer]).unsqueeze(0).to('cuda')
        if not torch.isfinite(target_state).all():
            print(f"[NaN debug] target_state is non-finite at token={token_index}, layer={target_layer}")
            print(f"[NaN debug] target_state finite ratio: {torch.isfinite(target_state).float().mean().item():.6f}")
            print(f"[NaN debug] target_state min/max: {torch.nan_to_num(target_state).min().item():.6f}, {torch.nan_to_num(target_state).max().item():.6f}")

        target_logits = model.lm_head(target_state)
        target_logits = target_logits.float()
        if not torch.isfinite(target_logits).all():
            print(f"[NaN debug] target_logits is non-finite at token={token_index}")
            print(f"[NaN debug] target_logits finite ratio: {torch.isfinite(target_logits).float().mean().item():.6f}")
        logits_q = target_logits.detach().cpu()
        q = F.softmax(logits_q, dim=-1)
        if not torch.isfinite(q).all():
            print(f"[NaN debug] q softmax is non-finite at token={token_index}")
            print(f"[NaN debug] logits_q finite ratio: {torch.isfinite(logits_q).float().mean().item():.6f}")
        
        q_list.append(q)

        unsafe_hidden_state = unsafe_states_mean[token_index].reshape(1, -1).to('cuda')
        
        unsafe_logits = model.lm_head(unsafe_hidden_state)
        unsafe_logits = unsafe_logits.float()
        logits_p = unsafe_logits.detach().cpu()
        p = F.softmax(logits_p, dim=-1)
        if not torch.isfinite(p).all():
            print(f"[NaN debug] p softmax is non-finite at token={token_index}")
        
        p_list.append(p)
        
    
    p = torch.stack(p_list, dim=0)
    q = torch.stack(q_list, dim=0)
    
    p = p.permute(1, 0, 2)
    q = q.permute(1, 0, 2)


    p_sum = torch.zeros_like(p[:, 0])
    q_sum = torch.zeros_like(q[:, 0])
    

    # exponential decay
    for token_index in range(min(n_token, len(base_state1))):
        p_sum += gamma ** token_index * p[:, token_index]
        q_sum += gamma ** token_index * q[:, token_index]

    if not torch.isfinite(p_sum).all() or not torch.isfinite(q_sum).all():
        print(f"[NaN debug] p_sum finite: {torch.isfinite(p_sum).all().item()}, q_sum finite: {torch.isfinite(q_sum).all().item()}")
        print(f"[NaN debug] p_sum finite ratio: {torch.isfinite(p_sum).float().mean().item():.6f}")
        print(f"[NaN debug] q_sum finite ratio: {torch.isfinite(q_sum).float().mean().item():.6f}")
        
        
    cosine_sim = cosine_similarity(p_sum, q_sum)[0][0]

    print('Sim score:', cosine_sim)
    shifting_coefficient = sigmoid(cosine_sim, sigmoid_x, sigmoid_k, sigmoid_inverse=False)
    
    
    return shifting_coefficient
    
    
    
def get_cossim(args, model, base_state1, gamma=0.3, n_token=3):
    
    if getattr(args, "unsafe_states_file", None) is not None:
        unsafe_states_mean = torch.load(args.unsafe_states_file)
    elif '7b' in args.eval_model:
        unsafe_states_mean = torch.load('./parameters/unsafe_states_prompt_7b.pt')
    elif '13b' in args.eval_model:
        unsafe_states_mean = torch.load('./parameters/unsafe_states_prompt_13b.pt')
    
    target_layer = -1
    
    p_list = []
    q_list = []

    
    for token_index in range(min(n_token, len(base_state1))):

        target_state = torch.from_numpy(base_state1[token_index, target_layer]).unsqueeze(0).to('cuda')

        target_logits = model.lm_head(target_state)
        target_logits = target_logits.float()
        logits_q = target_logits.detach().cpu()
        q = F.softmax(logits_q, dim=-1)
        
        q_list.append(q)

        unsafe_hidden_state = unsafe_states_mean[token_index].reshape(1, -1).to('cuda')
        
        unsafe_logits = model.lm_head(unsafe_hidden_state)
        unsafe_logits = unsafe_logits.float()
        logits_p = unsafe_logits.detach().cpu()
        p = F.softmax(logits_p, dim=-1)
        
        p_list.append(p)
        
    
    p = torch.stack(p_list, dim=0)
    q = torch.stack(q_list, dim=0)
    
    p = p.permute(1, 0, 2)
    q = q.permute(1, 0, 2)


    p_sum = torch.zeros_like(p[:, 0])
    q_sum = torch.zeros_like(q[:, 0])
    
    # exponential decay
    for token_index in range(min(n_token, len(base_state1))):
        p_sum += gamma ** token_index * p[:, token_index]
        q_sum += gamma ** token_index * q[:, token_index]
        
    cosine_sim = cosine_similarity(p_sum, q_sum)[0][0]
    
    return cosine_sim
    
    
