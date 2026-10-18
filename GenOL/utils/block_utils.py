import torch

MODEL_BLOCK_DICT = {
    'resnet18': [['initial'],
                 ['group1.blocks.block0'],
                 ['group1.blocks.block1'],
                 ['group2.blocks.block0'],
                 ['group2.blocks.block1'],
                 ['group3.blocks.block0'],
                 ['group3.blocks.block1'],
                 ['group4.blocks.block0'],
                 ['group4.blocks.block1'],
                 ['pool', 'fc']],
    
    'resnet32': [['initial'],
                 ['group1.blocks.block0'],
                 ['group1.blocks.block1'],
                 ['group1.blocks.block2'],
                 ['group1.blocks.block3'],
                 ['group1.blocks.block4'],
                 ['group2.blocks.block0'],
                 ['group2.blocks.block1'],
                 ['group2.blocks.block2'],
                 ['group2.blocks.block3'],
                 ['group2.blocks.block4'],
                 ['group3.blocks.block0'],
                 ['group3.blocks.block1'],
                 ['group3.blocks.block2'],
                 ['group3.blocks.block3'],
                 ['group3.blocks.block4'],
                 ['pool', 'fc']],
    
    'vit': [['vit_model.patch_embed', 'vit_model.pos_drop', 'vit_model.patch_drop', 'vit_model.norm_pre'], 
            ['vit_model.blocks.0'], 
            ['vit_model.blocks.1'], 
            ['vit_model.blocks.2'], 
            ['vit_model.blocks.3'], 
            ['vit_model.blocks.4'], 
            ['vit_model.blocks.5'], 
            ['vit_model.blocks.6'], 
            ['vit_model.blocks.7'], 
            ['vit_model.blocks.8'], 
            ['vit_model.blocks.9'], 
            ['vit_model.blocks.10'], 
            ['vit_model.blocks.11'], 
            ['vit_model.norm', 'vit_model.fc_norm', 'vit_model.head_drop', 'head']]
}

REMIND_MODEL_BLOCK_DICT = {
    'resnet18': [['model_G.initial'],
                 ['model_G.group1.blocks.block0'],
                 ['model_G.group1.blocks.block1'],
                 ['model_G.group2.blocks.block0'],
                 ['model_G.group2.blocks.block1'],
                 ['model_G.group3.blocks.block0'],
                 ['model_G.group3.blocks.block1'],
                 ['model_F.group4.blocks.block0'],
                 ['model_F.group4.blocks.block1'],
                 ['model_F.pool', 'fc']],
    
    'resnet32': [['model_G.initial'],
                 ['model_G.group1.blocks.block0'],
                 ['model_G.group1.blocks.block1'],
                 ['model_G.group1.blocks.block2'],
                 ['model_G.group1.blocks.block3'],
                 ['model_G.group1.blocks.block4'],
                 ['model_G.group2.blocks.block0'],
                 ['model_G.group2.blocks.block1'],
                 ['model_G.group2.blocks.block2'],
                 ['model_G.group2.blocks.block3'],
                 ['model_G.group2.blocks.block4'],
                 ['model_G.group3.blocks.block0'],
                 ['model_G.group3.blocks.block1'],
                 ['model_G.group3.blocks.block2'],
                 ['model_F.group3.blocks.block3'],
                 ['model_F.group3.blocks.block4'],
                 ['model_F.pool','fc']],
    
    'vit': [['model_G.0'], 
            ['model_G.1'], 
            ['model_G.2'], 
            ['model_G.3'], 
            ['model_G.4'], 
            ['model_G.5'], 
            ['model_G.6'], 
            ['model_G.7'], 
            ['model_G.8'], 
            ['model_G.9'], 
            ['model_F.0'], 
            ['model_F.1'], 
            ['model_F.2'], 
            ['norm'],['fc']]
}

MEMO_MODEL_BLOCK_DICT = {
    'resnet18': [['backbone.initial'],
                 ['backbone.group1.blocks.block0'],
                 ['backbone.group1.blocks.block1'],
                 ['backbone.group2.blocks.block0'],
                 ['backbone.group2.blocks.block1'],
                 ['backbone.group3.blocks.block0'],
                 ['backbone.group3.blocks.block1'],
                 ['AdaptiveExtractors.0.group4.blocks.block0'],
                 ['AdaptiveExtractors.0.group4.blocks.block1'],
                 ['AdaptiveExtractors.0.pool'], ['fc']],
    
    'resnet32': [['backbone.initial'],
                 ['backbone.group1.blocks.block0'],
                 ['backbone.group1.blocks.block1'],
                 ['backbone.group1.blocks.block2'],
                 ['backbone.group1.blocks.block3'],
                 ['backbone.group1.blocks.block4'],
                 ['backbone.group2.blocks.block0'],
                 ['backbone.group2.blocks.block1'],
                 ['backbone.group2.blocks.block2'],
                 ['backbone.group2.blocks.block3'],
                 ['backbone.group2.blocks.block4'],
                 ['AdaptiveExtractors.0.group3.blocks.block0'],
                 ['AdaptiveExtractors.0.group3.blocks.block1'],
                 ['AdaptiveExtractors.0.group3.blocks.block2'],
                 ['AdaptiveExtractors.0.group3.blocks.block3'],
                 ['AdaptiveExtractors.0.group3.blocks.block4'],
                 ['AdaptiveExtractors.0.pool'], ['fc']],
    
    'vit': [['backbone.0'], 
            ['backbone.1'], 
            ['backbone.2'], 
            ['backbone.3'], 
            ['backbone.4'], 
            ['backbone.5'], 
            ['backbone.6'], 
            ['backbone.7'], 
            ['AdaptiveExtractors.0.0'], 
            ['AdaptiveExtractors.0.1'],
            ['AdaptiveExtractors.0.2'], 
            ['AdaptiveExtractors.0.3'], 
            ['AdaptiveExtractors.0.4'], 
            ['norm', 'fc']]
}


def get_blockwise_flops(flops_dict, model_name, method=None):

    if method=="memo":
        assert model_name in MEMO_MODEL_BLOCK_DICT.keys()
        block_list = MEMO_MODEL_BLOCK_DICT[model_name]
    elif method=="remind":
        assert model_name in REMIND_MODEL_BLOCK_DICT.keys()
        block_list = REMIND_MODEL_BLOCK_DICT[model_name]
    else:
        assert model_name in MODEL_BLOCK_DICT.keys()
        block_list = MODEL_BLOCK_DICT[model_name]
    
    forward_flops = []
    backward_flops = []
    G_forward_flops = []
    G_backward_flops = []
    F_forward_flops = []
    F_backward_flops = []
    G_forward, G_backward, F_forward, F_backward = [], [], [], []
    
    for block in block_list:
        forward_flops.append(sum([flops_dict[layer]['forward_flops']/10e9 for layer in block]))
        backward_flops.append(sum([flops_dict[layer]['backward_flops']/10e9 for layer in block]))
    
        if method=="remind":
            for layer in block:
                if "model_G" in layer:
                    G_forward.append(flops_dict[layer]['forward_flops']/10e9)
                    G_backward.append(flops_dict[layer]['backward_flops']/10e9)
                else:
                    F_forward.append(flops_dict[layer]['forward_flops']/10e9)
                    F_backward.append(flops_dict[layer]['backward_flops']/10e9)
                    
            G_forward_flops.append(sum(G_forward))
            G_backward_flops.append(sum(G_backward))
            F_forward_flops.append(sum(F_forward))
            F_backward_flops.append(sum(F_backward))
            
        elif method=="memo":
            for layer in block:
                if "backbone" in layer:
                    G_forward.append(flops_dict[layer]['forward_flops']/10e9)
                    G_backward.append(flops_dict[layer]['backward_flops']/10e9)
                elif "AdaptiveExtractors" in layer or "fc" in layer:
                    F_forward.append(flops_dict[layer]['forward_flops']/10e9)
                    F_backward.append(flops_dict[layer]['backward_flops']/10e9)
  
            G_forward_flops.append(sum(G_forward))
            G_backward_flops.append(sum(G_backward))
            F_forward_flops.append(sum(F_forward))
            F_backward_flops.append(sum(F_backward))


    return forward_flops, backward_flops, G_forward_flops, G_backward_flops, F_forward_flops, F_backward_flops