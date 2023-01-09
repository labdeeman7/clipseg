#😉 This is gonna maybe be the hardest code for me to read. This and VITSEG. 
#😉 Okay, I think the main difference between this and VITSEG is that VITSEG is that the CLIP visual encoder is used. 

import math
from os.path import basename, dirname, join, isfile
import torch
from torch import nn
from torch.nn import functional as nnf
from torch.nn.modules.activation import ReLU

#😉 This gets a list of prompts. We format these prompts with input to the network. We have plain, fixed, shuffle, shuffle+ which chamges the style of the prompt. Note that most code have this get_prompt_list style. 
def get_prompt_list(prompt):
    if prompt == 'plain':
        return ['{}']    
    elif prompt == 'fixed':
        return ['a photo of a {}.']
    elif prompt == 'shuffle':
        return ['a photo of a {}.', 'a photograph of a {}.', 'an image of a {}.', '{}.']
    elif prompt == 'shuffle+':
        return ['a photo of a {}.', 'a photograph of a {}.', 'an image of a {}.', '{}.',
                            'a cropped photo of a {}.', 'a good photo of a {}.', 'a photo of one {}.',
                            'a bad photo of a {}.', 'a photo of the {}.']
    else:
        raise ValueError('Invalid value for prompt')        


def forward_multihead_attention(x, b, with_aff=False, attn_mask=None): #😉 Ohh so they did not use the block transformer directly from CLIP, instead they used another version. 
    #🙋‍♂️ What the fuck do they mean by the mlp and the layer norm come from CLIP. How? There should be an MLP and layer norm in the encoder of a transformer normally. They want to make changes to the transformer, probably because they want to add attention masks. 
    """ 
    Simplified version of multihead attention (taken from torch source code but without tons of if clauses). 
    The mlp and layer norm come from CLIP. 
    x: input.
    b: multihead attention module. 
    """
    #🙋‍♂️ What does this mean? that the layer norm and mlp comes from CLIP? 
    #😉 This code seems to be used in visual forward, so output of CLIP visual model. Why would this be needed? 👌 Because they want to make changes to the transformer in the CLIP if attention masks are available.   

    x_ = b.ln_1(x) #😉 Layer normalization 1. First, okay. 
    q, k, v = nnf.linear(x_, b.attn.in_proj_weight, b.attn.in_proj_bias).chunk(3, dim=-1) #🙋‍♂️ Not sure what attn. The block multi head attention? IN projection bias, not sure.  This all is so condision. I think the main thing is to perform encoding of a transformer, but leave space for attention masking. 
    tgt_len, bsz, embed_dim = q.size()

    head_dim = embed_dim // b.attn.num_heads
    scaling = float(head_dim) ** -0.5

    q = q.contiguous().view(tgt_len, bsz * b.attn.num_heads, b.attn.head_dim).transpose(0, 1)
    k = k.contiguous().view(-1, bsz * b.attn.num_heads, b.attn.head_dim).transpose(0, 1)
    v = v.contiguous().view(-1, bsz * b.attn.num_heads, b.attn.head_dim).transpose(0, 1)

    q = q * scaling

    attn_output_weights = torch.bmm(q, k.transpose(1, 2)) #  n_heads * batch_size, tokens^2, tokens^2
    if attn_mask is not None:


        attn_mask_type, attn_mask = attn_mask
        n_heads = attn_output_weights.size(0) // attn_mask.size(0)
        attn_mask = attn_mask.repeat(n_heads, 1)
        
        if attn_mask_type == 'cls_token':
            # the mask only affects similarities compared to the readout-token.
            attn_output_weights[:, 0, 1:] = attn_output_weights[:, 0, 1:] * attn_mask[None,...]
            # attn_output_weights[:, 0, 0] = 0*attn_output_weights[:, 0, 0]

        if attn_mask_type == 'all':
            # print(attn_output_weights.shape, attn_mask[:, None].shape)
            attn_output_weights[:, 1:, 1:] = attn_output_weights[:, 1:, 1:] * attn_mask[:, None]
        
    
    attn_output_weights = torch.softmax(attn_output_weights, dim=-1)

    attn_output = torch.bmm(attn_output_weights, v)
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = b.attn.out_proj(attn_output)

    x = x + attn_output
    x = x + b.mlp(b.ln_2(x))

    if with_aff:
        return x, attn_output_weights
    else:
        return x #


class CLIPDenseBase(nn.Module): #😉 This is the base of the network with the text and image CLIP model. 

    def __init__(self, version, reduce_cond, reduce_dim, prompt, n_tokens):
        super().__init__()

        import clip #😉 it imports clip as expected. 

        # prec = torch.FloatTensor
        self.clip_model, _ = clip.load(version, device='cpu', jit=False) #😉 CLIP model loaded.
        self.model = self.clip_model.visual #😉 the visual CLIP is kept as self.model.  

        # if not None, scale conv weights such that we obtain n_tokens.
        self.n_tokens = n_tokens #😉 if it is not none, scale the weights so we can obtain n_otkens, 🙋‍♂️problem is that how deos scaling reduce the number of simething, or increase its number? 

        for p in self.clip_model.parameters(): #😉 Do not train.
            p.requires_grad_(False)

        # conditional
        if reduce_cond is not None: #😉 Either text or image conditional/ We reduce the size of the conditional vector if a different value is givien. 🙋‍♂️My only problem is that the conditional variable says it is not trained. Which makes no sense. We thank god the conditional variable was never trained.
            self.reduce_cond = nn.Linear(512, reduce_cond)
            for p in self.reduce_cond.parameters():
                p.requires_grad_(False)
        else:
            self.reduce_cond = None        

        #😉film layers to be used at the beginning of the decoder. The mul and add inputs are the outputs from CLIP on the prompt.
        self.film_mul = nn.Linear(512 if reduce_cond is None else reduce_cond, reduce_dim)
        self.film_add = nn.Linear(512 if reduce_cond is None else reduce_cond, reduce_dim)
        
        self.reduce = nn.Linear(768, reduce_dim)

        self.prompt_list = get_prompt_list(prompt)   #😉 Get prompt list for the specifici prompt.  

        # precomputed prompts #😉 If prompts computed elesewere we can put them into the list directly here. But why get_prompt list, then get a list of precompute prompts.
        import pickle
        if isfile('precomputed_prompt_vectors.pickle'):
            precomp = pickle.load(open('precomputed_prompt_vectors.pickle', 'rb'))
            self.precomputed_prompts = {k: torch.from_numpy(v) for k, v in precomp.items()}        
        else:
            self.precomputed_prompts = dict()
    
    def rescaled_pos_emb(self, new_size): #😉 Rescale the pos_embedding if we rescaled
        assert len(new_size) == 2

        a = self.model.positional_embedding[1:].T.view(1, 768, *self.token_shape)
        b = nnf.interpolate(a, new_size, mode='bicubic', align_corners=False).squeeze(0).view(768, new_size[0]*new_size[1]).T
        return torch.cat([self.model.positional_embedding[:1], b])

    def visual_forward(self, x_inp, extract_layers=(), skip=False, mask=None): #😉 Corner stone code. This generates the outputs when an image is passed into CLIP Image branch without training. note clip Image is stored as self.model. 
        

        with torch.no_grad():

            inp_size = x_inp.shape[2:] #😉Image size. 0 and 1, are probably B and C

            if self.n_tokens is not None: #😉 if number of tokens is not None, it is usually non.  
                stride2 = x_inp.shape[2] // self.n_tokens #😉 A side, of the image, is used and divided by the desired amount of tokens. Assuming that we have 384 and we want 28 tokens, we would have strid2 be 13.🙋‍♂️ is this not too much?
                conv_weight2 = nnf.interpolate(self.model.conv1.weight, (stride2, stride2), mode='bilinear', align_corners=True) #😉 Lots of interpolation are been done. 🙋‍♂️I guess when working with something that has to do with freezing weights, then it maybe advisable to freeze weights. 🙋‍♂️They changed the wieghts of conv. still not sure why. THis is the conv weight gilter. I guess they are trying to ensure that the putput of conv1 is a particular shape.   
                x = nnf.conv2d(x_inp, conv_weight2, bias=self.model.conv1.bias, stride=stride2, dilation=self.model.conv1.dilation) #🙋‍♂️ COnv is done here with the updated shape conv1 as the filter weights. bias is the same. also the stride is stride2. 
            else:
                x = self.model.conv1(x_inp)  # shape = [*, width, grid, grid] #😉 Not sure what this is referring to. grid grid is the height and width of the token. width, ot sure, maybe the amount of tokens, not sure.🙋‍♂️ Also, this is the output of a conv1. So maybe width is the output of conv1. Also, we are working with conv1, why does a transformer model have a conv1? CLIP VIT does not have any convs, I am missing something definintely.  Is the conv1 done before entering the transformers, when linear projection of flattened patches is done? 

            x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2] #😉 okay they reshape. TO the shape said by the writer.   
            x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]  #😉 okay 

            x = torch.cat([self.model.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width] #😉 The took the class embeding CLS token., added this vector to a more complex array using boradcasting maybe. Then concatenated this value with the output of the conv x. 🙋‍♂️ Are they just trying to add the CLS token? should that not be done automatically?  

            standard_n_tokens = 50 if self.model.conv1.kernel_size[0] == 32 else 197 #😉 not exactly sure what is going on here. The standard amount of tokens should be 50, tokems, but if we had changed the size of the kernel? 🙋‍♂️Where do we change this weight? We changed the output of conv1, but not the conv1, itself. Lost to say the least.

            if x.shape[1] != standard_n_tokens: #😉 If we are not using the standard number of tokens, then,  
                new_shape = int(math.sqrt(x.shape[1]-1)) #🙋‍♂️ we root the shape -1, what??. Then we have to rescale the position embedding. Maybe they know what the way things shapes should be. Not sure. 
                x = x + self.rescaled_pos_emb((new_shape, new_shape)).to(x.dtype)[None,:,:] #😉 Add the positional embedding to x normally after rescaling  
            else:
                x = x + self.model.positional_embedding.to(x.dtype)  #😉 Add the positional embedding to x normally

            x = self.model.ln_pre(x) #🙋‍♂️ Not sure what it is.👌  It is a layer normalization. 🙋‍♂️But why not just use the forward of CLIP in the forward? Why does the method go through all this? 

            x = x.permute(1, 0, 2)  # NLD -> LND

            activations, affinities = [], [] #😉 Activations are the outputs of the transformers.👌 Ohhh that might be the reason that the CLIP network is not just used like that, because activations are needed. 🙋‍♂️ What are affinities? Also why are there so much options to change things?   

            for i, res_block in enumerate(self.model.transformer.resblocks): #😉 For each transformer block, 🙋‍♂️ maybe the transformer bloc it is residual
                
                if mask is not None: #🙋‍♂️ Not sure exactly what the mask, is. But apparently if it is there, there is a layer, there is a type, then a tensor. It is an attention mask. CLIPSEG talked about mask pooling in one-shot segmentation. Also, they talked about applying masking on the tolems, but that this did not work very well. 
                    mask_layer, mask_type, mask_tensor = mask
                    if mask_layer == i or mask_layer == 'all':
                        # import ipdb; ipdb.set_trace()
                        size = int(math.sqrt(x.shape[0] - 1))
                        
                        attn_mask = (mask_type, nnf.interpolate(mask_tensor.unsqueeze(1).float(), (size, size)).view(mask_tensor.shape[0], size * size)) #😉 An interpolation of the mask tensor and a size, which is sqrt of the x shape -1. 🙋‍♂️Not sure at this moment anymore
                        
                    else:
                        attn_mask = None
                else:
                    attn_mask = None

                x, aff_per_head = forward_multihead_attention(x, res_block, with_aff=True, attn_mask=attn_mask) #😉 This is where the MHA is used. So does that mean that we for each transformer block, we pass tge ers_bloc to forward_multihead_attention. note that with_add is true. and multihead_attention outputs aff per head. Affinities seem to be attn_output_weights

                if i in extract_layers: #😉 If we are extracting this layer for the decoder, then we 
                    affinities += [aff_per_head]  #😉 1, store affinities which are the attention output weights. 

                    #if self.n_tokens is not None:
                    #    activations += [nnf.interpolate(x, inp_size, mode='bilinear', align_corners=True)]
                    #else:
                    activations += [x] #😉 store activations, which are the outputs of the trransformer block.

                if len(extract_layers) > 0 and i == max(extract_layers) and skip:
                    print('early skip')
                    break
                
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = self.model.ln_post(x[:, 0, :]) #😉 Post layer normalization. 

            if self.model.proj is not None: #😉 Projections.
                x = x @ self.model.proj

            return x, activations, affinities

    def sample_prompts(self, words, prompt_list=None):

        prompt_list = prompt_list if prompt_list is not None else self.prompt_list

        prompt_indices = torch.multinomial(torch.ones(len(prompt_list)), len(words), replacement=True) #🙋‍♂️ Why is this multimodal used when we are working with prompts? They have not been converted to vetors yet. 👌 For randomization and selection of prompts I guess.
        prompts = [prompt_list[i] for i in prompt_indices]
        return [promt.format(w) for promt, w in zip(prompts, words)]

    def get_cond_vec(self, conditional, batch_size): #😉 Get the conditional vectors which refers to either an image or a text prompt. It takes in conditional and batch_size. 🙋‍♂️I never figured out why the batch_size has to be the size of the conditional.
        # compute conditional from a single string
        if conditional is not None and type(conditional) == str: #😉 If conditional is a string compute_conditional and repeat till batch size.
            cond = self.compute_conditional(conditional)
            cond = cond.repeat(batch_size, 1)

        # compute conditional from string list/tuple
        elif conditional is not None and type(conditional) in {list, tuple} and type(conditional[0]) == str:  #😉 If conditional is a list of strngs, ensure that the list is the same size as the batch_size, then compute conditional. I guess, one method is one text prompt for all images in the batch while the second is a text prompt per image.  
            assert len(conditional) == batch_size
            cond = self.compute_conditional(conditional)

        # use conditional directly
        elif conditional is not None and type(conditional) == torch.Tensor and conditional.ndim == 2: #😉 precomputed conditional I bet.
            cond = conditional

        # compute conditional from image
        elif conditional is not None and type(conditional) == torch.Tensor: #😉 Image conditional. visual_forward done. 
            with torch.no_grad():
                cond, _, _ = self.visual_forward(conditional)
        else:
            raise ValueError('invalid conditional')
        return cond   

    def compute_conditional(self, conditional): #😉 compute conditional for texts. 
        import clip

        dev = next(self.parameters()).device 

        if type(conditional) in {list, tuple}: #😉 if conditional is a list, tokenize the list and econd the text. 
            text_tokens = clip.tokenize(conditional).to(dev)
            cond = self.clip_model.encode_text(text_tokens)
        else:
            if conditional in self.precomputed_prompts: #😉 If conditionals have been precomputed use them. 
                cond = self.precomputed_prompts[conditional].float().to(dev)
            else: #😉 If  if it is a single string, make array, tokenize and encode. 
                text_tokens = clip.tokenize([conditional]).to(dev)
                cond = self.clip_model.encode_text(text_tokens)[0]
        
        if self.shift_vector is not None: #😉 If shift vector is present, then add shift vector to conditional. 
            return cond + self.shift_vector
        else:
            return cond


def clip_load_untrained(version): #😉Not used anywhere.
    assert version == 'ViT-B/16'
    from clip.model import CLIP
    from clip.clip import _MODELS, _download
    model = torch.jit.load(_download(_MODELS['ViT-B/16'])).eval()
    state_dict = model.state_dict()

    vision_width = state_dict["visual.conv1.weight"].shape[0]
    vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
    vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
    grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
    image_resolution = vision_patch_size * grid_size
    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    return CLIP(embed_dim, image_resolution, vision_layers, vision_width, vision_patch_size, 
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers)    


class CLIPDensePredT(CLIPDenseBase): #😉 THis is CLIPSEG main work. Note that the CLIP visual input is broken down entirely.  It also includes the decoder, similar to VITSEG.
    def __init__(self, version='ViT-B/32', extract_layers=(3, 6, 9), cond_layer=0, reduce_dim=128, n_heads=4, prompt='fixed', 
                 extra_blocks=0, reduce_cond=None, fix_shift=False,
                 learn_trans_conv_only=False,  limit_to_clip_only=False, upsample=False, 
                 add_calibration=False, rev_activations=False, trans_conv=None, n_tokens=None, complex_trans_conv=False):
        
        super().__init__(version, reduce_cond, reduce_dim, prompt, n_tokens)
        # device = 'cpu'

        self.extract_layers = extract_layers
        self.cond_layer = cond_layer
        self.limit_to_clip_only = limit_to_clip_only
        self.process_cond = None
        self.rev_activations = rev_activations
        
        depth = len(extract_layers)

        if add_calibration:
            self.calibration_conds = 1

        self.upsample_proj = nn.Conv2d(reduce_dim, 1, kernel_size=1) if upsample else None

        self.add_activation1 = True

        self.version = version
        
        self.token_shape = {'ViT-B/32': (7, 7), 'ViT-B/16': (14, 14)}[version]

        if fix_shift:
            # self.shift_vector = nn.Parameter(torch.load(join(dirname(basename(__file__)), 'clip_text_shift_vector.pth')), requires_grad=False)
            self.shift_vector = nn.Parameter(torch.load(join(dirname(basename(__file__)), 'shift_text_to_vis.pth')), requires_grad=False)
            # self.shift_vector = nn.Parameter(-1*torch.load(join(dirname(basename(__file__)), 'shift2.pth')), requires_grad=False)
        else:
            self.shift_vector = None

        if trans_conv is None:
            trans_conv_ks = {'ViT-B/32': (32, 32), 'ViT-B/16': (16, 16)}[version]
        else:
            # explicitly define transposed conv kernel size
            trans_conv_ks = (trans_conv, trans_conv)

        if not complex_trans_conv:
            self.trans_conv = nn.ConvTranspose2d(reduce_dim, 1, trans_conv_ks, stride=trans_conv_ks)
        else:
            assert trans_conv_ks[0] == trans_conv_ks[1]

            tp_kernels = (trans_conv_ks[0] // 4, trans_conv_ks[0] // 4)

            self.trans_conv = nn.Sequential(
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(reduce_dim, reduce_dim // 2, kernel_size=tp_kernels[0], stride=tp_kernels[0]),
                nn.ReLU(),
                nn.ConvTranspose2d(reduce_dim // 2, 1, kernel_size=tp_kernels[1], stride=tp_kernels[1]),               
            )

#        self.trans_conv = nn.ConvTranspose2d(reduce_dim, 1, trans_conv_ks, stride=trans_conv_ks)
        
        assert len(self.extract_layers) == depth

        self.reduces = nn.ModuleList([nn.Linear(768, reduce_dim) for _ in range(depth)])
        self.blocks = nn.ModuleList([nn.TransformerEncoderLayer(d_model=reduce_dim, nhead=n_heads) for _ in range(len(self.extract_layers))])
        self.extra_blocks = nn.ModuleList([nn.TransformerEncoderLayer(d_model=reduce_dim, nhead=n_heads) for _ in range(extra_blocks)])
        
        # refinement and trans conv

        if learn_trans_conv_only:
            for p in self.parameters():
                p.requires_grad_(False)
            
            for p in self.trans_conv.parameters():
                p.requires_grad_(True)

        self.prompt_list = get_prompt_list(prompt)


    def forward(self, inp_image, conditional=None, return_features=False, mask=None):

        assert type(return_features) == bool #😉 Ensuring a value is correct.

        inp_image = inp_image.to(self.model.positional_embedding.device)  #😉 move to correct device.

        if mask is not None: #😉 Mask also not supported for this method. Remember, the mask has to do with things similar to 
            raise ValueError('mask not supported')

        # x_inp = normalize(inp_image)
        x_inp = inp_image    #😉 Inpute image

        bs, dev = inp_image.shape[0], x_inp.device  #😉 device, bs is batch

        cond = self.get_cond_vec(conditional, bs)  #😉 First get the conditional vector using the conditional.

        visual_q, activations, _ = self.visual_forward(x_inp, extract_layers=[0] + list(self.extract_layers))  #😉 Visual forward on CLIP. activations are the outputs from various transfromers in CLIP including the final output.

        activation1 = activations[0]
        activations = activations[1:]

        _activations = activations[::-1] if not self.rev_activations else activations #😉  Not sure what rev_activations is. It reverses the activation list. 

        a = None #😉 Useful for doing something in the first step. 
        for i, (activation, block, reduce) in enumerate(zip(_activations, self.blocks, self.reduces)): #😉 for all the blocks. 
            
            if a is not None:
                a = reduce(activation) + a #linear layer + residual connection
            else:
                a = reduce(activation)

            if i == self.cond_layer:
                if self.reduce_cond is not None:
                    cond = self.reduce_cond(cond)
                
                a = self.film_mul(cond) * a + self.film_add(cond) #😉 film on the conditional layer.

            a = block(a) #😉 Transformer.

        for block in self.extra_blocks: #😉 Residual transformers without activations from earlier layers
            a = a + block(a)

        a = a[1:].permute(1, 2, 0) # rm cls token and -> BS, Feats, Tokens

        size = int(math.sqrt(a.shape[2]))

        a = a.view(bs, a.shape[1], size, size) #😉 reshape to almost shape of the image

        a = self.trans_conv(a) #😉 upconv.

        if self.n_tokens is not None:
            a = nnf.interpolate(a, x_inp.shape[2:], mode='bilinear', align_corners=True) #🙋‍♂️ why interpolate? Interpolation at the end of your network is not good normally.  

        if self.upsample_proj is not None:
            a = self.upsample_proj(a) #😉 Upsample projection
            a = nnf.interpolate(a, x_inp.shape[2:], mode='bilinear') #😉 Interpolate.

        if return_features: #😉 Returns the a values and activations.
            return a, visual_q, cond, [activation1] + activations
        else:
            return a, #😉 Returns just output.



class CLIPDensePredTMasked(CLIPDensePredT):

    def __init__(self, version='ViT-B/32', extract_layers=(3, 6, 9), cond_layer=0, reduce_dim=128, n_heads=4, 
                 prompt='fixed', extra_blocks=0, reduce_cond=None, fix_shift=False, learn_trans_conv_only=False, 
                 refine=None, limit_to_clip_only=False, upsample=False, add_calibration=False, n_tokens=None):

        super().__init__(version=version, extract_layers=extract_layers, cond_layer=cond_layer, reduce_dim=reduce_dim, 
                         n_heads=n_heads, prompt=prompt, extra_blocks=extra_blocks, reduce_cond=reduce_cond, 
                         fix_shift=fix_shift, learn_trans_conv_only=learn_trans_conv_only,
                         limit_to_clip_only=limit_to_clip_only, upsample=upsample, add_calibration=add_calibration,
                         n_tokens=n_tokens)

    def visual_forward_masked(self, img_s, seg_s):
        return super().visual_forward(img_s, mask=('all', 'cls_token', seg_s))

    def forward(self, img_q, cond_or_img_s, seg_s=None, return_features=False):

        if seg_s is None:
            cond = cond_or_img_s
        else:
            img_s = cond_or_img_s

            with torch.no_grad():
                cond, _, _ = self.visual_forward_masked(img_s, seg_s)

        return super().forward(img_q, cond, return_features=return_features)



class CLIPDenseBaseline(CLIPDenseBase):

    def __init__(self, version='ViT-B/32', cond_layer=0, 
                extract_layer=9, reduce_dim=128, reduce2_dim=None, prompt='fixed', 
                 reduce_cond=None, limit_to_clip_only=False, n_tokens=None):
        
        super().__init__(version, reduce_cond, reduce_dim, prompt, n_tokens)
        device = 'cpu'

        # self.cond_layer = cond_layer
        self.extract_layer = extract_layer
        self.limit_to_clip_only = limit_to_clip_only
        self.shift_vector = None

        self.token_shape = {'ViT-B/32': (7, 7), 'ViT-B/16': (14, 14)}[version]
        
        assert reduce2_dim is not None

        self.reduce2 = nn.Sequential(
            nn.Linear(reduce_dim, reduce2_dim),
            nn.ReLU(),
            nn.Linear(reduce2_dim, reduce_dim)
        )
        
        trans_conv_ks = {'ViT-B/32': (32, 32), 'ViT-B/16': (16, 16)}[version]
        self.trans_conv = nn.ConvTranspose2d(reduce_dim, 1, trans_conv_ks, stride=trans_conv_ks)


    def forward(self, inp_image, conditional=None, return_features=False):

        inp_image = inp_image.to(self.model.positional_embedding.device)

        # x_inp = normalize(inp_image)
        x_inp = inp_image

        bs, dev = inp_image.shape[0], x_inp.device

        cond = self.get_cond_vec(conditional, bs)

        visual_q, activations, affinities = self.visual_forward(x_inp, extract_layers=[self.extract_layer])

        a = activations[0]
        a = self.reduce(a)
        a = self.film_mul(cond) * a + self.film_add(cond)

        if self.reduce2 is not None:
            a = self.reduce2(a)

        # the original model would execute a transformer block here

        a = a[1:].permute(1, 2, 0) # rm cls token and -> BS, Feats, Tokens

        size = int(math.sqrt(a.shape[2]))

        a = a.view(bs, a.shape[1], size, size)
        a = self.trans_conv(a)

        if return_features:
            return a, visual_q, cond, activations
        else:
            return a,


class CLIPSegMultiLabel(nn.Module):

    def __init__(self, model) -> None:
        super().__init__()

        from third_party.JoEm.data_loader import get_seen_idx, get_unseen_idx, VOC

        self.pascal_classes = VOC

        from models.clipseg import CLIPDensePredT
        from general_utils import load_model
        # self.clipseg = load_model('rd64-vit16-neg0.2-phrasecut', strict=False)
        self.clipseg = load_model(model, strict=False)
        
        self.clipseg.eval()

    def forward(self, x):

        bs = x.shape[0]
        out = torch.ones(21, bs, 352, 352).to(x.device) * -10

        for class_id, class_name in enumerate(self.pascal_classes):
        
            fac = 3 if class_name == 'background' else 1

            with torch.no_grad():
                pred = torch.sigmoid(self.clipseg(x, class_name)[0][:,0]) * fac

            out[class_id] += pred


        out = out.permute(1, 0, 2, 3)

        return out

        # construct output tensor
                    
