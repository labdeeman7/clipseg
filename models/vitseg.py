#😉 I think I would start from this. It would give some concepts about the way transformers work especially the VIT. 
# 😉This code is supposed to illustrate that for a CLIP style network, it is not good to use CLIP text without using CLIP Image model. 

import math
from posixpath import basename, dirname, join
# import clip
from clip.model import convert_weights
import torch
import json
from torch import nn
from torch.nn import functional as nnf
from torch.nn.modules import activation
from torch.nn.modules.activation import ReLU
from torchvision import transforms

normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)) #😉This is not the imagenet normalization, I assume it is the normalization of a dataset that the authors have on hand.

from torchvision.models import ResNet


def process_prompts(conditional, prompt_list, conditional_map):
    # DEPRECATED #😉 Good to know.
            
    # randomly sample a synonym
    words = [conditional_map[int(i)] for i in conditional]
    words = [syns[torch.multinomial(torch.ones(len(syns)), 1, replacement=True).item()] for syns in words]
    words = [w.replace('_', ' ') for w in words]

    if prompt_list is not None:
        prompt_indices = torch.multinomial(torch.ones(len(prompt_list)), len(words), replacement=True)
        prompts = [prompt_list[i] for i in prompt_indices]
    else:
        prompts = ['a photo of {}'] * (len(words))

    return [promt.format(w) for promt, w in zip(prompts, words)] #😉 This makes more sense. We are adding the name of the object to the prompt.


#😉 So there are no docstrings for functions and classes which suck. I also should ensure to add docstrings for my code, so I do not suck.
class VITDenseBase(nn.Module): 
    #😉 there is no init for this class. so we do not really know what the class wants
    
    def rescaled_pos_emb(self, new_size): #😉 Okay, positional embeddings if I remember correctly are things that are added to the various parts of the image, or a text to ensure that we can track the positional relationship between various text/images after they have been split up. They can be sine waves, or even free random variables to be learnt/
        #🙋‍♂️ Why would one need to rescale the position embedings? of all possition embedding apart from cls?. The function rescaled_pos_emb does not seem to be used anywhere.
        assert len(new_size) == 2

        a = self.model.positional_embedding[1:].T.view(1, 768, *self.token_shape) #🙋‍♂️ I do not remember nn.module have a self.model variabl. 👌 It is a method for the child class. I am not exactly sure what 768 is, but I am guessing that this is the size of each positional embedding.
        #🙋‍♂️ Positional embedding is likely a list. 
        # 🙋‍♂️They do not wanna include the first positional embedding in this calculation which is likely the 0 pos embeddig for CLS. 
        # 🙋‍♂️Token.shape is the shape of each token. I am guessing if it is flattened would just be a value, if not it would be a 2dim tuple that is unrolled. 👌It is a tuple (14x14). I think the flatten occurs somewhere inside the model.
        # 😉size of a appears to be 1x768x14x14. 

        b = nnf.interpolate(a, new_size, mode='bicubic', align_corners=False).squeeze(0).view(768, new_size[0]*new_size[1]).T #👌 Nice. Changes the size of the token shape to new_size.
        return torch.cat([self.model.positional_embedding[:1], b]) #👌 But nothing happens to the first positional embedding. 🙋‍♂️ Would this not cause trouble later? 

    def visual_forward(self, x_inp, extract_layers=(), skip=False, mask=None): #😉 mask and skip not used in function and mask is not supported anywhere in this code base. 
        # extract layers is a tuple with indexes of indices to be extracted.  

        
        with torch.no_grad(): #🙋‍♂️ Why no grad? Is this not a forward? Probably because this is not to train the network but to extract results such as the output of the network and activatons at various parts of the encoder. 

            x_inp = nnf.interpolate(x_inp, (384, 384)) #🙋‍♂️ why interpolate to 384? Maybe because the input to the model is an image and we are just interpolating this image 👌 Yes.
            # 🙋‍♂️What is the input? Is that the patch embeddings? or the full image(maybe this)  👌 Image is the input.

            x = self.model.patch_embed(x_inp) #🙋‍♂️ I guess turn the image into a patch embeddings.
            cls_token = self.model.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks #🙋‍♂️I guess the cls token is gotten directly from here? Not sure, a bit furstrated.
            if self.model.dist_token is None: #😉 I guess the original paper had something called distillation tokens. Not sure what they do, I would have to check the google implementation or some review of the code. 
                x = torch.cat((cls_token, x), dim=1) #😉 Concatenate cls with other tokens
            else:
                x = torch.cat((cls_token, self.model.dist_token.expand(x.shape[0], -1, -1), x), dim=1) #😉 if distil token present, Concatenate cls, dist token, and image embeddings.
            x = self.model.pos_drop(x + self.model.pos_embed) #😉 Checking TIMM code, pos_drop seems to be a dropout method.

            activations = [] #🙋‍♂️ I guess this is where activations which are passed into the decoder are stored.
            for i, block in enumerate(self.model.blocks): #😉 For each block, blocks are  MHA with the addition and layer normalization.
                x = block(x)

                if i in extract_layers:
                    # permute to be compatible with CLIP
                    activations += [x.permute(1,0,2)]   #😉 Do not forget, the + operator on lists is the same as the extend operator. The activations are definitely what is used for the decoders.         
 
            x = self.model.norm(x) #😉 A layer normalization. 
            x = self.model.head(self.model.pre_logits(x[:, 0])) #😉 A linear layer for the final output. 

            # again for CLIP compatibility
            # x = x.permute(1, 0, 2)

        return x, activations, None

    def sample_prompts(self, words, prompt_list=None): #🙋‍♂️ not still sure what this function does. It is called sample prompts. Maybe sampling text prompts for some form of data augmentation?
        #🙋‍♂️I am not sure what the inputs to the prompt lists refer to. words, prompt_list?
        prompt_list = prompt_list if prompt_list is not None else self.prompt_list  #😉 use function input if available, if not, use prompt list from class. 

        prompt_indices = torch.multinomial(torch.ones(len(prompt_list)), len(words), replacement=True) #🙋‍♂️ What happened here. Why do we need a mulitinomial distribution? 
        prompts = [prompt_list[i] for i in prompt_indices] #🙋‍♂️ Not sire what exactly is going on here honestly. Is words a vector or a string, are we zipping prompts with words? All not sure. 
        return [promt.format(w) for promt, w in zip(prompts, words)]

    def get_cond_vec(self, conditional, batch_size): #🙋‍♂️ What are the conditional vectors? Compute conditionals from a string? So we get a string and then we compute how the string relates to a conditional?  
        # compute conditional from a single string
        if conditional is not None and type(conditional) == str: #😉 If the conditional is a string, repeat the conditional the size of batches.
            cond = self.compute_conditional(conditional) #😉 Compute conditional is a method. 🙋‍♂️ I believe it takes the text prompt and encodes it, ready to be used for the film layer
            cond = cond.repeat(batch_size, 1)  #🙋‍♂️ I beleive that the conditionals are repeated the amount of batch, because we want to ensure that each conditional can be used on all images in the batch. 

        # compute conditional from string list/tuple
        elif conditional is not None and type(conditional) in {list, tuple} and type(conditional[0]) == str:  #😉 If conditional is a list, ensue list is the same size as batch. 🙋‍♂️We ll that essentially throws away my theory of looping over each text prompt/ 
            assert len(conditional) == batch_size #🙋‍♂️ I am not sure why conditionals in a list need to be the size of the batch. 
            cond = self.compute_conditional(conditional)

        # use conditional directly
        elif conditional is not None and type(conditional) == torch.Tensor and conditional.ndim == 2: #🙋‍♂️ If condiional is already done, then use it. 
            cond = conditional

        # compute conditional from image #😉 I think I was right. Conditional is the conditional image. It passes through visual_forward, which requires no_grad hahahahahaha.
        elif conditional is not None and type(conditional) == torch.Tensor:
            with torch.no_grad():
                cond, _, _ = self.visual_forward(conditional)
        else:
            raise ValueError('invalid conditional')
        return cond   

    def compute_conditional(self, conditional):
        import clip #😉 Clip needed to compute conditional on a single string. 🙋‍♂️ Conditional maybe the input text prompt

        dev = next(self.parameters()).device #😉 Get the device the parameters of the network are on. 

        if type(conditional) in {list, tuple}: #😉 if the conditional is some form of array, a bunch of conditionals given 
            text_tokens = clip.tokenize(conditional).to(dev) #😉first tokenize(used to encode text)  
            cond = self.clip_model.encode_text(text_tokens)  #🙋‍♂️ Then encode text. I guess pass it through the encoder
        else:
            if conditional in self.precomputed_prompts: #🙋‍♂️ if the conditional is just an int or more likely a key in a dict, check if it is in precomputed  
                cond = self.precomputed_prompts[conditional].float().to(dev)
            else: #😉 if not tokenize and encode as well. 
                text_tokens = clip.tokenize([conditional]).to(dev) 
                cond = self.clip_model.encode_text(text_tokens)[0]
        
        return cond


class VITDensePredT(VITDenseBase): #🙋‍♂️VITDenseBase seems to be a parent class of the VITDensePredT where I guess all we are adding is the ability to predict? Maybe the encoder + decoder? 
    # 😉It seems the Prediction and training are going to be done here. While the extraction with CLIP is done inn VITDenseBase. Also note that the visual pred used timm. THe final clipseg probabl does not use timm. 

    def __init__(self, extract_layers=(3, 6, 9), cond_layer=0, reduce_dim=128, n_heads=4, prompt='fixed', 
                 depth=3, extra_blocks=0, reduce_cond=None, fix_shift=False,
                 learn_trans_conv_only=False, refine=None, limit_to_clip_only=False, upsample=False, 
                 add_calibration=False, process_cond=None, not_pretrained=False): #😉 THe extract layers are 3, 6 and 9. Not all layers like in u-net, cond_layer = 0. Not sure what this referes to, 
                 #🙋‍♂️ reduce_dim not sure, probably the place where they are talking about rescaling the dimension of the position embedding, so maybe this.
        super().__init__()
        # device = 'cpu'

        self.extract_layers = extract_layers #😉 Just adding properties to the class maybe. 
        self.cond_layer = cond_layer #🙋‍♂️ I am guessing that this is the layer in which the conditional layer is put into the decoder   
        self.limit_to_clip_only = limit_to_clip_only #🙋‍♂️ Not sure what this variable is.
        self.process_cond = None #🙋‍♂️ Not sure what this is currently. 
        
        if add_calibration: 
            self.calibration_conds = 1 #🙋‍♂️ Not sure where calibration is needed  

        self.upsample_proj = nn.Conv2d(reduce_dim, 1, kernel_size=1) if upsample else None #🙋‍♂️ upsample projection, maybe something in the decoder. 

        self.add_activation1 = True #🙋‍♂️ Add activation. WHat does this mean? 🙋‍♂️ I am still guessing it is the CLS token. I need to reread how self attention works in the transformer to make this decision.

        import timm 
        self.model = timm.create_model('vit_base_patch16_384', pretrained=True) #😉This is where self.model is declared, but in VITDenseBase, we should have also declared self.model as None in my opinion then overided it. It is from TIMM. Timm has a vision-transformer model in it. Note that it is a port from the original code, so we mogith be required to look at the original code. For now, guess!
        self.model.head = nn.Linear(768, 512 if reduce_cond is None else reduce_cond) #😉 THe head of the VIT model we used the redice_cond if it exists as the output of the conditional. 

        for p in self.model.parameters(): # 😉 Remove grad  for the vit entirely. As it is not gonna be trained. 
            p.requires_grad_(False) 

        import clip
        self.clip_model, _ = clip.load('ViT-B/16', device='cpu', jit=False) #🙋‍♂️ Note that we only need the text portion of clip 
        # del self.clip_model.visual #😉 We do not need the visual part, as the author tried to delete it. 
        
        
        self.token_shape = (14, 14) #😉 shape of the tokens, which is how much we are reducing the image. If the image, is 384x384, then there should be a litle above 27 or 28 features. flattend I am expect around 784 size. 

        # conditional
        if reduce_cond is not None: 
            self.reduce_cond = nn.Linear(512, reduce_cond) #😉Reduce conditional is a linear layer for the conditional part of the network I am sure. 
            for p in self.reduce_cond.parameters(): #🙋‍♂️ What??? We do not traine the reduced conditional? WHy? is it then initialized to random?
                p.requires_grad_(False)
        else:
            self.reduce_cond = None

        #😉 Film layer finally. Remember film requires two inputs. the input from the encoder which is to be modulated by the conditional input. The multi and addition are based on the conditional. 
        # 🙋‍♂️I assume the conditional enter mul and add linear layers. Then the output is used to multiply and add the signal from the encoder.  👌 I am correct.
           
        # self.film = AVAILABLE_BLOCKS['film'](512, 128)
        self.film_mul = nn.Linear(512 if reduce_cond is None else reduce_cond, reduce_dim)
        self.film_add = nn.Linear(512 if reduce_cond is None else reduce_cond, reduce_dim)
        
        # DEPRECATED
        # self.conditional_map = {c['id']: c['synonyms'] for c in json.load(open(cond_map))}
        
        assert len(self.extract_layers) == depth #🙋‍♂️ Depth is likely the depth of the decoder. A big question is how is the decoder designed. IT is fully made of transformers/ 
        #🙋‍♂️ DOes that mean only an encoder layer with MHA, layer norm and Addition. I believe so. 

        self.reduces = nn.ModuleList([nn.Linear(768, reduce_dim) for _ in range(depth)])  #🙋‍♂️ I believe the 768 is the output for the vit model. There is a nn.linear for each laye 
        self.blocks = nn.ModuleList([nn.TransformerEncoderLayer(d_model=reduce_dim, nhead=n_heads) for _ in range(len(self.extract_layers))]) #🙋‍♂️ There is a block for each layer. I beleive the projection is likely gonna be added adter this layer. 
        self.extra_blocks = nn.ModuleList([nn.TransformerEncoderLayer(d_model=reduce_dim, nhead=n_heads) for _ in range(extra_blocks)]) #😉 Second encoder after the first encoder.

        trans_conv_ks = (16, 16) #🙋‍♂️ Transpose convolution. Definitely for increasing the size of the features. But transconvs work on grids, and we do not have a grid. 
        self.trans_conv = nn.ConvTranspose2d(reduce_dim, 1, trans_conv_ks, stride=trans_conv_ks)

        # refinement and trans conv
        #😉 This is false by the way.
        if learn_trans_conv_only:
            for p in self.parameters():
                p.requires_grad_(False)
            
            for p in self.trans_conv.parameters():
                p.requires_grad_(True)

        #😉 I was right. the prompt and prompt_list is a list of strings, that gives things like a photo of a {} which can then be formated later. That is so cool with format. I nrvrt thought of that. 
        #😉 There are four  possible types of prompts.

        if prompt == 'fixed':
            self.prompt_list = ['a photo of a {}.']
        elif prompt == 'shuffle':
            self.prompt_list = ['a photo of a {}.', 'a photograph of a {}.', 'an image of a {}.', '{}.']
        elif prompt == 'shuffle+':
            self.prompt_list = ['a photo of a {}.', 'a photograph of a {}.', 'an image of a {}.', '{}.',
                                'a cropped photo of a {}.', 'a good photo of a {}.', 'a photo of one {}.',
                                'a bad photo of a {}.', 'a photo of the {}.']

        elif prompt == 'shuffle_clip':
            from models.clip_prompts import imagenet_templates #😉 THis does not seem to exist anymore in the code  
            self.prompt_list = imagenet_templates

        #🙋‍♂️ Not exactly sure what is going on here. are we trying to process the final output of the conditional prompt(visual or text), 
        if process_cond is not None:
            if process_cond == 'clamp' or process_cond[0] == 'clamp': #🙋‍♂️ We can clamp, I am guessing put in a range, but why is it that there can be a list of process_cnditions?

                val = process_cond[1] if type(process_cond) in {list, tuple} else 0.2 #🙋‍♂️ is this a value for the clamping? 👌 Yes.

                def clamp_vec(x): #torch 
                    return torch.clamp(x, -val, val)

                self.process_cond = clamp_vec

            elif process_cond.endswith('.pth'): #😉 If instead it is a saved value, 
                
                shift = torch.load(process_cond) #😉 then jst load the value, and call t shift, then add the shift?? Why add a shift? I guess it is a shift for the whole cond. We already know by how much we want to shift? 
                def add_shift(x):
                    return x + shift.to(x.device)

                self.process_cond = add_shift
                #😉 process_cond seems to be used nowhere.

        import pickle
        precomp = pickle.load(open('precomputed_prompt_vectors.pickle', 'rb'))
        self.precomputed_prompts = {k: torch.from_numpy(v) for k, v in precomp.items()} #😉 If we have already precomputed the prompt vectors.
        # 🙋‍♂️ Why are there a lot of places with code about precomputing/.👌 I am confident it is because of speed. When training, it makes no sense to compute all the values for a dataset all over again, when it is possible to just save this values. 
        

    def forward(self, inp_image, conditional=None, return_features=False, mask=None): #😉 Input image, conditional currently none, but it would be supplied with the forward. return_features, false. I remember mask is not supported yet. 

        assert type(return_features) == bool #😉 Sure. Maybe forces you to have right order. I should have more assert statements in my code. 

        # inp_image = inp_image.to(self.model.positional_embedding.device)

        if mask is not None:
            raise ValueError('mask not supported')

        # x_inp = normalize(inp_image)
        x_inp = inp_image

        bs, dev = inp_image.shape[0], x_inp.device #😉batchsize

        inp_image_size = inp_image.shape[2:] #😉 size of image, 

        cond = self.get_cond_vec(conditional, bs) #😉Comditional is an input. It passes through visual forward if this is an image, or passes throigh 

        visual_q, activations, _ = self.visual_forward(x_inp, extract_layers=[0] + list(self.extract_layers)) #😉 the 0th layer is also extracted, so likely four. No CLIP training. which is the reason for the torch.no_grad.

        activation1 = activations[0] #😉 CLS encoding solution
        activations = activations[1:] #😉 other encodign solution.

        a = None #😉 initialization.of a, we are about to finally go through with the part that needs to train. 
        
        for i, (activation, block, reduce) in enumerate(zip(activations[::-1], self.blocks, self.reduces)): #😉 They got rid of the CLS token for when training the decoder. 
            
            if a is not None:
                a = reduce(activation) + a #🙋‍♂️ . This is where the skip  connections are added. The skip connections are used with a nn.linear layer.   
            else:
                a = reduce(activation) #😉 For the first layer, the entry into the decoder, there is no other activation. No slip connection. 

            if i == self.cond_layer: #😉 If i is the cond_layer, which is zero normally, 
                if self.reduce_cond is not None:
                    cond = self.reduce_cond(cond) #😉reduce cond if needed, 
                
                a = self.film_mul(cond) * a + self.film_add(cond) #😉 oerform the film on a. But this is before any of the transformers are involved. 

            a = block(a) #😉 pass through block (encoder).
            #😉 This is done for all the encoders in block first. Before we then move to extra blocks. 

        for block in self.extra_blocks:
            a = a + block(a) #😉 A resudyal connection for all the extra blocsk. They are done entirely after the first transformers are done. Where are the outputs of the previous layer added?

        a = a[1:].permute(1, 2, 0) # rm cls token and -> BS, Feats, Tokens
        #😉 cls token is been removed again, and the B, features, tokens becomes the size. 

        size = int(math.sqrt(a.shape[2])) #😉 the token size, an int of the sqrt is made. if I remember it is sqrt 784 which should be around 27ish.  

        a = a.view(bs, a.shape[1], size, size) #😉 we reshape to size, size. 🙋‍♂️How sure are we that this would work?

        if self.trans_conv is not None:
            a = self.trans_conv(a) #🙋‍♂️ A transpose convolution, likely to increase the size of the image, to a given size. 

        if self.upsample_proj is not None: #😉 Redice it to a 1 channel solution. I guess for binary segmentation. 
            a = self.upsample_proj(a)
            a = nnf.interpolate(a, x_inp.shape[2:], mode='bilinear') #😉 Interpolation to get desired shape. 

        a = nnf.interpolate(a, inp_image_size)

        if return_features:
            return a, visual_q, cond, [activation1] + activations
        else:
            return a,
