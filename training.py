import torch
import inspect
import json
import yaml
import math
import os
import sys
import wandb
import time

from general_utils import log

import numpy as np
from functools import partial
from os.path import expanduser, join, isfile, basename

from torch.cuda.amp import autocast, GradScaler #😉used all the time.
from contextlib import nullcontext 
from torch.utils.data import DataLoader

from general_utils import TrainingLogger, get_attribute, filter_args, log, training_config_from_cli_args


def validate(model, dataset, config): #😉the validate loop is here. I also think I should save my validation, then write a test script from now on. Save metrics like the IoU, thhe correct classes etc in my validation. 
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False) #😉 Anonymous dataset given to make dataloader,

    metric_class, use_metric = config.val_metric_class, config.use_val_metric #😉 A config dictionaty we get metric_class and use_metric.
    loss_fn = get_attribute(config.loss) #😉 Gets the loss_function function from its name.

    model.eval() 
    model.cuda() #😉 Use model.to("device"), we like consistency.

    if metric_class is not None:
        metric = get_attribute(metric_class)() #😉 Use attribute to get the mettrics from a config. I think this is the way configs are used. they get attributes. 

    with torch.no_grad(): #😉 No grad, not training.  This is the validation script.

        i, losses = 0, [] #😉 Nice, initialize them together. 
        for data_x, data_y in data_loader: #😉 Data loader output. It is just an image and text. 

            data_x = [x.cuda() if isinstance(x, torch.Tensor) else x for x in data_x] 
            data_y = [x.cuda() if isinstance(x, torch.Tensor) else x for x in data_y]

            prompts = model.sample_prompts(data_x[1], prompt_list=('a photo of a {}',)) #😉 First get the prompt of the image, I guess data_x, is a list containing the information and the corresponding image.  
            pred, visual_q, _, _  = model(data_x[0], prompts, return_features=True) #😉 Validate model with image. Visual q is output of clip visual encoder, pred is final prediction.

            if metric_class is not None: 
                metric.add([pred], data_y) #😉 So similar to my class loggers for each loss value and metrics, They use a metric clss.

            # pred = model(data_x[0], prompts)
            # loss = loss_fn(pred[0], data_y[0])
            loss = loss_fn(pred, data_y[0]) #😉 Loss functions 
            losses += [float(loss)] #😉 loss to float32.

            i += 1 #😉 Increase i 

            if config.val_max_iterations is not None and i > config.val_max_iterations: 
                break

    if use_metric is None:
        return np.mean(losses), {}, False #😉 return values
    else:
        metric_scores = {m: s for m, s in zip(metric.names(), metric.value())} if metric is not None else {}
        return np.mean(losses), metric_scores, True #😉 return osses and metric_scores.


def main():
    config = training_config_from_cli_args() #😉it is a config style work, and we get config from cli it has to do with yaml files too. I need to use it and understand it later.

    #😉 logging while training
    #2 wandb
    experiment = wandb.init(project='clipseg_phrasecut', resume='allow', anonymous='must')
    experiment.config.update(
        config
    )  

    val_interval, best_val_loss, best_val_score = config.val_interval, float('inf'), float('-inf') #😉 Intializations. 
    
    model_cls = get_attribute(config.model) #😉 model class from attribute
    _, model_args, _ = filter_args(config, inspect.signature(model_cls).parameters)#😉model args are filtered.
    model = model_cls(**model_args).cuda() #😉 model class from  config.

    if config.pretrained:
        print("using pretrained weights")
        model.load_state_dict(torch.load(config.pretrained_path), strict=False)

    dataset_cls = get_attribute(config.dataset)  #😉 dataset args from config
    _, dataset_args, _ = filter_args(config, inspect.signature(dataset_cls).parameters) #😉 dataset args filtered

    dataset = dataset_cls(**dataset_args) #😉 dataset from cls

    print(f'Train dataset {dataset.__class__.__name__} (length: {len(dataset)})') #😉logs train dataset

    if val_interval is not None: #😉 a way to control if vaidation would be done.
        dataset_val_args = {k[4:]: v for k,v in config.items() if k.startswith('val_') and k != 'val_interval'} #😉 validation args. 
        _, dataset_val_args, _ = filter_args(dataset_val_args, inspect.signature(dataset_cls).parameters) #😉 validation args. 
        print('val args', {**dataset_args, **{'split': 'val', 'aug': 0}, **dataset_val_args})

        dataset_val = dataset_cls(**{**dataset_args, **{'split': 'val', 'aug': 0}, **dataset_val_args}) #😉 the val dataset.

    # optimizer
    opt_cls = get_attribute(config.optimizer) #😉 optimizer class.
    if config.optimize == 'torch.optim.SGD':
        opt_args = {'momentum': config.momentum if 'momentum' in config else 0} 
    else:
        opt_args = {}
    opt = opt_cls(model.parameters(), lr=config.lr, **opt_args) #😉 optimizer object

    batch_size, max_iterations = config.batch_size, config.max_iterations #😉bs, iterations from config. 

    loss_fn = get_attribute(config.loss) #😉 an actual function from config.

    if config.amp:
        print('Using AMP')
        autocast_fn = autocast 
        scaler = GradScaler()
    else:
        autocast_fn, scaler = nullcontext, None #😉null context, nice. Need to read python docs more.


    save_only_trainable = True #🙋‍♂️ Not sure what this is. 👌 It is used s a flag for the save function. if true, we do not save all the model, but only the trainable parts of the model, whih wiuld be the and any custom trainable parts. 
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=2) #🛑 remember to change back.

    epochs = math.ceil(max_iterations/ (len(dataset)/batch_size)) # approximately, 4 epoch. 20k/ 80,000/16. Original paper used batchsize 64, so it would be approximately 20 epochs. Why has my epoch value not increased? I should print epochs
    print(f"amount of epochs is {epochs}")
    print(f"len(dataset) is {len(dataset)}")

    # disable config when hyperparam. opt. to avoid writing logs.
    tracker_config = config if not config.hyperparameter_optimization else None

    logger = TrainingLogger(log_dir=config.name, model=model, config=tracker_config)
     
    i = 0
    for epoch in range(epochs): #😉 Trains Starts
        end = time.time()
        for data_x, data_y in data_loader:

            # between caption and output feature.
            # 1. Sample random captions
            # 2. Check alignment with CLIP

            # randomly mix text and visual support conditionals
            if config.mix: #😉 A form of data augmentation. 🙋‍♂️ is this phrasecut++? 👌THis is for when we are mixing the text and image conditionals. Not dat augmentation per se but ya, it is close. 

                assert config.mask.startswith('text_and') #😉 Assert statements good.

                # data_x[1] = text label
                prompts = model.sample_prompts(data_x[1]) #😉 remeber data_x[0] is img, data_x[1] is the phrase prompt

                # model.clip_model()

                text_cond = model.compute_conditional(prompts) #😉 Text conditional to train. Can be class names, heck it should be class names or a form of the class names. 
                if model.__class__.__name__ == 'CLIPDensePredTMasked': #😉 DensePredTMasked also requires a msk input.
                    # when mask=='separate'
                    visual_s_cond, _, _ = model.visual_forward_masked(data_x[2].cuda(), data_x[3].cuda()) #😉 you need to be careful with the yaml file. x[2] is support image, x[3] is support mask. This is oneshot done in the model.
                else:
                    # data_x[2] = visual prompt
                    visual_s_cond, _, _ = model.visual_forward(data_x[2].cuda()) #😉 This is the already input masked support image and mask 

                max_txt = config.mix_text_max if config.mix_text_max is not None else 1  #🙋‍♂️ we are mixing the vecotrs of txt and visuals. I am guessing this is a ratio of how much text and visuals are to be mixed
                batch_size = text_cond.shape[0] #😉 okay, the amount of text_conditionals is the same as the batch. Each image, has its own text conditionals.

                # sample weights for each element in batch #🙋‍♂️ Sample weights? where are sample weights needed and why?  
                text_weights = torch.distributions.Uniform(config.mix_text_min, max_txt).sample((batch_size,))[:, None] #🙋‍♂️ I am not sure what is going on here. #👌 We have a uniform sample between mintext value and maxtext value and then We get n samples where n is the batch size. We are trying to mix the text and the image, and we want various amounts of mix. 
                text_weights = text_weights.cuda() #😉 The weights to cuda. 

                #😉 Dataset dependent code. 
                if dataset.__class__.__name__ == 'PhraseCut': #😉If we are using phrasecut, 
                    # give full weight to text where support_image is invalid
                    visual_is_valid = data_x[4] if model.__class__.__name__ == 'CLIPDensePredTMasked' else data_x[3] #🙋‍♂️ Not sure, I need to check what is given by the dataset. But it seems we can have as much as 4 outputs from the dataset. Image is 3, but I am not sure what 4 is. Maybe it is a masked image? 
                    text_weights = torch.max(text_weights[:,0], 1 - visual_is_valid.float().cuda()).unsqueeze(1) #😉 text weghts an either be the rrent amount, or the 1-visual is valid for phrasecut dataset.

                cond = text_cond * text_weights + visual_s_cond * (1 - text_weights) #😉 Finally, the conditional becomes a vetor that is based on the text wieghts with the text conditional and the visual weights and the visual conditional.

            else: 
                # no mix #😉Yep, no mix.
                
                if model.__class__.__name__ == 'CLIPDensePredTMasked': #😉if we still have masks for one shot segmentation, 
                    # compute conditional vector using CLIP masking
                    with autocast_fn(): #😉 Autocast context.
                        assert config.mask == 'separate' #🙋‍♂️ What is separate, what are the possible values for mask?
                        cond, _, _ = model.visual_forward_masked(data_x[1].cuda(), data_x[2].cuda()) #😉The conditional vector, is computed with CLIP masking in mind. 🙋‍♂️I thought image was x[3] now it seems it is x[2] again. I need to check the dataset then go through all the code again. 
                else:
                    cond = data_x[1] #🙋‍♂️ conditional data is 1 now, I am so confised. No visual forward, no sample prompts, cond is just is. 
                    if isinstance(cond, torch.Tensor):
                        cond = cond.cuda()

            visual_q = None #😉 Visual q is the output from the CLIP visual encoder. 

            pred, visual_q, _, _  = model(data_x[0].cuda(), cond, return_features=True) #😉 Pred is the predictions. 🙋‍♂️Why is is data_x[0] the input?

            loss = loss_fn(pred, data_y[0].cuda())  #😉 Calculate loss.

            if torch.isnan(loss) or torch.isinf(loss): #😉 Loss errors. 🙋‍♂️What could cause these? 
                # skip if loss is nan
                print('encountered a nan or an -inf, stop this current loop')
                continue

            opt.zero_grad() #🙋‍♂️no grad scaler? 👌 Scaler is used when you wanna call loss.backwards, step and update on the loss. 
            loss.backward()
            opt.step()    
            
            #logging
            step_duration = time.time() - end
            end = time.time()
            memory_info = torch.cuda.mem_get_info(torch.device("cuda:0"))
         
            i += 1
            print(f"step is {i}")
            loss_value = loss.item()

            experiment.log({
                            'step': i,
                            'loss': loss_value,
                            'epoch': epoch,
                            'time_per_step': step_duration,
                            'free_gpu_memory': memory_info[0],
                            'total_available_memory': memory_info[1],
                        }) 

            if i >= max_iterations: #😉 If u is greater than this max_iterations. 

                if not isfile(join(logger.base_path, 'weights.pth')): #😉 If we do not have this file,
                    # only write if no weights were already written
                    logger.save_weights(only_trainable=save_only_trainable) #😉 save weights. Only trainiable. 
                
                sys.exit(0)

                
            if config.save_checkpoint_iterations and i % config.save_checkpoint_freq == 0: 
                logger.save_weights(only_trainable=save_only_trainable, weight_file=f'weights_{i}.pth') #😉 Save only trainable variables for all weight variables. 

            
        print('epoch complete') #😉 Nice. Lets try.


if __name__ == '__main__':
    main()