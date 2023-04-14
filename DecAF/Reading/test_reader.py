# Copyright (c) Facebook, Inc. and its affiliates.
# Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import transformers
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, SequentialSampler


import src.slurm
import src.util
from src.options import Options
import src.data
import src.evaluation
import src.model
from tqdm import tqdm
import json

class FiDT5(src.model.FiDT5):
    def generate(self, input_ids, attention_mask, max_length, 
                    num_beams=1, num_return_sequences=1, use_cache=False):
        self.encoder.n_passages = input_ids.size(1)
        
        return transformers.T5ForConditionalGeneration.generate(
            self,
            input_ids=input_ids.view(input_ids.size(0), -1),
            attention_mask=attention_mask.view(attention_mask.size(0), -1),
            max_length=max_length,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            use_cache=use_cache,
        )


def write_output(glob_path, output_path):
    files = list(glob_path.glob('*.json'))
    files.sort()
    with open(output_path, 'w') as outfile:
        output_data = {}
        for path in files:
            with open(path, 'r') as f:
                data = json.load(f)
                output_data.update(data)
            path.unlink()
        json.dump(output_data, outfile, indent=4)
    glob_path.rmdir()

def evaluate(model, dataset, dataloader, tokenizer, opt):
    loss, curr_loss = 0.0, 0.0
    model.eval()
    if hasattr(model, "module"):
        model = model.module
    if opt.write_crossattention_scores:
        model.overwrite_forward_crossattention()
        model.reset_score_storage() 
    total = 0
    exactmatch = []
    if opt.write_results:
        write_path = Path(opt.checkpoint_dir) / opt.name / 'test_results'
        fw = open(write_path / ('%d.json'%opt.global_rank), 'w')
        output_results = {}
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader)):
            (idx, _, _, context_ids, context_mask) = batch

            if opt.write_crossattention_scores:
                model.reset_score_storage()

            outputs = model.generate(
                input_ids=context_ids.cuda(),
                attention_mask=context_mask.cuda(),
                max_length=opt.answer_maxlength,
                num_beams=opt.num_beams,
                num_return_sequences=opt.num_beams,
                use_cache=opt.use_cache_eval,
            )
            
            outputs = outputs.reshape(len(idx), -1, outputs.shape[-1])
            
            if opt.write_crossattention_scores:
                crossattention_scores = model.get_crossattention_scores(context_mask.cuda())

            for k, o in enumerate(outputs):
                ans_list = []
                for o_i in o:
                    ans = tokenizer.decode(o_i, skip_special_tokens=True)
                    ans_list.append(ans)
                example = dataset.data[idx[k]]
                if 'answers' in example:
                    score = src.evaluation.ems(ans_list[0], example['answers'])
                    exactmatch.append(score)

                if opt.write_results:
                    # fw.write(str(example['id']) + "\t" + ans + '\n')
                    output_results[example["id"]] = {
                        "question": example["question"],
                        "gold answers": example['answers'],
                        "predicted answers": ans_list
                    }
                if opt.write_crossattention_scores:
                    for j in range(context_ids.size(1)):
                        example['ctxs'][j]['score'] = crossattention_scores[k, j].item()

                total += 1
            if (i + 1) % opt.eval_print_freq == 0:
                log = f'Process rank:{opt.global_rank}, {i+1} / {len(dataloader)}'
                if len(exactmatch) == 0:
                    log += '| no answer to compute scores'
                else:
                    log += f' | average = {np.mean(exactmatch):.3f}'
                logger.warning(log)

    if opt.write_results:
        json.dump(output_results, fw, indent=4)
        fw.close()
    logger.warning(f'Process rank:{opt.global_rank}, total {total} | average = {np.mean(exactmatch):.3f}')
    if opt.is_distributed:
        torch.distributed.barrier()
    score, total = src.util.weighted_average(np.mean(exactmatch), total, opt)
    
    return score, total


if __name__ == "__main__":
    options = Options()
    options.add_reader_options()
    options.add_eval_options()
    options.parser.add_argument('--num_beams', type=int, default=1)
    options.parser.add_argument('--use_cache_eval', action='store_true')
    opt = options.parse()
    src.slurm.init_distributed_mode(opt)
    src.slurm.init_signal_handler()
    opt.train_batch_size = opt.per_gpu_batch_size * max(1, opt.world_size)

    dir_path = Path(opt.checkpoint_dir)/opt.name
    directory_exists = dir_path.exists()
    if opt.is_distributed:
        torch.distributed.barrier()
    dir_path.mkdir(parents=True, exist_ok=True)
    if opt.write_results:
        (dir_path / 'test_results').mkdir(parents=True, exist_ok=True)
    logger = src.util.init_logger(opt.is_main, opt.is_distributed, Path(opt.checkpoint_dir) / opt.name / 'run.log')
    if not directory_exists and opt.is_main:
        options.print_options(opt)

    tokenizer = transformers.T5Tokenizer.from_pretrained('t5-base', return_dict=False)

    collator_function = src.data.Collator(opt.text_maxlength, tokenizer)
    eval_examples = src.data.load_data(
        opt.eval_data, 
        global_rank=opt.global_rank, #use the global rank and world size attibutes to split the eval set on multiple gpus
        world_size=opt.world_size
    )
    eval_dataset = src.data.Dataset(
        eval_examples, 
        opt.n_context, 
    )

    eval_sampler = SequentialSampler(eval_dataset) 
    eval_dataloader = DataLoader(
        eval_dataset, 
        sampler=eval_sampler, 
        batch_size=opt.per_gpu_batch_size,
        num_workers=20, 
        collate_fn=collator_function
    )
    
    model_class = FiDT5
    model = model_class.from_pretrained(opt.model_path)
    model = model.to(opt.device)

    logger.info("Start eval")
    exactmatch, total = evaluate(model, eval_dataset, eval_dataloader, tokenizer, opt)

    logger.info(f'EM {100*exactmatch:.2f}, Total number of example {total}')

    if opt.write_results and opt.is_main:
        glob_path = Path(opt.checkpoint_dir) / opt.name / 'test_results'
        eval_data_name = opt.eval_data.split('/')[-1].replace('.json', '')
        write_path = Path(opt.checkpoint_dir) / opt.name / f'final_output_{eval_data_name}.json'
        write_output(glob_path, write_path) 
    if opt.write_crossattention_scores:
        src.util.save_distributed_dataset(eval_dataset.data, opt)

