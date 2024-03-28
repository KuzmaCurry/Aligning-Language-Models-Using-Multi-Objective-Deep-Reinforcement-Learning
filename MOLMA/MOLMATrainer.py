import torch
from torch import nn
import torch.nn.functional as F
import random
import copy
import numpy as np
import time
from sklearn.model_selection import train_test_split
import gc
import GPUtil
from torch.cuda.amp import autocfast as autocast
from torch.optim.lr_scheduler import LambdaLR
import torch.distributed as dist
from transformers.trainer_pt_utils import IterableDatasetShard
from datasets import load_dataset
import os
import argparse
import json
from transformers import PhiModel, AutoTokenizer, AutoModelForCausalLM, PhiPreTrainedModel, AutoConfig, PhiForSequenceClassification
from trl import AutoModelForCausalLMWithValueHead
from torch.autograd import Variable
import trl
from trl.trainer.utils import AdaptiveKLController
from accelerate import Accelerator
import transformers
from solver import ProcrustesSolver

BATCH_SIZE = 8
EPOCHS = 1
gamma = 1
lamda = 0.95
epsilon = 0.2
k_epochs = 1
log_interval = 5

SEED = 30
output_dir='./MOLMA'
data_dir='query_only_MODRL_dataset.json' # queries are randomly selected from the Cleaned Alpaca and the Anthropic Harmless datasets

def main():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.padding_side = "left"
    
    model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    ref_model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    critic_model = AutoModelForCausalLMWithValueHead.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    reward_model = PhiForSequenceClassification.from_pretrained('./RMhelp') # directory path of the RMhelp
    reward_model2 = PhiForSequenceClassification.from_pretrained('./RMsafe') # directory path of the RMsafe
    with open(data_dir, 'r') as file:
        dataset = json.load(file)
    random.shuffle(dataset)
    moppo_dataset = dataset[:6400] # 100 iterations
    dataloader = torch.utils.data.DataLoader(
        dataset=moppo_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    MOLMA = MOLMATrain(model, ref_model, critic_model, reward_model, reward_model2, tokenizer, dataloader)
    MOLMA.train()
    
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class ModelWrapper(nn.Module):
    def __init__(self, policy_model, critic_model):
        super().__init__()
        self.policy_model = policy_model
        self.critic_model = critic_model

    def forward(self, input_ids, *args, **kwargs):
        return self.policy_model(input_ids, *args, **kwargs), self.critic_model(input_ids, *args, **kwargs)

# class AdaptiveKLController:
#     def __init__(self, init_kl_coef, target, horizon):
#         self.value = init_kl_coef
#         self.target = target
#         self.horizon = horizon
#     def update(self, current, n_steps):
#         target = self.target
#         proportional_error = torch.clamp(current / target - 1, -0.2, 0.2)
#         mult = 1 + proportional_error * n_steps / self.horizon
#         self.value *= mult
           
def collate_fn(batch):
    max_len = max([len(x) for x in batch])
    query = []
    for data in batch:
        curr_len = len(data)
        query.append([-1] * (max_len - curr_len) + data)
    return torch.tensor(query)

            
class MOLMATrain:
    def __init__(self, policy_model, ref_model, critic_model, reward_model, reward_model2, tokenizer, dataloader):
        self.accelerator = Accelerator()
        self.model = ModelWrapper(policy_model=policy_model, critic_model=critic_model)
        self.ref_model = ref_model
        self.reward_model = reward_model
        self.reward_model2 = reward_model2
        self.model, self.ref_model, self.reward_model, self.reward_model2 = self.accelerator.prepare(self.model, self.ref_model, self.reward_model, self.reward_model2)
        self.policy_model = self.accelerator.unwrap_model(self.model).policy_model
        self.critic_model = self.accelerator.unwrap_model(self.model).critic_model
        self.tokenizer = tokenizer
        self.dataloader = dataloader
        self.MSE = nn.MSELoss()
        # self.klc = AdaptiveKLController(0.05, 6, 10000)
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.000001, betas=(0.9, 0.95), eps=0.00001, weight_decay=0.1)
        (
            self.optimizer,
            self.dataloader,
        ) = self.accelerator.prepare(
            self.optimizer, self.dataloader
        )
        
    def generate(self, queries):
        responses = []
        for query in queries:
            input_ids = query.unsqueeze(0).to(self.accelerator.device)
            for i in range(128):
                with torch.no_grad():
                    outputs, _ = self.model(input_ids)
                logits = outputs.logits[:,-1,:].unsqueeze(1)
                output = torch.argmax(logits, dim=-1)
                input_ids = torch.cat((input_ids, output), dim=-1)
            generate_ids = input_ids.squeeze(0).tolist()
            if self.tokenizer.eos_token_id in generate_ids:
                idx = generate_ids.index(self.tokenizer.eos_token_id)
                generate_ids = generate_ids[:idx+1]
            response = torch.tensor(generate_ids[query.size(-1):]) # [response_len]
            responses.append(response)
        return responses
        
    def compute_values_logprobs(self, query_batches, response_batches): # [batch_size, response_len] 
        all_values = []
        all_logprobs = []
        all_ref_logprobs = []
        all_input = []
        for q, r in zip(query_batches, response_batches):
            q = q.to(self.accelerator.device)
            r = r.to(self.accelerator.device)
            start = q.size(-1) - 1
            input = torch.cat((q,r), dim=-1).unsqueeze(0)
            all_input.append(input)
            input = input.to(self.accelerator.device)
            with torch.no_grad():
                policy_output, critic_output = self.model(input)
                logits = policy_output.logits
                values = critic_output[-1]
                ref_logits = self.ref_model(input).logits # [batch_size, q+res_length, vocab_size]
            torch.cuda.empty_cache()
            logprobs = self.logprobs_from_logits(logits, input)
            ref_logprobs = self.logprobs_from_logits(ref_logits, input)           
            all_values.append(values[:, start:-1])
            all_logprobs.append(logprobs[:, start:])
            all_ref_logprobs.append(ref_logprobs[:, start:]) # list of [batch_size, response_len]
        return all_input, all_values, all_logprobs, all_ref_logprobs
        
    
    def compute_vpreds_logits(self, query_batch, input_batch):
        start = query_batch.size(-1) - 1
        input_batch  = input_batch.to(self.accelerator.device)
        policy_output, critic_output = self.model(input_batch)
        logits = policy_output.logits
        vpreds = critic_output[-1]
        logprobs = self.logprobs_from_logits(logits, input_batch)
        return logits[:, start:-1, :], vpreds[:, start:-1], logprobs[:, start:]
        
    
    def logprobs_from_logits(self, logits, input_batch):
        logp = F.log_softmax(logits[:, :-1, :], dim=-1)
        logpy = torch.gather(logp, 2, input_batch[:, 1:].unsqueeze(2)).squeeze(-1)
        return logpy # [minibatch_size, q+res_length]
    
    def train(self):  
        start = time.time()
        for epoch in range(1, EPOCHS+1): 
            step = 0
            for query in self.dataloader:
                step += 1
                queries = [q[torch.numel(q[q<0]):] for q in query]
                responses = self.generate(queries)
                self.step(queries, responses, step)
                gc.collect()
                torch.cuda.empty_cache()
        elapsed = time.time() - start
        self.policy_model.save_pretrained(
            output_dir,
            is_main_process=self.accelerator.is_main_process,
            save_function=self.accelerator.save,
            state_dict=self.policy_model.state_dict(),
        )
        if self.accelerator.is_main_process:
            print(round(elapsed, 5))
                 
    
    def step(self, queries, responses, step):
        length = len(queries)
        all_input, all_values, all_logprobs, all_ref_logprobs = self.compute_values_logprobs(queries, responses)
        rewards1, scores1 = self.compute_reward(all_input, all_logprobs, all_ref_logprobs, 'help')
        rewards2, scores2 = self.compute_reward(all_input, all_logprobs, all_ref_logprobs, 'safe')
        for k in range(k_epochs):
            for i in range(length):
                grads = []
                
                q = queries[i]
                logits, vpreds, logprobs = self.compute_vpreds_logits(q, all_input[i]) 
                loss1 = self.loss(all_ref_logprobs[i], all_values[i], rewards1[i], logits, vpreds, logprobs, 'help')
                
                self.optimizer.zero_grad()
                for p in self.model.parameters():
                    if p.grad is not None:
                        p.grad.data.zero_()
                loss1.backward()
                grad = torch.cat([p.grad.flatten().clone() if p.grad is not None else torch.zeros_like(p).flatten() for p in self.model.parameters()])
                grads.append(grad)

                logits, vpreds, logprobs = self.compute_vpreds_logits(q, all_input[i])
                loss2 = self.loss(all_ref_logprobs[i], all_values[i], rewards2[i], logits, vpreds, logprobs, 'safe')
                for p in self.model.parameters():
                    if p.grad is not None:
                        p.grad.data.zero_()
                loss2.backward()
                grad = torch.cat([p.grad.flatten().clone() if p.grad is not None else torch.zeros_like(p).flatten() for p in self.model.parameters()])
                grads.append(grad)
                for p in self.model.parameters():
                    if p.grad is not None:
                        p.grad.data.zero_()
                
                grads = torch.stack(grads, dim=0)
                grads, weights, singulars = ProcrustesSolver.apply(grads.T.unsqueeze(0))
                grad, weights = grads[0].sum(-1), weights.sum(-1)
                self.set_shared_grad(self.model.parameters(), grad)
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
            if self.accelerator.is_main_process and step%log_interval == 0:
                print(loss1, loss2)
                print("record: ", round(scores1, 4), round(scores2, 4))
        # self.klc.update(torch.mean(current_kl), BATCH_SIZE*self.accelerator.num_processes)
    
    def set_shared_grad(self, shared_params, grad_vec):
        offset = 0
        for p in shared_params:
            if p.grad is None:
                continue
            _offset = offset + p.grad.shape.numel()
            p.grad.data = grad_vec[offset:_offset].view_as(p.grad)
            offset = _offset
    
    def compute_reward(self, all_input, logprobs, ref_logprobs, label):
        scores = []
        rewards = []
        for input in all_input:
            input_ids = input.to(self.accelerator.device)
            input = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)[0] # avoid unk tokens to make reward scores accurate
            input_ids = self.tokenizer(input.strip()+'\n'+self.tokenizer.eos_token, add_special_tokens=False, padding=False, return_tensors='pt').input_ids
            with torch.no_grad():
                if label == 'help':
                    outputs = self.reward_model(input_ids)
                else:
                    outputs = self.reward_model2(input_ids)
            score = outputs.logits # [minibatch_size, 1]
            scores.append(score.squeeze(0))
        scores_ = torch.cat(scores) # [BATCH_SIZE]
        scores = trl.core.whiten(torch.logit(scores_, eps=1e-6))
        scores = torch.clamp(scores, -0.4, 0.4)
        length = len(all_input)
        for i in range(length):
            kl = -0.01*(logprobs[i] - ref_logprobs[i]) # [minibatch_size, response_len]
            reward = scores[i] + kl
            rewards.append(reward)
        return rewards, scores_.mean().item()
    
    def loss(self, ref_logprobs, values, rewards, logits, vpreds, logprobs, label):
        advantages_reversed = []
        last = 0
        gen_len = rewards.size(-1)
        for t in reversed(range(gen_len)):
            nextvalue = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = rewards[:, t] + gamma * nextvalue - values[:, t]
            last = delta + gamma * lamda * last
            advantages_reversed.append(last)
        advantages = torch.stack(advantages_reversed[::-1]).transpose(0, 1)
        returns = advantages + values
        
        vf_loss1 = (vpreds - returns) ** 2
        vf_loss = 0.5 * torch.mean(vf_loss1)
        
        sq_loss = torch.mean((logprobs - advantages - ref_logprobs) ** 2) # APA loss
        loss = sq_loss + vf_loss
        return 0.01*loss


if __name__=="__main__":
    set_seed(SEED)
    main()

