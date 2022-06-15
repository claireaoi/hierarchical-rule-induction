
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import random

def sample_gumbel(shape, scale, eps, fix_gumbel):
    """
    Gumbel sample
    """
    U = scale*torch.rand(shape)
    U = -torch.log(-torch.log(U+eps)+eps)
    return U

def gumbel_softmax_sample(logits, tau, scale, fix_gumbel, eps=1e-20, use_gpu=False):
    """
    Gumbel softmax
    """
    samp = sample_gumbel(logits.size(), scale, eps, fix_gumbel)
    if use_gpu:
        samp = samp.cuda()
    y = logits + samp #add noise 
    out = torch.nn.Softmax(dim=0)(y/tau)
    return out


#------
def fuzzy_and(v1, v2, mode):
    """
    Different fuzzy and. Here vector are assumed scalar.
    """
    if mode=="min":
        return torch.min(v1, v2)
    elif mode=="product":
        return v1*v2
    elif mode=="norm_product":
        return v1*v2/ (v1+v2-v1*v2+1e-20)#needed 1e-20?
    elif mode=="lukas":
        return torch.max(0, v1+v2-1)
    else:
        raise NotImplementedError

#------
def fuzzy_and_vct(val1, val2, mode):
    """
    Different fuzzy and.
    Inputs:
        v1 size rpcc, c being number constant
        v2 size rpcc
    Outputs:
        fuzzy and: size ppccc
    """
    r,p,c,cc=val1.shape

    if mode=="min":
        v1=val1.unsqueeze(3).repeat((1,1,1,c, 1)).unsqueeze(2).repeat((1,1,p,1, 1,1))#rpxz, insert y and q...
        v2=val2.transpose(2,3).unsqueeze(2).repeat((1,1,c, 1,1)).unsqueeze(1).repeat((1,p,1,1, 1,1))##qzy
        out=torch.min(v1, v2)
        
    elif mode=="product":
        out=torch.einsum('rpxz,rqzy->rpqxyz', val1, val2)
    else:
        raise NotImplementedError
    assert list(out.shape)==[r,p,p,c,c,c]
    return out

  


def get_unifs(rules, embeddings, args=None, mask=None, temperature=None, gumbel_noise=None):
    """
    Compute unifications score of (soft) rules with embeddings (all:background+ symbolic+soft)

    Inputs:
        embeddings: tensor size (p,f) where p number of predicates, f is feature dim, (and currently p=f)
        rules: tensor size (r, num_body*f) where f is feature dim,  r nb rules, and num_body may be 2 or 3 depending on model considered
    Outputs:        
    """
    # -- 00---init
    num_rules, d=rules.shape
    num_predicates, num_feat= embeddings.shape

    if temperature is None:
        print("take generic temp argument but better specify  temperature")
        temperature=args.temperature_end
    if gumbel_noise is None:
        print("Take generic noise argument but better specify gumbel noise")
        gumbel_noise=args.gumbel_noise
        
    # 0: ---add True and False
    if args.add_p0:
        num_predicates += 2
        num_feat += 2
        # NOTE: add False which is (0,1,0...) in second position
        row = torch.zeros(1, embeddings.size(1))
        col = torch.zeros(embeddings.size(0)+1, 1)
        col[0][0] = 1
        if args.use_gpu:
            row = row.cuda()
            col = col.cuda()
        embeddings = torch.cat((row, embeddings), 0)
        embeddings = torch.cat((col, embeddings), 1)
        # NOTE: add True which is (1,0,...) in first position
        row_F = torch.zeros(1, embeddings.size(1))
        col_F = torch.zeros(embeddings.size(0)+1, 1)
        col_F[0][0]=1
        if args.use_gpu:
            row_F = row_F.cuda()
            col_F = col_F.cuda()
        embeddings = torch.cat((row_F, embeddings), 0)
        embeddings = torch.cat((col_F, embeddings), 1)
    
    assert d % num_feat == 0
    num_body=d//num_feat

    # -- 1---prepare rules and embedding in good format for computation below
    if num_body == 2:
        rules_aux = torch.cat(
            (rules[:, : num_feat],
                rules[:, num_feat: 2 * num_feat]),
            0)
    elif num_body == 3:
        rules_aux = torch.cat(
            (rules[:, : num_feat],
                rules[:, num_feat: 2 * num_feat],
                rules[:, 2 * num_feat: 3 * num_feat]),
            0)
    else:
        raise NotImplementedError

    rules_aux = rules_aux.repeat(num_predicates, 1)#size (p*3*r, f)
    
    embeddings_aux = embeddings.repeat(1, num_rules * num_body).view(-1, num_feat)
    # -2-- compute similarity score between predicates and rules body
    if args.similarity == "cosine":
        sim = F.cosine_similarity(embeddings_aux, rules_aux).view(num_predicates, num_body, num_rules)
    elif args.similarity == "L1":
        sim = torch.linalg.norm(embeddings_aux-rules_aux, ord=1, dim=1).view(num_predicates, num_body, num_rules)
    elif args.similarity == "L2":
        sim = torch.linalg.norm(embeddings_aux-rules_aux, ord=2, dim=1).view(num_predicates, num_body, num_rules)
    elif args.similarity == "scalar_product":
        sim=torch.einsum('bd,bd->b', embeddings_aux, rules_aux).view(num_predicates, num_body, num_rules)
    else:
        raise NotImplementedError

    #---3---treatment negative similarity score:
    # NOTE: For now, treat negative as minus similarity score as 0
    #NOTE: relu, equivalent to clamping negative scores
    if args.clamp=="sim":
        sim=nn.ReLU()(sim) 

    #--4---- HIERARCHICAL mask here in case hierarchical model
    if args.softmax == "softmax" or args.softmax == "gumbel":
        cancel_out=-10000
    else:
        cancel_out=0
    if mask is not None:
        mask=mask.double()
        sim[mask==0] = cancel_out

    #-5-----tgt mask: other rule body not being matched to tgt
    #NOTE: tgt shall not being matched to itself too
    #NOTE: this would not work for even/odd in non unified variants!
    if args.unified_templates:
        sim[-1,:,:]= cancel_out*torch.ones(num_body, num_rules)

    # --3-- possibly apply softmax to normalise or gumbel softmax to normalise + explore
    if args.softmax == "softmax":
        if args.use_gpu:
            temperature = torch.tensor(temperature).cuda()
        unifs = nn.Softmax(dim=0)(sim / temperature).view(-1)
    elif args.softmax == "gumbel":
        unifs = gumbel_softmax_sample(
            sim, temperature, gumbel_noise, use_gpu=args.use_gpu, fix_gumbel=args.fix_gumbel).view(-1)
    elif args.softmax == "none":
        unifs = sim
        #TODO: Could clamp simiarlity instead of parameters!
    else:
        raise NotImplementedError

    if not ((torch.max(unifs.view(-1)).item()<=1) and (torch.min(unifs.view(-1)).item()>=0)):
        print("ERROR UNIFS not in BOUNDARY", torch.max(unifs.view(-1)).item(), torch.min(unifs.view(-1)).item())
        pass

    return unifs.view(num_predicates, -1)


def fuzzy_or(v1, v2, mode):
    """
    Fuzzy Or
    """
    if mode=="max":
        return torch.max(v1, v2)
    elif mode=="prodminus":
        return v1+v2-v1*v2
    elif mode=="lukas":
        return torch.min(1, v1+v2)
    else:
        raise NotImplementedError


def merge(old, new, mode="sum"):
    if mode=='max':
        return torch.max(old, new.float())
    elif mode=='sum':
        return old + new
    else:
        raise NotImplementedError()

def pool(tsr, mode="sum", dim=[0]):
    if mode=='max':
        if dim==[0,1]:#temporary fix
            return torch.max(torch.max(tsr, dim=0)[0], dim=0)[0]
        elif len(dim)==1:
            return torch.max(tsr, dim=dim[0])[0]
        else:
            raise NotImplementedError
    elif mode=='sum':
        if dim==[0,1]:
            return torch.sum(torch.sum(tsr, dim=1),dim=0)
        elif len(dim)==1:
            return torch.sum(tsr, dim=dim[0])
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError()

def map_rules_to_pred(num_aux, idx_aux, num_rules, pred_two_rules=None):
    """
    Map rules to predicates depending on ones which are associated to two rules.
    Outputs: 
        RULES_TO_PREDICATES[rule]=predicate (index of intensional predicate)
        PREDICATES_TO_RULES[rule]=[list index rules associated to this predicate]

    """

    RULES_TO_PREDICATES, PREDICATES_TO_RULES=[], []
    count=0
    for pred in range(num_aux):
        if idx_aux[pred] in pred_two_rules: #of idx_auxiliary[pred] for actual index pred...
            RULES_TO_PREDICATES.append(pred)
            RULES_TO_PREDICATES.append(pred)
            PREDICATES_TO_RULES.append([count, count+1])
            count+=2
        else:
            RULES_TO_PREDICATES.append(pred)
            PREDICATES_TO_RULES.append([count])
            count+=1

    return RULES_TO_PREDICATES, PREDICATES_TO_RULES



#---------------
def depth_sorted_idx(depth_predicates):
    sorted_idx=[]#double array
    max_depth=np.max(depth_predicates)
    for depth in range(max_depth+1):
        current_depth_idx=[i for i in range(len(depth_predicates)) if depth_predicates[i]==depth]
        sorted_idx.append(current_depth_idx)
    return sorted_idx


#----------------- top k and top p sampling

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ 
    Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    """
    assert logits.dim() == 1  # batch size 1 for now 
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits
    
def top_k_top_p_sampling(logits, top_p=0.9, top_k=0, temperature=1.0):
    """
    Top K Top P Sampling
    NOTE: here logits assumed dimension 1
    """
    filtered_logits = top_k_top_p_filtering(torch.tensor(logits)/temperature, top_k=top_k, top_p=top_p)
    #  Sample from the filtered distribution
    probabilities = F.softmax(filtered_logits, dim=-1)
    sample = torch.multinomial(probabilities, 1)#here is indice...
    return sample


def equal(val_vct, valuation):
    equal=True
    num_aux_pred=val_vct.size(0)
    num_constants=val_vct.size(1)
    for predicate in range(num_aux_pred):
        val=valuation[predicate]
        if list(val.size())==[]:#case  p0, p1 beginning
            val_=val.repeat((num_constants, num_constants))
        elif val.size()[1] == 1:#unary
            val_=val.repeat((1, num_constants))
        else:
            val_=val
        equal=equal and torch.equal(val_vct[predicate,:,:], val_)
    return equal


def iterline(fpath):
    with open(fpath) as f:

        for line in f:

            line = line.strip()
            if line == '':
                continue

            yield line


def print_dict(dictt):
    for k, v in dictt.items():
        print(f'{k}: {v}')
