import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F

class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        if not isinstance(self.base_transform, list):
            return [self.base_transform(x) for i in range(self.n_views)]
        else:
            return [self.base_transform[i](x) for i in range(self.n_views)]

class SupConLoss(torch.nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR
    From: https://github.com/HobbitLong/SupContrast"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        logits=logits.to(device)

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss



def info_nce_logits(features, n_views=2, temperature=1.0, device='cuda'):

    b_ = 0.5 * int(features.size(0))

    labels = torch.cat([torch.arange(b_) for i in range(n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    logits = logits / temperature
    return logits, labels



def findNeighbors(x,feat_bank,K=3):
    similarity_matrix= torch.matmul(x,feat_bank.T)
    distance_near, idx_near = torch.topk(similarity_matrix, dim=-1, largest=True, k= K + 2)
    distance_near = distance_near[:,2:]
    idx_near = idx_near[:,2:]  # batch x K
    fea_near = feat_bank[idx_near]

    return idx_near, distance_near,fea_near



def info_expanded_negative_logits(features,feat_bank, n_views=2, temperature=1.0, device='cuda',K=3,expanded_K=2):
    
    
    

    features = F.normalize(features, dim=1)
    sim_mat=features@features.T
    
    similarity_matrix=features @ feat_bank.T
    
    #finding neighbors
    idx_near,logit_near,neighbors=findNeighbors(features,feat_bank,K)
   
    logit_expanded=[]
    neighbor_expanded=[]
    #finding expanded neighbors
    for j in range(neighbors.shape[1]):
        fea_near=neighbors[:,j,:]
        idx,logit,nearest=findNeighbors(fea_near,feat_bank,expanded_K)

       
        idx_expanded.append(idx)
        logit_expanded.append(logit)
        neighbor_expanded.append(nearest)
    idx_expanded=torch.stack(idx_expanded,dim=1)
   
    idx_expanded=idx_expanded.reshape(256,K*expanded_K)
    

    logit_expanded=torch.stack(logit_expanded,dim=1)
   
    logit_expanded=logit_expanded.reshape(256,K*expanded_K)
    
    neighbor_expanded=torch.stack(neighbor_expanded,dim=1)
    
    neighbor_expanded=neighbor_expanded.reshape(256,K*expanded_K,256)
    
    #delete the expanded neighbors from the similarity matrix
    similarity_matrix_mask=similarity_matrix.detach().clone()
    similarity_matrix_mask[torch.arange(similarity_matrix_mask.size(0)).unsqueeze(1), idx_near]=-1
    similarity_matrix_mask[torch.arange(similarity_matrix_mask.size(0)).unsqueeze(1), idx_expanded]=-1
   
    expand_no_neighbors=(neighbors.shape[1]*neighbors.shape[1])
    total_neighbors=neighbors.shape[1]+ expand_no_neighbors
    
    similarity_matrix_sort,_=torch.sort(similarity_matrix_mask,descending=True)
    sim_mat_sort,_=torch.sort(sim_mat,descending=True)
    
    similarity_matrix_sort=similarity_matrix_sort[:,1:]
   
    positives=similarity_matrix_sort[:,0:1]
   
    nearest_neighbors=logit_near
   
    expanded_neighbors=logit_expanded
   
    nn=2+K
    negatives=sim_mat_sort[:,nn:]
   
    
    logits = torch.cat([positives,nearest_neighbors,expanded_neighbors,negatives], dim=1)
    
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)
    
    labels1 = torch.ones(logits.shape[0], dtype=torch.long).to(device)

    logits = logits / temperature

    return logits, labels,labels1,expand_no_neighbors

def upper_triangle(matrix,device='cuda'):
    upper = torch.triu(matrix, diagonal=0).to(device)
    #diagonal = torch.mm(matrix, torch.eye(matrix.shape[0]))
    diagonal_mask = torch.eye(matrix.shape[0]).to(device)
    #diagonal_mask = torch.eye(matrix.shape[0])
   
    return upper * (1.0 - diagonal_mask)

def regularizer(proto,device='cuda'):
   
    c_seen = proto.shape[0]
    proto_expand1 = proto.unsqueeze(0).to(device)
    proto_expand2 = proto.unsqueeze(1).to(device)
   
    proto_norm_mat = torch.sum((proto_expand2 - proto_expand1)**2, dim=-1).to(device)

    proto_norm_upper = upper_triangle(proto_norm_mat).to(device)
    
    d_mean = (2.0 / (c_seen**2 - c_seen) * torch.sum(proto_norm_upper)).to(device)
    
    sim_mat=torch.matmul(proto,proto.T).to(device)
   
    sim_mat_upper=(upper_triangle(sim_mat)).to(device)
    m=torch.max(sim_mat_upper)

    m=torch.min(sim_mat_upper)
   
   

    residuals = (upper_triangle((-proto_norm_upper +d_mean+ sim_mat_upper))).to(device)
    
    rw = (1 /c_seen )* (torch.sum(residuals)).to(device)

    return(rw)


def get_params_groups(model):
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]

class DistillLoss(nn.Module):
    def __init__(self, warmup_teacher_temp_epochs, nepochs, 
                 ncrops=2, warmup_teacher_temp=0.07, teacher_temp=0.04,
                 student_temp=0.1):
        super().__init__()
        self.student_temp = student_temp
        self.ncrops = ncrops
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax(teacher_output / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        return total_loss


class DistillLoss2(nn.Module):
    def __init__(self, warmup_teacher_temp_epochs, nepochs, 
                 ncrops=2, warmup_teacher_temp=0.07, teacher_temp=0.04,
                 student_temp=0.1):
        super().__init__()
        self.student_temp = student_temp
        self.ncrops = ncrops
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output,near_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        
        student_out = student_output / self.student_temp

        #print("student_out dimension",len(student_out))
        
        student_out2 = student_out.chunk(self.ncrops)
        #print("student_out dimension",len(student_out2))

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax(teacher_output / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)
        #print("teacher_out dimension",len(teacher_out))


        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out2)):
                #print("v",v)
                #print("iq",iq)
                #print("q",q.shape)
                #("student_out2 dimension ", student_out2[v].shape)
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out2[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        #print("total_loss",total_loss)
        near_out=[]
        total_loss_near=0
        n_loss_terms = 0
        for l in range(len(near_output)):
            temp= near_output[l] / self.student_temp
            near_out.append(temp)
            near_out[l] = F.softmax(near_out[l],dim=-1)
            near_out[l]= near_out[l].detach()

            #print("near_out dimension ", near_out[l].shape)
            #print("student_out dimension ", student_out.shape)

            loss=torch.sum(-(near_out[l] )* F.log_softmax(student_out, dim=-1), dim=-1)
            total_loss_near += loss.mean()
            n_loss_terms += 1

        #print("total_loss_near",total_loss_near)
        #("n_loss_terms",n_loss_terms)
        total_loss_near /= n_loss_terms
        total_loss = (total_loss+total_loss_near)/2

        return total_loss
