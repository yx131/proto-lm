import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForSequenceClassification
import pytorch_lightning as pl
import torchmetrics.functional as tmf
import torchmetrics.functional.classification as tmfc
from utils import get_sims_for_prototypes

class proto_lm(pl.LightningModule):
    def __init__(self,
                 pretrained_model,
                 max_seq_length=50,
                 num_prototypes=200,
                 hidden_shape=1024,
                 num_classes=2,
                 cohsep_ratio=0.5,
                 lambda0=0.5,
		         lambda1=0.25,
                 lambda2=0.25,
                 lr=3e-4,
                 optim_eps=1e-8,
                 betas=(0.9, 0.999),
                 dist_eps=1e-8,
                 analyze_mode=False,
                 proto_training_weights=False,
                 ):
        super().__init__()
        self.save_hyperparameters(ignore='pretrained_model')

        #get model obj
        self.LLM = pretrained_model

        self.hidden_shape = hidden_shape

        self.l0 = lambda0
        self.l1 = lambda1
        self.l2 = lambda2
        #self.l1 = (1 - self.l0) / 2
        #self.l2 = self.l1  #evenly distribute lambda1 and lambda2

        #create prototypes
        self.prototypes = nn.Parameter(torch.randn(size=(self.hparams.num_prototypes,
                                                         self.hidden_shape)),
                                       requires_grad=True)

        #a vector to keep track of which prototype belongs to which class
        self.prototype_class_vec = torch.zeros(self.hparams.num_prototypes, self.hparams.num_classes)
        self.num_prototypes_per_class = self.hparams.num_prototypes // self.hparams.num_classes
        for j in range(self.hparams.num_prototypes):
            self.prototype_class_vec[j, j // self.num_prototypes_per_class] = 1

        #hyper parameter K is the number of prototypes to push/pull during min/max loss
        if self.hparams.cohsep_ratio >= 1: #if using a flat constant for K
            self.K = int(self.hparams.cohsep_ratio)
        else: #if a ratio then calculate that number of prototypes
            self.K = int(self.num_prototypes_per_class * self.hparams.cohsep_ratio)

        self.prev_proto = self.prototypes.detach().clone()


        #parameters for word-level-attention
        self.fc_word_level = torch.nn.Linear(in_features=self.hidden_shape, out_features=self.hidden_shape, bias=True)
        self.W_nu = nn.Parameter(torch.randn(size=(self.hidden_shape, 1)))

        #dense layer connecting attention/similarity scores to classes
        self.dense = torch.nn.Linear(self.hparams.num_prototypes, self.hparams.num_classes)

        #if we're in the prototype training stage, initialize weights for pos/neg association for each class
        self.initialize_weights_for_prototype_training()
        self.set_grads_for_proto_train()
        self.misclassified = []

    def initialize_weights_for_prototype_training(self, negative_assoc=-.5):
        positive_one_weights_locations = torch.t(self.prototype_class_vec)
        negative_one_weights_locations = 1 - positive_one_weights_locations

        positive_assoc = 1
        self.dense.weight.data.copy_(
            positive_assoc * positive_one_weights_locations
            + negative_assoc * negative_one_weights_locations)

    def set_grads_for_proto_train(self):
        for p in self.LLM.parameters():
            p.requires_grad = bool(self.hparams.proto_training_weights)
        self.prototypes.requires_grad = True
        for p in self.dense.parameters():
            p.requires_grad = True

    def set_grads_for_dense_weights_optimization(self):
        for p in self.LLM.parameters():
            p.requires_grad = False
        self.prototypes.requires_grad = False
        for p in self.dense.parameters():
            p.requires_grad = True

    def _get_proto_grad_changes(self):
        cur_proto = self.prototypes.detach().clone()
        if self.prev_proto.device != cur_proto.device:
            self.prev_proto = self.prev_proto.to(cur_proto.device)
        delta = torch.sum(torch.abs(cur_proto - self.prev_proto))
        self.prev_proto = cur_proto
        return delta


    def hierarchical_attention_calculation(self, hidden_states):
       #hidden states expected to be Batch X Length X Hidden
       upsilon = torch.tanh(self.fc_word_level(hidden_states)) #still Batch X Length X Hidden

       nu_stack = []
       for i in range(upsilon.shape[0]): #for each instance in batch
           nu_i = torch.mv(upsilon[i], self.W_nu.squeeze()) #calculate nu_t for each step, i call all the nu_t's together nu_i, as in nu's for the instance
           nu_stack.append(nu_i)
       nu_stack = torch.stack(nu_stack, dim=0) #Batch X Length
       alphas = F.softmax(nu_stack, dim=1) #Batch X Length

       #calculate attention-weighted hidden states
       all_S = []
       for sample_alphas, sample_hidden in zip(alphas, hidden_states):
           S_i = torch.mm(sample_alphas.unsqueeze(0), sample_hidden).squeeze(0)
           all_S.append(S_i)
       all_S = torch.stack(all_S, dim=0)

       all_sims = []
       for S_i in all_S:
           diff_i = S_i - self.prototypes
           diff_i_sqrd = diff_i.pow(2)
           diff_i_summed = diff_i_sqrd.sum(dim=1)
           sim_i = 1 / (diff_i_summed.sqrt() + self.hparams.dist_eps)
           all_sims.append(sim_i)
       all_sims = torch.stack(all_sims, dim=0) #Batch x num_prototypes
       return alphas, all_S, all_sims


    def forward(self, **inputs):
        #in case the key 'labels' is part of the kwarg input, delete it, because it can't be handled by the base model
        if 'labels' in inputs.keys():
            del inputs['labels']

        llm_out = self.LLM(**inputs, output_hidden_states=True)
        last_hidden_states = llm_out.hidden_states[-1]
        alphas, proto_hiddens, similarities = self.hierarchical_attention_calculation(last_hidden_states)

        # similiarities, sim_windows = get_sims_for_prototypes(hidden_states,self.prototypes, return_windows=self.hparams.analyze_mode)
        logits = self.dense(similarities)
        probs = F.softmax(logits, dim=1)

        out_dict = {
            'loss':None, #loss is calculated in another function
            'probs':probs,
            'logits':logits.detach().clone() if self.hparams.analyze_mode else logits,
            'hidden_states':last_hidden_states,
            'llm_attention':llm_out, #can specificy attention here
            'alphas': alphas,
            'proto_hiddens': proto_hiddens,
            'similarities': similarities
        }


        return out_dict

    def calc_loss(self, logits, labels, similarities):
        if self.hparams.num_classes > 1: #classification
            ce_loss =  self.l0 * F.cross_entropy(logits, labels)
        elif self.hparams.num_classes == 1: #regression
            ce_loss = self.l0 * F.mse_loss(logits, labels)
            labels = torch.zeros(size=labels.size()) #all examples belong to the "0" class, in case of regression

        proto_lables = torch.argmax(self.prototype_class_vec, dim=1).cuda()
        correct_class_sims = []
        for lab, sim in zip(labels, similarities):
            lab_mask = proto_lables == lab
            correct_class_sims.append(sim[lab_mask])
        correct_class_sims = torch.stack(correct_class_sims, dim=0)

        #find, for each example in batch, the top K least similar protoypes in its class
        least_sims_in_class , _ = torch.topk(correct_class_sims, largest=False, k=self.K, dim=1)
        #loss for each example, is the average of top_n_sim
        cohesion_cost_example = torch.mean(least_sims_in_class, dim=1)
        #loss for the batch is the average of losses for each example
        cohesion_cost = torch.mean(cohesion_cost_example)
        cohesion_loss =  self.l1 * cohesion_cost


        if self.hparams.num_classes > 1: #classification
            self.prototype_class_vec = self.prototype_class_vec.to(labels.device)
            prototypes_of_correct_class = torch.t(self.prototype_class_vec[:, labels]).to(labels.device)
            prototypes_of_wrong_class = 1 - prototypes_of_correct_class
            incorrect_class_sim = similarities * prototypes_of_wrong_class
            # find, for each example in batch, the top K most similar prototypes from other classes
            most_sim_out_of_class, _ = torch.topk(incorrect_class_sim, largest=True, k=self.K, dim=1)
            separation_cost_example = torch.mean(most_sim_out_of_class, dim=1)
            separation_cost = torch.mean(separation_cost_example)
            # -1 because we're "maximizing" this loss
            separataion_loss = -1 * self.l2 * separation_cost
        elif self.hparams.num_classes == 1: #no separateion cost for regression
            separataion_loss = 0

        #similarly find separation cost


        all_losses = {}
        all_losses['ce_loss'] = ce_loss
        all_losses['cohesion_loss'] = cohesion_loss
        all_losses['separation_loss'] = separataion_loss
        all_losses['total_loss'] = ce_loss + cohesion_loss + separataion_loss

        return all_losses

    def _shared_eval_step(self, stage='train', **batch):
        labels = batch['labels']
        seq_class_out = self(**batch)
        similarities = seq_class_out['similarities']
        logits = seq_class_out['logits']
        probs = seq_class_out['probs']

        all_losses = self.calc_loss(logits, labels, similarities)
        seq_class_out['loss'] = all_losses['total_loss']

        #create metrics dictionary
        metrics_dict = {}

        #record losses
        # for the special case 'loss' during train stage, do not prepend the prefix "train"
        metrics_dict[f'{stage + "_" if stage != "train" else ""}loss'] = seq_class_out['loss']
        metrics_dict['ce_loss'] = all_losses['ce_loss']
        metrics_dict['cohesion_loss'] = all_losses['cohesion_loss']
        metrics_dict['separation_loss'] = all_losses['separation_loss']

        #record grad changes in prototype layer
        metrics_dict[f'{stage}_prototype_grads'] = self._get_proto_grad_changes()

        if self.hparams.num_classes > 1: #classification
            #calculate some metrics
            metrics_dict[f'{stage}_f1'] = tmfc.multiclass_f1_score(probs, labels, average='micro', num_classes=self.hparams.num_classes)
            metrics_dict[f'{stage}_precision'] = tmfc.multiclass_precision(probs, labels, average='micro', num_classes=self.hparams.num_classes)
            metrics_dict[f'{stage}_recall'] = tmfc.multiclass_recall(probs, labels, average='micro',  num_classes=self.hparams.num_classes)
            metrics_dict[f'{stage}_accuracy'] = tmfc.multiclass_accuracy(probs, labels, average='micro', num_classes=self.hparams.num_classes)
        if self.hparams.num_classes == 1: #regresion:
            metrics_dict[f'{stage}_mse'] = tmf.mean_squared_error(logits.squeeze(), labels)
            metrics_dict[f'{stage}_rho'] = tmf.pearson_corrcoef(logits.squeeze(), labels)

        self.log_dict(metrics_dict, prog_bar=True)

        return seq_class_out, metrics_dict

    def training_step(self, batch, batch_idx):
        outputs, metrics = self._shared_eval_step(stage='train', **batch)
        loss = outputs['loss']
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs, metrics = self._shared_eval_step(stage='val', **batch)
        val_loss, probs = metrics['val_loss'], outputs['probs']
        preds = probs
        labels = batch['labels']
        return metrics.update({'preds': preds, 'labels': labels})

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        outputs, metrics = self._shared_eval_step(stage='test', **batch)
        test_loss, probs = metrics['test_loss'], outputs['probs']
        preds = probs
        labels = batch['labels']

        if self.hparams.num_classes > 1: #classification
            missed_mask = preds.argmax(dim=1) != labels
            missed_indicies = missed_mask.nonzero()
            for idx in missed_indicies:
                self.misclassified.append((batch['input_ids'][idx], labels[idx]))
        elif self.hparams.num_classes == 1: #regression
            pass

        return metrics.update({'preds': preds, 'labels': labels})

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, eps=self.hparams.optim_eps, betas=self.hparams.betas)
        return optimizer
