import torch
from torch.nn import CosineSimilarity

cosine_sim_func = CosineSimilarity(dim=2, eps=1e-6)


def sliding_window_similarity(hidden_state, prototypes, sim_func=cosine_sim_func, return_windows=False):
    hidden_len = hidden_state.shape[0]  # hidden state expected to be a single Length X Embedding_Size tensor
    proto_len = prototypes.shape[1]  # prototypes expected to be a num_prototype X prototype_len X Emebedding_Size tensor
    sims_for_this_hidden = []

    for i in range(0, hidden_len - proto_len + 1, 1):
        hidden_subsection_window = hidden_state[i:i + proto_len, :]
        sim_i = cosine_sim_func(hidden_subsection_window, prototypes)
        sim_i_activation = torch.mean(sim_i, dim=1)
        sims_for_this_hidden.append(sim_i_activation)
    sims_for_this_hidden_tensor = torch.stack(sims_for_this_hidden)

    returned_sim = torch.amax(sims_for_this_hidden_tensor, dim=0)
    returned_windows = []

    if return_windows:
        window_starting_idxs = torch.argmax(sims_for_this_hidden_tensor, dim=0)
        # because length of prototype is the window size
        returned_windows = [slice(idx, idx + proto_len) for idx in window_starting_idxs]
        # in this case, returned_windows should be a num_prototype X 1 list of slice objects

    return returned_sim, returned_windows


# hidden_states expected to be a Batch x Length x Embedding tensor
def get_sims_for_prototypes(hidden_states, all_prototypes, sim_func=cosine_sim_func, return_windows=False):
    all_sims = []
    all_sim_windows = []
    for hs in hidden_states:
        hs_sims, sim_windows = sliding_window_similarity(hs, all_prototypes, sim_func=sim_func,
                                                         return_windows=return_windows)
        all_sims.append(hs_sims)
        all_sim_windows.append(sim_windows)

    all_sims_tensor = torch.stack(all_sims)
    return all_sims_tensor, all_sim_windows


#function for tunrning an input ids list, an activation window and a tokenizer into a string
def get_activated_tokens(input_ids_list, activation_window, tokenizer):
    activated_ids = input_ids_list[activation_window]
#     activated_tokens = tokenizer.convert_ids_to_tokens(activated_ids)
    activated_tokens = tokenizer.decode(activated_ids)

    return activated_tokens

def l2_similarity_func(prototype, embeddings, eps=1e-4):
    diff = prototype - embeddings
    diff_sqrd = diff.pow(2)
    diff_sum = diff_sqrd.sum(dim=1)
    sim = 1 / (diff_sum.sqrt() + eps)
    return sim


