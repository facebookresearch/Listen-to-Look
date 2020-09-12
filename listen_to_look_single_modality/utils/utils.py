import torch
import torchaudio
import random
import numpy as np

def init_hidden(batch_size, cell_size):
    init_cell = torch.Tensor(batch_size, cell_size).zero_()
    if torch.cuda.is_available():
        init_cell = init_cell.cuda()
    return init_cell

def find_length(list_tensors):
    """find the length of list of tensors"""
    length = [x.shape[0] for x in list_tensors]
    return length

def pad_tensor(tensor, length):
    """Pad a tensor, given by the max length"""
    if tensor.size(0) == length:
        return tensor
    return torch.cat([tensor, torch.zeros([length - tensor.size(0), tensor.size(1)])])
   
def trim_tensor(tensor, start_index, length):
    """trim a tensor, given by the max length"""
    return tensor[start_index:start_index+length, :]

def process_list_tensors(gt_features, predictions, max_length=None, episode_length=None):
    """Pad a list of tensors and return a list of tensors"""
    tensor_length = find_length(gt_features)
    #store the indexes that are too short
    tensors_too_short = [index for index in range(len(tensor_length)) if tensor_length[index] < episode_length]

    if max_length is None:
        max_length = max(tensor_length)
    else:
        if max(tensor_length) < max_length:
            max_length = max(tensor_length)
    processed_gt_features = []
    processed_predictions = []
    for i in range(len(gt_features)):
        gt_tensor = gt_features[i]
        prediction_tensor = predictions[i]
        #print(image_tensor.shape)
        if gt_tensor.shape[0] < max_length:
            gt_tensor = pad_tensor(gt_tensor, max_length)
            prediction_tensor = pad_tensor(prediction_tensor, max_length)
        elif gt_tensor.shape[0] > max_length:
            start_index = random.randint(0, gt_tensor.shape[0] - max_length)
            gt_tensor = trim_tensor(gt_tensor, start_index, max_length)
            prediction_tensor = trim_tensor(prediction_tensor, start_index, max_length)
        #print(image_tensor.shape)
        processed_gt_features.append(gt_tensor)
        processed_predictions.append(prediction_tensor)
    #make sure number of meaningful features is larger or equal to episode length, other wise copy the last features
    if len(tensors_too_short) > 0:
        for index in tensors_too_short:
            current_length = tensor_length[index]
            for position in range(episode_length - current_length):
                random_index = random.randint(0,current_length-1)
                processed_gt_features[index][position + current_length, :] = processed_gt_features[index][random_index, :]          
                processed_predictions[index][position + current_length, :] = processed_predictions[index][random_index, :]          

    #print(len(processed_audio_features))
    return torch.stack(processed_gt_features), torch.stack(processed_predictions), tensor_length, max_length

def create_mask(batchsize, max_length, episode_length, length):
    """Given the length create a mask given a padded tensor"""
    tensor_mask = torch.zeros(batchsize, max_length)
    for idx, row in enumerate(tensor_mask):
        row[:max(episode_length,length[idx])] = 1
    return tensor_mask