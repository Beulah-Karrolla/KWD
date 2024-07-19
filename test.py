import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
import numpy as np
from model import SpeechClassifierModel, SpeechClassifierModelTransformer, ConformerModel
from data import SpeechCommandsDataset, collate_fn

def main(args):
    local_rank = args.device
    torch.cuda.set_device(local_rank)
    #device = torch.device('cpu')
    device = torch.device('cuda:{:d}'.format(local_rank))

    # Load the trained model from the saved checkpoint
    checkpoint_path = args.checkpoint_path
    checkpoint = torch.load(checkpoint_path)
    model_state_dict = checkpoint["model_state_dict"]
    #import ipdb;ipdb.set_trace()
    model_params = {'num_classes': checkpoint['model_params']['num_classes'],
                    'feature_size': checkpoint['model_params']['feature_size'], 
                    'hidden_size': checkpoint['model_params']['hidden_size'], 
                    'num_layers': checkpoint['model_params']['num_layers'], 
                    'dropout': checkpoint['model_params']['dropout'], 
                    'bidirectional': checkpoint['model_params']['bidirectional'], 
                    'device': device}  
    #model = SpeechClassifierModel(**model_params)
    model = ConformerModel(**model_params)
    model=model.to(device)
    
    model.load_state_dict(model_state_dict)
    model.eval()

    # Define the test dataset and data loader
    test_dataset = SpeechCommandsDataset(args.dataset_path, 'testing', device=device)
    test_dataloader = DataLoader(test_dataset,batch_size=64, shuffle=1, collate_fn=collate_fn)

    # Iterate over the test data and pass it through the model to get predictions
    '''correct = 0
    total = 0
    count = 0
    fp = 0
    fn = 0
    tp = 0
    tn = 0'''
    # Initialize metrics for different label lengths
    metrics = {}
    for i in range(1, 15):
        metrics[i] = {'total': 0, 'correct': 0, 'fp': 0, 'fn': 0, 'tp': 0, 'tn': 0}

    count  = 0
    mistakes = []
    loss_list = []
    len_list_true = []
    loss_fn = nn.BCELoss()
    f = open(args.results_path ,'a')
    with torch.no_grad():
        for inputs, labels, pholders, text, text_ori in test_dataloader:
            data, label, pholder, text, text_ori = inputs.to(device), labels.to(device), pholders, text.to(device), text_ori
            with torch.no_grad():
                output = model(data, text)
            label_float = label.float()
            loss = loss_fn(torch.flatten(output), label_float)
            loss_list.append(loss.item())
            #best = np.where(output.cpu() < 1, 0, 1)
            #predicted = best
            '''for i in range(len(predicted)):
                total += 1
                correct += (predicted[i].item() == labels[i])
                if predicted[i] != labels[i]:
                    if predicted[i] == 1:
                        fp+=1
                    elif predicted[i] == 0:
                        fn+=1'''
            for i in range(len(output)):
                predicted = torch.where(output[i].cpu() < 0.9, 0, 1)
                label_length = len(text_ori[i])
                metrics[label_length]['total'] += 1
                metrics[label_length]['correct'] += (predicted.item() == label[i])
                if predicted != label[i]:
                    if predicted == 1:
                        metrics[label_length]['fp'] += 1
                    elif predicted == 0:
                        metrics[label_length]['fn'] += 1
                    mistakes.append([pholder[i], predicted, labels[i], output[i]])
                else:
                    if predicted == 1:
                        metrics[label_length]['tp'] += 1
                    elif predicted == 0:
                        metrics[label_length]['tn'] += 1
                    #correct += 1
                print(count)
                count +=1
            '''label = label.item()
            #import ipdb;ipdb.set_trace()
            #_, predicted = torch.max(output.data, 1)
            total += 1
            correct += (predicted.item() == label)
            if predicted != label:
                mistakes.append([pholder, predicted, label, output])
            else: 
                None
            print(count)
            count +=1'''
    import ipdb;ipdb.set_trace()
    for length, metric in metrics.items():
        print("Metrics for label length {}: ".format(length))
        print("Total number of examples: {}".format(metric['total']))
        print("Total number of correct predictions: {}".format(metric['correct']))
        print("False positives: {}".format(metric['fp']))
        print("False negatives: {}".format(metric['fn']))
        print("True positives: {}".format(metric['tp']))
        print("True negatives: {}".format(metric['tn']))
        print()
    import ipdb;ipdb.set_trace()
    print("Total number of mistakes: {}".format(len(mistakes)))
    print("Total number of test examples: {}".format(total))
    print("Total number of correct predictions: {}".format(correct))
    print(mistakes)
    print("Total number of mistakes: {}".format(len(mistakes)))
    print("Total number of test examples: {}".format(total))
    print("Total number of correct predictions: {}".format(correct))
    f.write(str(mistakes))
    f.close()
            
    import ipdb;ipdb.set_trace()
    # Calculate the accuracy of the model on the test data
    accuracy = float(correct) / float(total)

    # Print the accuracy
    print("Accuracy: {:.2f}%".format(accuracy * 100))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Speech Commands Testing for Wake Word Detection')
    parser.add_argument('--dataset_path', default=None, type=str, help='Path to dataset')
    parser.add_argument('--device', default=None, type=int, help='Cuda device')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to the best checkpoint')
    parser.add_argument('--results_path', type=str, default=None, help='Path to the save the results')
    args = parser.parse_args()

    main(args)