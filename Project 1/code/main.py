 ### YOUR CODE HERE
# import tensorflow as tf
# import torch
import os, argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from Model import MyModel
from DataLoader import load_data, train_valid_split, load_testing_images
from Configure import model_configs, training_configs

parser = argparse.ArgumentParser()
parser.add_argument("mode", help="train, test or predict")
for key, value in model_configs.items():
    parser.add_argument(("--"+key), type=type(value), default=value)
args = parser.parse_args()

data_dir = "cifar-10-batches-py"
private_data = 'private_test_images_2022.npy'
result_dir = 'predictions.npy'

if __name__ == '__main__':
    model_configs = args
    model = MyModel(model_configs)
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda:0':
        model = model.cuda()
        model.network = torch.nn.DataParallel(model.network)
        cudnn.benchmark = True
    
    model.model_setup();

    if args.mode == 'train':
        x_train, y_train, x_test, y_test = load_data(data_dir)
        x_train_new, y_train_new, x_valid, y_valid = train_valid_split(x_train, y_train, 0.9)
        
        print("Training data shape")
        print(x_train_new.shape)
        print(y_train_new.shape)
        
        print("Validation data shape")
        print(x_valid.shape)
        print(y_valid.shape)

        model.train(x_train_new, y_train_new, training_configs, x_valid, y_valid)
        model.evaluate(x_test, y_test)
        

    elif args.mode == 'test':
        checkpointfile = os.path.join('../saved_models', 'best_model.ckpt')
        model.load(checkpointfile)
        _, _, x_test, y_test = load_data(data_dir)
        model.evaluate(x_test, y_test)
    elif args.mode == 'predict':
        checkpointfile = os.path.join('../saved_models', 'best_model.ckpt')
        model.load(checkpointfile)
        x_test = load_testing_images(private_data)
        predictions = model.predict_prob(x_test)
        predictions = predictions.cpu().numpy()
        np.save(result_dir, predictions)

### END CODE HERE

