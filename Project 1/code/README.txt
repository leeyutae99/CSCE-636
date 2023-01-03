This is project submission for Yutae Lee in CSCE636 Class at Texas A&M University.


This .txt file contains information about how to run the main.py file:

Before running, the codes does not download cifar-10 data by itself, so make sure that you have cifar-10 dataset in the location where main.py is at

When training run python main.py train --save_dir --weight_decay --learning_rate --momentum --max_epochs

If you simply want to use the default hyperparameters that I made in Configure.py, you can just run python main.py train

The default hyperparameters that I chosed is following:

save_dir = '../saved_models/'
weight_decay = 5e-4
learning_rate = 0.01
momentum = 0.7
max_epochs = 100

When training is done, you should find model-%d.ckpt%(epoch) and best_model.ckpt files in the direction ../saved_models/

If you want test the model run python main.py test

If you want to create a prediction using best_model.ckpt you can run python main.py predict

Then you will be able to find prediction.npy file in the location where you have the main.py file.

Also you can find saved_models with prediction.npy, model-%d.ckpt%(epoch) and best_model.ckpt files are included in the submission.





