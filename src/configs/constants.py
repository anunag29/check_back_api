import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

num_workers = 6
should_log = True

batch_size = 6
train_epochs = 5
word_len_padding = 12  # will be overriden if the dataset contains labels longer than the constant
learning_rate = 5e-6