# NeuralTQE
A stacking model of quality estimation implemented with PyTorch 

# Training and Testing Data Format
Training and Testing data should be in the format as :
source_sentence \t target_sentence \t \ score
Note that all sentences should be segmented.

# Using GPU
If you are running the code with GPU, simply set the ``use_cuda=torch.cuda.is_available()'', otherwise set ``use_cuda=False''

# Pretrained Embeddings
Download or pretrain monolingual or crosslingual embeddings and have the first line in each file deleted.  Note that pretrained embeddings may be  problematic due to some lines contain empty words or null vectors. 


