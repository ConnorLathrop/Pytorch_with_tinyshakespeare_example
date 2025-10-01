# Pytorch_with_tinyshakespeare_example

This is a test of creating a Transformer Model using Pytorch and training it with tiny shakespeare. The code was adapted from https://github.com/pytorch/examples/tree/main/word_language_model with the use of AI (see AI_HELP) for examples of what was done.
# Install and Run
To install it is recommened to use a venv
```
python -m venv venv
# On windows:
source venv/Scripts/Activate
```
The requirements to run can then be installed using pip
```
pip install -r requirements.txt
```
Once installed the model can be trained and generate text to the outputs directory
```
python train.py
...
# Wait for training to complete
python generate.py --prompt "ROMEO:"
```
This will output a generated text to the outputs directory.

# Results
The results for this project were interesting. I had expected on of the AI's to generate a better project, but found that the other worked better. There was an issue getting the cuda to work so a CPU was used instead of a GPU but that was most likely user error. The generated text was okay, there were definently mistakes in some of the spelling of the words and the grammar, but the model seemed to produce results with the names properly. The CPU time to produce each model was about ~160 seconds which was lower than I had expected. Running 10 epochs in < 30 minutes seemed to be a good result. The outputs folder contains some of the generated texts and a json file that shows the training and validation loss for each epoch.
