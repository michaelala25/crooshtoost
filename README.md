# CrooshToost - Readme

## Intro

**CrooshToost** is a server-side virtual assistant for interacting with Keras models from afar. Given you have some neural architecture in Keras/Tensorflow you wish to run, once you've set it up with CrooshToost all you have to do is
text CrooshToost various commands in order to interact with your server-side model.

CrooshToost doesn't require you to send very specific command line arguments or python code, it'll simply interpret your natural language as a command.

## Desired Functionality

#### Example 1 - Progress Report
**User:** Hey CrooshToost, what's the status of the currently training model?
**CrooshToost:** We're currently at Epoch 32/150,  batch number 564/7258, with a loss of 3.9264e-05.
**User:** Can you start checkpointing the epochs?
**CrooshToost:** Sure, any desired file format?
**User:** Nah just the default is fine.
**CrooshToost:** Ok. Also, save_weights_only=True or False?
**User:** False.
**CrooshToost:** Got it.
**User:** Also, what's the loss history for this model?
**CrooshToost:** The last 20 epochs have finished with losses of [3.9887e-05, 3.9843e-05, ..., 3.9264e-05]

#### Example 2 - Changing Learning Rate Mid-Training
**User:** CrooshToost, how is FR_CNN's training going?
**CrooshToost**: We're currently at Epoch 17/100, batch number 1660/2663, with a loss of 1.7763e-05. I should let you know that the loss hasn't changed in the past 6 epochs.
**User:** Oh huh. What's the current learning rate?
**CrooshToost**: Currently the learning rate is 0.01, but I can't be sure this won't change in the future, the model's LearningRateScheduler has an unannotated lambda expression which I can't read.
**User:** Delete the current learning rate scheduler and try setting the learning rate to 0.1. Let me know if you see any change after 5 epochs.
**CrooshToost**: Ok.

*5 epochs later*

**CrooshToost**: After changing the learning rate to 0.1, the model's loss still hasn't changed from 1.7763e-05.
**User:** Interesting, do you have any suggestions?
**CrooshToost**: I noticed that shuffle=False in model.fit's keyword arguments. Do you want to try retraining with shuffle=True.
**User:** Yeah sure.
**CrooshToost**: Alright, cancelling the current training session.