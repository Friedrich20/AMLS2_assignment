# README

This repo serves as the code implementation of the project assigned in the module Applied Machine Learning Systems II ELEC0135.

# Description

In this project, we are required to solve two tasks upon Twitter sentiment analysis in the field of natural language processing. The first task (A) deals with a three-point scale tweet polarity classification problem, namely to classify whether the sentiment of the tweet is positive, negative or neutral, while the second task (B) works on a two-point scale topic-based message polarity classification issue, aiming to distinguish the positive and negative sentiments of tweets towards the topic. In this paper, a method of Recurrent Neural Network (RNN), or more precisely Long Short-Term Memory (LSTM) is applied to tackle these two tasks, along with a complete text preprocessing pipeline and word embedding approaches. The combination of these techniques has brought us a high classification accuracy of 62.9% for task A and 86.5% for task B. Need to mention that these two tasks are originally the Competition Task 4 held by SemEval 2017 where our results rank 13rd/38 and 3rd/23 on the list although we are not officially participating.

# Results

| Task |  Model  | Train Acc | Test Acc | Epoch Count | Elapsed Time |
| :--: | :-----: | :-------: | :------: | :---------: | :----------: |
|  A   | Bi-LSTM |   72.8%   |  62.9%   |     17      |   00:27:50   |
|  B   | Bi-LSTM |   95.1%   |  86.5%   |     18      |   00:15:43   |

# Requirements

- Python 3.7+ (Python 3.7.10 is recommended which is the version used in the development.)
- macOS or Windows
- Required modules
  - check *requirement.txt* to ensure all modules included have been installed
  - or run ```pip3 install -r requirements.txt``` in terminal to install with ease

# Usage

1. Switch to the root directory of the repo
2. Run ```python3 main.py``` in terminal to start the main program
3. Monitor the running process in *base_log.log* if you like (logging level is set to ```INFO``` by default, feel free to alter)

# Structure

```.
├── A                      # the folder of Task A
│   ├── A_lstm_model.h5    # trained model for Task A
│   └── A_sentiment_classification.py                     # scripts for Task A
├── B                      # the folder of Task B
│   ├── B_lstm_model.h5    # trained model for Task B
│   └── B_topic_based_sentiment_classification.py         # scripts for Task B
├── Datasets               # the folder of datasets
│   ├── 4A-English
│   └── 4B-English
├── README.md               # this file
├── helper                  # the folder of some necessary files
│   ├── a_learning curve.png  # learning curve for Task A
│   ├── b_learning curve.png  # learning curve for Task B
│   └── base_log.log        # the main log, generated once the main program srarts
├── main.py                 # the entrance to main program
└── requirements.txt        # the list of required libraries
```

# Having problems?

If you run into problems, please either file a github issue or send an email to uceewta@ucl.ac.uk.

