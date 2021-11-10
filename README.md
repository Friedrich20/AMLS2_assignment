# README

This repo serves as the code implementation of the project assigned in the module Applied Machine Learning Systems II ELEC0135.

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

