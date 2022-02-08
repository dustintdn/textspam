# Text Message Spam Identifier
This repository is work done in Natural Language Processing coursework in the QMSS program.

A project that identifies whether a given text message is spam or "ham" which is a nickname for a legitimate text message.

The data is sourced from (https://www.dt.fee.unicamp.br/~tiago/smsspamcollection/) which includes 5,574 SMS text messages consisting of 4,827 legitimate messages and 747 spam messages.

This repository will show how the text data was cleaned through stopword removal, lemmatization, and tokenization (unigram and bigrams were tested, with unigrams performing with better metrics across all models). There is python script file for all descriptive graphics created for the data [descriptive_graphs.py] such as character length and punctuation histograms as well as word clouds for most commonly appearing words in spam-labeled text messages and ham-labeled text messages. Multiple models were tested, optimized with a GridSearchCV pipeline, and cross validated to determine the best model. The script file [models.py] will show confusion matrix reports for each model utilized.

Results showed that the Support Vector Machine (SVM) Classier performed with the highest F1, which was the primary metric in determining model performance.
