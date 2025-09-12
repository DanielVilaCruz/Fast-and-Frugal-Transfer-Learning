# Fast-and-Frugal-Transfer-Learning
This repository provides a novel lightweight and energy-aware training strategy designed for environments with hardware limitations. 

The core idea relies on decoupling the classifier head from the feature extractor of a Convolutional Neural Network, so all features can be extracted only once, avoiding unnecesary iterations and computational overhead. To adapt the backbone to new domains, a thresholded batch normalization adaptation is proposed. Additionally, to increase the model representativeness and generalization capability, a new classifier head is proposed, among a weighted training for the classifier.

Results on Brain Cancer MRI, BreakHis, and PatchCamelyon datasets show a relevant reduction of training time and associated CO2 emissions while maintaining or even improving accuracy.

