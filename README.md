# Deep Autoencoding Gaussian Mixture Model for Unsupervised Anomaly Detection in PyTorch

My attempt at reproducing the paper [Deep Autoencoding Gaussian Mixture Model for Unsupervised Anomaly Detection](https://openreview.net/forum?id=BJJLHbb0-). Please Let me know if there are any bugs in my code. Thank you! =)

I implemented this on Python 3.6 using PyTorch 0.4.0.

### Dataset
KDDCup99 http://kdd.ics.uci.edu/databases/kddcup99/

### Some Test Results
Paper's Reported Results (averaged over 20 runs) : Precision : 0.9297, Recall : 0.9442, F-score : 0.9369

My Implementation (only one run) : Precision : 0.9677, Recall : 0.9538, F-score : 0.9607

### Visualizing the z-space:
<img src="https://github.com/danieltan07/dagmm/blob/master/z_space.png" width="50%"/>
