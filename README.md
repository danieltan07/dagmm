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

### Some Implementation Details
Below are code snippets of the two main components of the model. More specifically, computing the gmm parameters and sample energy.

```python
    def compute_gmm_params(self, z, gamma):
        N = gamma.size(0)
        # K
        sum_gamma = torch.sum(gamma, dim=0)
        # K
        phi = (sum_gamma / N)
        self.phi = phi.data
        # K x D
        mu = torch.sum(gamma.unsqueeze(-1) * z.unsqueeze(1), dim=0) / sum_gamma.unsqueeze(-1)
        self.mu = mu.data
        # z = N x D
        # mu = K x D
        # gamma N x K
        # z_mu = N x K x D
        z_mu = (z.unsqueeze(1)- mu.unsqueeze(0))

        # z_mu_outer = N x K x D x D
        z_mu_outer = z_mu.unsqueeze(-1) * z_mu.unsqueeze(-2)

        # K x D x D
        cov = torch.sum(gamma.unsqueeze(-1).unsqueeze(-1) * z_mu_outer, dim = 0) / sum_gamma.unsqueeze(-1).unsqueeze(-1)
        self.cov = cov.data
        return phi, mu, cov
```   
I added some epsilon on the diagonals of the covariance matrix, otherwise I get nan values during training.

I tried using `torch.potrf(cov_k).diag().prod()**2` to compute for the determinants, but for some reason I get errors after several epochs, so I used numpy's linalg to compute for the determinants instead.

```python
    def compute_energy(self, z, phi=None, mu=None, cov=None, size_average=True):
        # Compute the energy based on the specified gmm params. 
        # If none are specified use the cached values.
        
        if phi is None:
            phi = to_var(self.phi)
        if mu is None:
            mu = to_var(self.mu)
        if cov is None:
            cov = to_var(self.cov)
        k, D, _ = cov.size()

        z_mu = (z.unsqueeze(1)- mu.unsqueeze(0))

        cov_inverse = []
        det_cov = []
        cov_diag = 0
        eps = 1e-12
        for i in range(k):
            # K x D x D
            cov_k = cov[i] + to_var(torch.eye(D)*eps)
            cov_inverse.append(torch.inverse(cov_k).unsqueeze(0))
            det_cov.append(np.linalg.det(cov_k.data.cpu().numpy()* (2*np.pi)))
            cov_diag = cov_diag + torch.sum(1 / cov_k.diag())

        # K x D x D
        cov_inverse = torch.cat(cov_inverse, dim=0)
        # K
        det_cov = to_var(torch.from_numpy(np.float32(np.array(det_cov))))
        
        # N x K
        exp_term_tmp = -0.5 * torch.sum(torch.sum(z_mu.unsqueeze(-1) * cov_inverse.unsqueeze(0), dim=-2) * z_mu, dim=-1)
        # for stability (logsumexp)
        max_val = torch.max((exp_term_tmp).clamp(min=0), dim=1, keepdim=True)[0]

        exp_term = torch.exp(exp_term_tmp - max_val)

        sample_energy = -max_val.squeeze() - torch.log(torch.sum(phi.unsqueeze(0) * exp_term / (torch.sqrt(det_cov)).unsqueeze(0), dim = 1) + eps)
        
        if size_average:
            sample_energy = torch.mean(sample_energy)

        return sample_energy, cov_diag
```
