import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from scipy.stats import norm
import cv2 

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


def get_intersection(A, B):
    dx = min(A[2], B[2]) - max(A[0], B[0])
    dy = min(A[3], B[3]) - max(A[1], B[1])
    if (dx>=0) and (dy>=0):
        return dx*dy
    return 0

def get_area(A):
    # return width * height
    return (A[2]-A[0]) * (A[3]-A[1])

def compute_average_precision(target, output, IOU_TH = 0.1):
    (TP, FN, FP) = (0,1,2)
    f = [0,0,0]

    # Compute the intersection over union 
    list_of_t = []
    list_of_o = []
    for i,t in enumerate(target):
        for j,o in enumerate(output):
            if o in list_of_o: continue
            if t in list_of_t: continue    
            target_area = get_area(t)
            output_area = get_area(o)
            
            intersection = get_intersection(t,o)
            union = target_area + output_area - intersection
            

            if (intersection / union) > IOU_TH:
                f[TP] += 1
                list_of_t.append(t)        
                list_of_o.append(o)
            
                
            # print(i,j,"|",target_area, output_area,intersection,"|",f)
    f[FP] += len(output) - len(list_of_t)
    f[FN] += len(target) - len(list_of_t)

    return f


def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def compute_slope(y):
    N = len(y)
    x = np.stack((np.arange(N),np.ones((N,))),axis = 1)
    w = np.linalg.pinv(x.T.dot(x)).dot(x.T).dot(y)
    return np.squeeze(w[0])

def prob_slope(y, value):
    slope = compute_slope(y)
    var = np.var(y,ddof=1)
    N = len(y)
    p_slope = norm(slope,np.sqrt(12*var/(N**3 - N))).cdf(value)
    
    return p_slope

def count_steps_without_decrease(y):
    steps_without_decrease = 0
    n = len(y)
    for i in reversed(range(0,n-1,10)):
        if prob_slope(y[i:n],0) < 0.51:
            steps_without_decrease = n-i
    return steps_without_decrease
    
def count_steps_without_decrease_robust(y):
    p = np.percentile(y,90)
    
    return count_steps_without_decrease(y[y<p])
    
def bgr_to_rgb(image):
    (b,g,r) = cv2.split(image) # get b,g,r
    rgb_img = cv2.merge([r,g,b]) # switch it to rgb
    return rgb_img

class FocalLoss(nn.Module):
    """
        This criterion is a implemenation of Focal Loss, which is proposed in 
        Focal Loss for Dense Object Detection.
            
            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])
    
        The losses are averaged across observations for each minibatch.
        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5), 
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """
    def __init__(self, class_num, alpha=None, gamma=2, weights=None,size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.weights = weights
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        print(N)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        #print(class_mask)
        

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]
        
        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()
        #print('probs size= {}'.format(probs.size()))
        #print(probs)

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p 
        #print('-----bacth_loss------')
        #print(batch_loss)

        
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

class BinaryFocalLoss(nn.Module):
    """
        This criterion is a implemenation of Focal Loss, which is proposed in 
        Focal Loss for Dense Object Detection.
            
            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])
    
        The losses are averaged across observations for each minibatch.
        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5), 
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """
    def __init__(self, alpha=0.25, gamma=2, size_average=True, weight=None):
        super(BinaryFocalLoss, self).__init__()

        if alpha is None:
            self.alpha = 1
        else:
            self.alpha = alpha
        
        if weight is None:
        	self.weight = [1, 1]
       	else:
       		self.weight = weight

        self.gamma = gamma
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)

        probs = F.sigmoid(inputs)
      
    

        weights_positive = self.weight[0] * targets
        weights_negative = self.weight[1] * (1-targets)

        weights = weights_positive + weights_negative

        max_val = (-inputs).clamp(min=0)
        bce_loss = inputs - inputs * targets + max_val + ((-max_val).exp() + (-inputs - max_val).exp()).log()
        if self.gamma == 0 :
            batch_loss = self.alpha*bce_loss 
        else:
            probs_positive = probs * targets
            probs_negative = (1-probs) * (1-targets)    
            probs_flipped = (1 - (probs_positive + probs_negative))
            batch_loss = self.alpha*(torch.pow(probs_flipped, self.gamma))*bce_loss 

        batch_loss = batch_loss * weights
        
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()

        return loss

class CosineAnealing():
    def __init__(self, initial_lr, itr_per_epoch, cycle_multiplier):
        self.iteration = 0
        self.initial_lr = initial_lr
        self.itr_per_epoch = itr_per_epoch
        self.cycle_multiplier = cycle_multiplier

    def compute_lr(self):
        if self.iteration < self.itr_per_epoch / 20:
            self.iteration += 1
            return self.initial_lr / 100.0

        cos_out = np.cos(np.pi*self.iteration / self.itr_per_epoch) + 1
        self.iteration += 1
        if self.iteration == self.itr_per_epoch:
            self.iteration = 0
            self.itr_per_epoch *= self.cycle_multiplier

        return self.initial_lr / 2 * cos_out