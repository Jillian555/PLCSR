class WeightEMA (object):
    """
    Exponential moving average weight optimizer for mean teacher model
    """
    #teacher para  student para
    def __init__(self, params, src_params, alpha=0.999):
        self.params = list(params)
        self.src_params = list(src_params)
        self.alpha = alpha

        for p, src_p in zip(self.params, self.src_params):
            p.data[:] = src_p.data[:]

    def step(self):
        #t=t*α+s*(1-α)
        one_minus_alpha = 1.0 - self.alpha
        for p, src_p in zip(self.params, self.src_params):
            p.data.mul_(self.alpha)
            p.data.add_(src_p.data * one_minus_alpha)
            

class WeightEMA_reverse (object):
    """
    Exponential moving average weight optimizer for mean teacher model
    """
    def __init__(self, params, src_params, alpha=0.999):
        self.params = list(params)
        self.src_params = list(src_params)
        self.alpha = alpha

        for p, src_p in zip(self.params, self.src_params):
            p.data[:] = src_p.data[:]

    def step(self):
        one_minus_alpha = self.alpha - 1.0 
        k = 1.0 / self.alpha
        for p, src_p in zip(self.params, self.src_params):
            p.data.add_(src_p.data * one_minus_alpha)
            p.data.mul_(k)