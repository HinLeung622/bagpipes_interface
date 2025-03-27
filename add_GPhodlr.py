import bagpipes as pipes
import george
from george import kernels

# adding a GP noise option that uses the (possibly) faster hodlr solver into bagpipes.noise.py

def GP_exp_squared_hodlr(self):
    """ A GP noise model including an exponenetial squared kernel
    for corellated noise and white noise (jitter term). """

    scaling = self.param["scaling"]

    norm = self.param["norm"]
    length = self.param["length"]

    kernel = norm**2*kernels.ExpSquaredKernel(length**2)
    self.gp = george.GP(kernel)
    self.gp.compute(self.x, self.y_err*scaling)

    self.corellated = True

pipes.fitting.noise.noise_model.GP_exp_squared_hodlr = GP_exp_squared_hodlr
