import openturns as ot
import numpy as np
sample_E = ot.NumericalSample.ImportFromCSVFile("sample_E.csv") 
kernel_smoothing = ot.KernelSmoothing(ot.Normal())
bandwidth = kernel_smoothing.computeSilvermanBandwidth(sample_E)
E = kernel_smoothing.build(sample_E, bandwidth)
E.setDescription(['Young modulus'])

F = ot.LogNormal(30000, 9000, 15000, ot.LogNormal.MUSIGMA)
F.setDescription(['Load'])

L = ot.Uniform(250, 260)
L.setDescription(['Length'])

I = ot.Beta(2.5, 4, 310, 450)
I.setDescription(['Inertia'])

marginal_distributions = [F, E, L, I]

SR_cor = ot.CorrelationMatrix(len(marginal_distributions))
SR_cor[2, 3] = -0.2
copula = ot.NormalCopula(ot.NormalCopula.GetCorrelationFromSpearmanCorrelation(SR_cor))

X_distribution = ot.ComposedDistribution(marginal_distributions, copula)