
import matplotlib.pyplot as plt
import numpy as np
from sklearn.kernel_approximation import RBFSampler
from sklearn.preprocessing import MinMaxScaler
from sklearn_tda import *
from sklearn_tda.preprocessing import Clamping

D = np.array([[0.,4.],[1.,2.],[3.,8.],[6.,8.], [0., np.inf], [5., np.inf]])
diags = [D]

diags = DiagramSelector(use=True, point_type="finite").fit_transform(diags)
diags = DiagramScaler(use=True, scalers=[([0,1], MinMaxScaler())]).fit_transform(diags)
diags = DiagramScaler(use=True, scalers=[([1], Clamping(limit=.9))]).fit_transform(diags)

D = diags[0]
plt.scatter(D[:,0],D[:,1])
plt.plot([0.,1.],[0.,1.])
plt.title("Test Persistence Diagram for vector methods")
plt.show()