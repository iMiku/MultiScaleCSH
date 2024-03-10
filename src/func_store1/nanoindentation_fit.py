import numpy as np
from scipy.signal import savgol_filter, find_peaks
from scipy.optimize import minimize
from functools import partial
import math

def oliver_pharr(x, alpha, hp, m):
    return alpha*np.float_power((x-hp),m)

def oliver_pharr_cost(params, xdata, ydata):
    yfit  = oliver_pharr(xdata, params[0], params[1], params[2])
    cost  = np.sum(np.float_power(ydata - yfit, 2.0))
    return cost

class md_nanoindentation_curve(object):
    def __init__(self, displacement, load, unload_start_id=-1, max_depth=-1):
        assert len(displacement) == len(load)
        self.displacement = displacement  # float
        self.load = load  # float
        self.load_hat = savgol_filter(load, 101, 3)
        self.unload_start_id = unload_start_id
        if(unload_start_id<0):
            self.unload_start_id = self.find_unload_start()
        self.oliver_pharr_params = self.fit_oliver_pharr()
        self.unload_disp = self.displacement[self.unload_start_id:]
        self.unload_fit = oliver_pharr(self.unload_disp, self.oliver_pharr_params[0],
                                                         self.oliver_pharr_params[1],
                                                         self.oliver_pharr_params[2])
        alpha = self.oliver_pharr_params[0]
        hp    = self.oliver_pharr_params[1]
        m_val = self.oliver_pharr_params[2]
        self.stiffness = m_val*alpha*(self.unload_disp[0] - hp)**(m_val - 1)
        self.max_depth = max_depth
        if(max_depth<0):
            self.max_depth = self.displacement[unload_start_id] - self.displacement[0]
    
    def find_unload_start(self):
        peaks, properties = find_peaks(self.load_hat, prominence=100)
        prominence = properties["prominences"]
        selected_peak = peaks[np.argmax(prominence)]
        return selected_peak

    def fit_oliver_pharr(self):  
        xdata=self.displacement[self.unload_start_id:]
        ydata=self.load_hat[self.unload_start_id:]
        op_func = partial(oliver_pharr_cost, xdata=xdata, ydata=ydata)
        params0 = [100, 3.0, 2.0]
        bounds0 = ((None, None),(None, xdata[0]),(0.0,5.0))
        res = minimize(op_func, params0, method='Nelder-Mead', bounds=bounds0)
        return res.x

    def contact_area_calc(self):
        berkovich_area = 3*math.sqrt(3)*(math.tan(65.3/180*math.pi)*self.max_depth)**2
        return berkovich_area

    def reduced_modulus_calc(self):
        E_r = math.sqrt(math.pi)/2.0/math.sqrt(self.contact_area_calc())*self.stiffness
        return E_r
