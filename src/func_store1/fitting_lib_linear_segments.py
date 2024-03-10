import numpy as np
import math
from scipy.optimize import minimize 

def sigmoidEXP(xx):
    return 1/(1+np.exp(-1*xx)) 

def sigmoidEXP_center_width(xx, center, width):
    return sigmoidEXP((xx-center)/width)

def linear_seg3(params, x_vals):
    slope1, node12, width12, slope2, node23, width23, slope3 = params
    predicted_y  = slope1*x_vals
    predicted_y += (slope2*(x_vals-node12) - slope1*(x_vals-node12))*sigmoidEXP_center_width(x_vals, node12, width12)
    predicted_y += (slope3*(x_vals-node23) - slope2*(x_vals-node23))*sigmoidEXP_center_width(x_vals, node23, width23)
    return predicted_y

def cost_linear_seg3(params, x_vals, y_vals):
    predicted_y = linear_seg3(params, x_vals)
    cost = np.mean((predicted_y - y_vals) ** 2)
    return cost

def fit_linear_seg3(initial_params, x_vals, y_vals):
    def objective(params):
        return cost_linear_seg3(params, x_vals, y_vals)
    cons = ({'type': 'ineq', 'fun': lambda x:  x[1] },
            {'type': 'ineq', 'fun': lambda x:  x[4]-x[1] },
            {'type': 'ineq', 'fun': lambda x:  x[0] },
            {'type': 'ineq', 'fun': lambda x:  x[3] },
            {'type': 'ineq', 'fun': lambda x: -x[6] })
    res = minimize(objective, x0=initial_params, constraints=cons )
    return res

