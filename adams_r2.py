"""
 Gradient Descent Algorithm with Adam Optimization Algorithm:
 to Minimize 1 - r**2 Cost Function
"""

import pandas as pd
import numpy as np
import h5py

def cal_res(xi, yi, A, B, yinf):
    return A * xi + B - np.log(yi - yinf)

def cal_ssr(X, Y, A, B, yinf):
    cost = 0.
    for xi, yi in zip(X, Y):
        cost += cal_res(xi, yi, A, B, yinf)**2
    return cost

def cal_sst(Y, yinf):
    ymean = np.average(np.log(Y - yinf))
    sst = 0.
    for yi in Y:
        sst += (np.log(yi - yinf) - ymean)**2
    return sst

def cal_r2(X, Y, A, B, yinf):
    sst = cal_sst(Y, yinf)
    ssr = cal_ssr(X, Y, A, B, yinf)
    r2 = 1. - ssr / sst
    return r2 

def cal_grad(X, Y, A, B, yinf):
    sst = cal_sst(Y, yinf)
    ssr = cal_ssr(X, Y, A, B, yinf)

    grad = np.zeros((3))
    tmp = np.zeros((3))
    for xi, yi in zip(X, Y):
        res = cal_res(xi, yi, A, B, yinf) * 2.
        tmp[0] += res * xi 
        tmp[1] += res 
        tmp[2] += res / (yi - yinf) 
    grad = tmp / sst

    d1 = 0.
    d2 = 0.
    d3 = 0.
    for xi, yi in zip(X, Y):
        tmp1 = np.log(yi - yinf)
        tmp2 = 1. / (yi - yinf)
        d1 += tmp1
        d2 += tmp2
        d3 += tmp1 * tmp2 
    tmp = -2. * (d3 - d1 * d2 / len(X))
    grad[2] -= tmp * ssr / (sst)**2 
    return grad 

def gradientDescent(X, Y, A, B, yinf, alpha0=(1e-2, 1e-2, 1e-3), decay_rate = 1.,
                    n_epoch = 100000, beta = (0.9, 0.999), eps = 1e-8, 
                    maxit=100000000, conv_cst=1e-16, conv_grad_thrsh=1e-8):
    # beta[0]: momentum hyperparameter
    # beta[1]: RMSprop hyperparameter
    conv = False
    cst_hist = []
    print_it = n_epoch 
    #print_it = 1
    V = np.zeros((3))   # exponentially weighted average for gradient
    S = np.zeros((3))   # exponentially weighted average for square of gradient
    alpha = np.zeros((3)) 
    alpha0 = np.array(list(alpha0)) 
    for i in range(0, maxit):
        cst_hist.append(cal_r2(X, Y, A, B, yinf))
        # Adam optimization algorithm
        grad = cal_grad(X, Y, A, B, yinf)
        V = beta[0] * V + (1. - beta[0]) * grad
        Vcor = V / (1. - beta[0] ** (i+1))
        S = beta[1] * S + (1. - beta[1]) * np.square(grad)
        Scor = S / (1. - beta[1] ** (i+1))
        # learning rate decay
        alpha = alpha0 / (1. + decay_rate * i // n_epoch) 
        # update params
        A -= alpha[0] * Vcor[0] / np.sqrt(Scor[0])
        B -= alpha[1] * Vcor[1] / np.sqrt(Scor[1])
        yinf -= alpha[2] * Vcor[2] / np.sqrt(Scor[2])
        # check convergence
        r2 = cal_r2(X, Y, A, B, yinf)
        if i % print_it == 0:
            print("Iteration %10d | Cost: %20.16f | Grad(A, B, yinf): %12.8f, %12.8f, %12.8f | r2: %20.16f | yinf: %f"
                   % (i, 1. - cst_hist[-1], *grad, r2, yinf), flush=True)
        if np.isnan(r2):
            break
        if i > 1 and (np.linalg.norm(grad) < conv_grad_thrsh 
                   or np.abs(cst_hist[-1] - cst_hist[-2]) < conv_cst):
            print("Iteration %10d | Cost: %20.16f | Grad(A, B, yinf): %12.8f, %12.8f, %12.8f | r2: %20.16f | yinf: %f"
                   % (i, 1. - cst_hist[-1], *grad, r2, yinf), flush=True)
            print("Converged! | Norm grad: %15.8f | Delta Cost: %20.16f" %
                  (np.linalg.norm(grad), np.abs(cst_hist[-1] - cst_hist[-2])))
            conv = True
            break 
    return yinf, A, B, r2, conv

Mm_list = np.array(list(range(10,110,10)) + [5000])
def read(fn, m_max):
    ov2_df = pd.read_csv(fn, sep='\t')
    m_full = np.array(list(map(int,list(ov2_df.loc[:,"M"]))))
    ov2 = []
    m = []
    M_list_tmp = Mm_list[:np.where(Mm_list == m_max)[0][0]]
    for mm in M_list_tmp:
        ov2_tmp = np.array(list(map(float,list(ov2_df.loc[:,"%d"%mm]))))
        idx = np.isnan(ov2_tmp) 
        idx = [not i for i in idx]
        ov2_tmp = ov2_tmp[idx]
        m_full_tmp = m_full[idx]
        idx = np.where(m_full_tmp == m_max)[0][0]
        ov2.append(ov2_tmp[:idx+1])
        m.append(m_full_tmp[:idx+1])
    return m, ov2

rst = False 
import sys
M_list = [int(sys.argv[1])]
for m_max in M_list:
    print('M max: %d extrapol' % m_max, flush=True)
    m_l, ov2_l = read("ov2_revdmrg.dat", m_max)
    m_l = m_l[:-1]
    ov2_l = ov2_l[:-1]
    f = h5py.File('extrapol_%d.h5' % m_max, 'w')
    for it, (m, ov2) in enumerate(zip(m_l, ov2_l)):
        logm2 = (np.log(m))**2
        if not rst:
            yinf_init = ov2[-1] + (ov2[-1] - ov2[-2])*.5
    
            #A_init = (np.log(ov2[-1] - yinf_init) - np.log(ov2[-2] - yinf_init)) / (logm2[-1] - logm2[-2]) 
            #B_init = np.log(ov2[-1] - yinf_init) - A_init * logm2[-1] 
            model = np.polyfit(logm2, np.log(ov2 - yinf_init), 1)
            A_init, B_init = model[0], model[1]
        else:
            if it == 0:
                yinf_init, A_init, B_init = 0.6445419612701339, -0.14905411438187793, -0.1495646839271265 
#                yinf_init, A_init, B_init = 0.646025302608789, -0.15112905306068233, -0.13639681976626813 
#                yinf_init, A_init, B_init = 0.6463226005773193, -0.1513087508924492, -0.13681157767561386 
#                yinf_init, A_init, B_init = 0.6402534532649866, -0.14358015067632907, -0.18413095427979098 
#                yinf_init, A_init, B_init = 0.633823640361056, -0.1366887432608472, -0.2215535613627002
#                yinf_init, A_init, B_init = 0.6074073053068436, -0.11541835604601998, -0.31265548874465976
            elif it == 1: 
                yinf_init, A_init, B_init = 0.8247090937904641, -0.11320096472729721, -0.6590808726119607 
#                yinf_init, A_init, B_init = 0.8258007700487726, -0.1143642200556692, -0.6537342073044532 
#                yinf_init, A_init, B_init = 0.8253057901070122, -0.11345276155775098, -0.661733190997545 
#                yinf_init, A_init, B_init = 0.8276719435697173, -0.11570952038503575, -0.6534571248854372 
#                yinf_init, A_init, B_init = 0.8357409220280732, -0.12456296538593822, -0.6125208425964861
#                yinf_init, A_init, B_init = 0.8279957003694663, -0.11381412003656453, -0.6811580054975824
            elif it == 2:
                yinf_init, A_init, B_init = 0.881779500561344, -0.07602556918308484, -1.2463509582352061 
#                yinf_init, A_init, B_init = 0.8871291763657325, -0.08090706779243573, -1.2339364694301271 
#                yinf_init, A_init, B_init = 0.892937266948078, -0.08671188276988366, -1.2172904409752008
#                yinf_init, A_init, B_init = 0.9015110738087895, -0.09685352503376415, -1.1787692772397333 
#                yinf_init, A_init, B_init = 0.9295184192817186, -0.1563376853581377, -0.785567031187942
#                yinf_init, A_init, B_init = 0.9465755280000002, -0.2390583884194129, -0.03978596473567402
            elif it == 3:
                yinf_init, A_init, B_init = 0.9375190173143897, -0.09230206367846505, -1.4763277970036757 
#                yinf_init, A_init, B_init = 0.9436440548182955, -0.10314345492197409, -1.4312655879310192 
#                yinf_init, A_init, B_init = 0.9512201160440833, -0.12002037920327228, -1.3468455831888595
#                yinf_init, A_init, B_init = 0.9599236410044852, -0.14996524679927845, -1.1311052289444483 
#                yinf_init, A_init, B_init = 0.9840142644999998, -0.7526115996737668, 6.690233091998757
#            elif it == 4:
#                yinf_init, A_init, B_init = 0.9647749570242755, -0.12283503718725859, -1.4014909592788283 
#                yinf_init, A_init, B_init = 0.9700006559014779, -0.14449888544596617, -1.2296454879234244 
#                yinf_init, A_init, B_init = 0.9764206077000217, -0.18288214482158563, -0.886445766977892
#                yinf_init, A_init, B_init = 0.9898802405, -0.8542502285754974, 9.143395311190824
#            elif it == 5:
#                yinf_init, A_init, B_init = 0.97973187683912, -0.16956135191994545, -0.9796940249302839 
#                yinf_init, A_init, B_init = 0.9839885601850435, -0.21733284698625915, -0.4078383244221441 
#                yinf_init, A_init, B_init = 0.993319767, -0.9532877592133144, 11.586365955751234
#            elif it == 6:
#                yinf_init, A_init, B_init = 0.9886993107969986, -0.25453274978969276, 0.1900382749303751
#                yinf_init, A_init, B_init = 0.9949450375000002, -1.0501683962907187, 14.268474488404914 
#            elif it == 7:
#                yinf_init, A_init, B_init = 0.9958971475, -1.1452164450595612, 17.101420955914843
            else:
                assert False 

        r2_init = cal_r2(logm2, ov2, A_init, B_init, yinf_init)
        print('M = %d' % Mm_list[it], flush=True)
        print('init yinf, A, B, r2 =', yinf_init, A_init, B_init, r2_init, flush=True)
        yinf, A, B, r2, conv = gradientDescent(logm2, ov2, A_init, B_init, yinf_init) 
        print('conv =', conv, flush=True)
        print('final yinf, A, B, r2', yinf, A, B, r2, flush=True)
        g = f.create_group('%d' % (Mm_list[it]))
        g.create_dataset('yinf', data=yinf)
        g.create_dataset('A', data=A)
        g.create_dataset('B', data=B)
        g.create_dataset('r2', data=r2)
        import matplotlib.pyplot as plt
        plt.scatter(logm2, np.log(ov2 - yinf)) 
        x = np.arange(logm2[0], logm2[-1], 1)
        func = lambda x: A * x + B 
        y = func(x)
        plt.plot(x, y, '--', color='black') 
        plt.savefig('%d_%d.png' % (Mm_list[it], m_max))
    f.close()
        
