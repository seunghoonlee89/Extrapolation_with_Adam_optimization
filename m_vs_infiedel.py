import pandas as pd
import numpy as np
import h5py

def estimate(m_max): 
    Mm_list = np.array(list(range(20,110,10)) + [5000])
    M_list = Mm_list[:np.where(Mm_list == m_max)[0][0]+1]
    
    def read(fn, M_l):
        ov2_df = pd.read_csv(fn, sep='\t')
        m_full = np.array(list(map(int,list(ov2_df.loc[:,"M"]))))
        ov2 = []
        m = []
        for mm in M_l:
            ov2_tmp = np.array(list(map(float,list(ov2_df.loc[:,"%d"%mm]))))
            idx = np.isnan(ov2_tmp) 
            idx = [not i for i in idx]
            ov2.append(ov2_tmp[idx])
            m.append(m_full[idx])
        return m, ov2
    
    def read_extrapol(m_max):
        f = h5py.File('extrapol_%d.h5' % m_max, 'r')
        m_ex = np.array(list(map(float,list(f.keys()))))
        m_ex.sort()
        ov2_ex = []
        for mm in m_ex:
            ov2_ex.append(np.array(f['%d/yinf' % mm]))
        ov2_ex = np.array(ov2_ex)
        return m_ex[:-1], ov2_ex[:-1]

    logm2 = lambda x: (np.log10(x))**2
    loginf = lambda x: np.log10(1-x)
    expov2 = lambda x: 1 - 10**x
    import matplotlib.pyplot as plt
    m_l, ov2_l = read("m_vs_infiedel.dat", M_list)
    m_5000, ov2_5000 = read("m_vs_infiedel.dat", [5000])
    for it, (im, iov2) in enumerate(zip(m_l, ov2_l)):
        plt.plot(logm2(im), loginf(iov2), '-o', label="$M''=%d$" % (M_list[it])) 
    
    m_ex, ov2_ex = read_extrapol(m_max)

    def estimate(x, y, x0):
        z = np.polyfit(logm2(x), loginf(y), 1)
        p = np.poly1d(z)
        y0 = p(logm2(x0))
        return expov2(y0)

    ov2_max = estimate(m_ex, ov2_ex, m_max)

    plt.plot(logm2(m_5000[0]), loginf(ov2_5000[0]), '--', color='red', label="$M'' = 5000$")
    plt.plot(logm2(m_ex), loginf(ov2_ex), '--o', color='black', label="$M''= \\infty$")
    plt.scatter(logm2(m_max), loginf(ov2_max), marker='x', color='black')

    x1 = np.where(m_5000[0] == m_max)[0]
    print('m_max =', m_max)
    print('ov2_5000, ov2_estimate =', ov2_5000[0][x1], ov2_max)
    print('err =', ((1-ov2_5000[0][x1]) - (1-ov2_max))/(1-ov2_5000[0][x1]) * 100)
    
    plt.legend(loc='upper right')
    #plt.legend(loc='lower left')
    #plt.show()
    #plt.xlim([1.5, 10])
    #plt.ylim([-6, -0.5])
    plt.title("estimate $\\langle \\Psi_0 (%d) | \\Psi_0 (\\infty) \\rangle$" % m_max)
    plt.xlabel("$(logM')^2$")
    plt.ylabel("$log(1-|\\langle \\Psi_0 (M') | \\Psi_0 (M'') \\rangle|^2)$")

    plt.savefig('test_%d.png' % (m_max))
    plt.clf()

M_list = np.array([50, 60, 70, 80, 90, 100])
for m in M_list:
    estimate(m) 

