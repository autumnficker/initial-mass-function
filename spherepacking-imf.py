# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 13:17:57 2017

@author: autumnf
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from tqdm import tqdm

#==========================================
# VARIABLES
#     x - numpy array of shape (J,3) containing coordinates of each sphere
#     r - numpy array of shape (J) containing radii of each sphere
#     minmaxr - tuple of float bounding the minimum and maximum sphere radii
#     maxfillfactor - float for the desired fill factor, determining when the unit sphere is 'filled'
#     N - number of trials, each trial being the filling of a unit sphere to maxfillfactor
#------------------------------------------
# NOTES
#     minmaxr - realistic values roughly correspond to (0.027, 0.21) compared to average molecular cloud mass and the range of possible stellar masses
#     maxfillfactor - works best when within 0.2 - 0.3 (lower values don't fill out lower end of IMF, higher values run very long)
#     N - a minimum of 20 trials is recommended to get reliable results; use N = 1 to see a 3D projection of the trial
#==========================================

#==========================================
# Gets a uniformly distributed random point
# within a unit sphere
def randsphere():
    u = np.random.rand(3)
    r = np.power(u[0],1.0/3.0)      # Accounts for probability of being within radius r being proportional to r**3
    theta = np.arccos(2.0*u[1]-1.0) # Accounts for larger probability of being near "equator" than being near "poles"
    phi = u[2]*2.0*np.pi
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.array([x,y,z])

#==========================================
# Checks a potential new point xnew to see
# if it overlaps any existing spheres,
# giving a maximum radius to prevent
# overlap for the new sphere
def checkpoint(xnew,x,r,d,minmaxr):
    minr, maxr = minmaxr
    overlap = False    
    maxnewr = maxr
    d_edge = 1.0 - np.linalg.norm(xnew)
    if d_edge < maxr:
        maxnewr = d_edge
    if d_edge < minr:
        overlap = True
    # TODO: Edit code under here to optimize
    j = 0
    shell = np.argwhere(np.logical_and(d>1.0-d_edge-maxr,d<1.0-d_edge+maxr))
    while j < shell.size and not overlap:
        i = int(shell[j])
        dist = xnew-x[i]
        dist = np.linalg.norm(dist)
        if dist - r[i] < maxnewr and dist - r[i] > 0.0:
            maxnewr = dist - r[i]
        if dist - r[i] - minr < 0.0:
            overlap = True
        j += 1
    return overlap,maxnewr

#==========================================
# Adds a single new sphere to x and r
def addsphere(x,r,d,minmaxr):
    minr, maxr = minmaxr
    overlap = True
    xnew = None
    maxnewr = maxr
    while overlap:
        xnew = randsphere()
        overlap, maxnewr = checkpoint(xnew,x,r,d,minmaxr)
    rnew = (maxnewr-minr)*np.random.rand()+minr
    dnew = np.linalg.norm(xnew)
    x = np.append(x,xnew)
    x = x.reshape(int(x.size/3),3)
    r = np.append(r,rnew)
    d = np.append(d,dnew)
    return x,r,d

#==========================================
# Fills a unit sphere to a fill factor
# maxfillfactor with smaller spheres,
# returning the coordinates and radii
def fillunitsphere(minmaxr,maxfillfactor):
    x = np.zeros((0,3))
    r = np.zeros(0)
    d = np.zeros(0)
    fillfactor = 0.0
    while fillfactor < maxfillfactor:
        x,r,d = addsphere(x,r,d,minmaxr)
        m = np.power(r,3.0)
        fillfactor = np.sum(m)
    return x,r,d

def main():
    N = 1                   # CHOOSE THIS
    maxfillfactor = 0.3     # CHOOSE THIS
    minr = 0.027            # CHOOSE THIS
    maxr = 0.21             # CHOOSE THIS
    minmaxr = minr,maxr
    x = np.zeros((0,3))
    r = np.zeros(0)
    d = np.zeros(0)
    for i in tqdm(range(N)):
        xnew,rnew,dnew = fillunitsphere(minmaxr,maxfillfactor)
        x = np.append(x,xnew)
        x = x.reshape(int(x.size/3),3)
        r = np.append(r,rnew)
        d = np.append(d,dnew)
    mass = np.power(r,3.0)
    logm = np.log10(mass)
    n,xedges = np.histogram(logm,bins=100)
    xmiddles = 0.5*(xedges[0:xedges.size-1]+xedges[1:xedges.size])
    logn = np.log10(n)
    nonnan = np.isfinite(logn)
    m,b = np.polyfit(xmiddles[nonnan],logn[nonnan],1)
    
    # Display histogram of mass over all trials
    plt.figure(1)
    plt.clf()
    plt.hist(logm,bins=100,log=True)
    plt.title('Mass Distribution (N = ' + str(N) + ', fill = ' + str(maxfillfactor) + ')')
    plt.xlabel('mass')
    plt.ylabel('# of stars')
    plt.show()
    
    # Display 3D projection of trial if only one trial is executed
    # WARNING: Runs slow for high maxfillfactor
    if N == 1:
        fig = plt.figure(2)
        plt.clf()
        ax = fig.gca(projection='3d')
        u,v = np.mgrid[0.0:np.pi:30j, 0.0:2.0*np.pi:30j]
        for i in range(r.size):
            x_ = x[i,0] + r[i]*np.cos(u)*np.sin(v)
            y_ = x[i,1] + r[i]*np.sin(u)*np.sin(v)
            z_ = x[i,2] + r[i]*np.ones_like(u)*np.cos(v)
            surf = ax.plot_surface(x_,y_,z_,color=tuple(np.random.rand(3)),linewidth=0,antialiased=True)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

    print ('m = ' + str(m) + '    b = '  +str(b))
    
main()