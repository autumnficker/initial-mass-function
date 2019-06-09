# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 09:30:11 2017

@author: autumnf
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

#==========================================
# VARIABLES
#     x - numpy array of shape (J,2) containing coordinates of each circle
#     rad - numpy array of shape (J) containing radii of each circle
#     minmaxrad - tuple of two floats bounding the minimum and maximum sphere radii
#     maxfillfactor - float between 0 and 1 for the desired percentage of volume at or above which the unit circle is 'filled'
#------------------------------------------
# NOTES
#     minmaxrad - realistic values roughly correspond to (0.0044, 0.096) compared to average molecular cloud mass and the range of possible stellar masses
#     maxfillfactor - works best when within 0.5 - 0.65 (lower values don't fill out lower end of IMF, higher values run too long)
#==========================================

#==========================================
# Gets a uniformly distributed random point
# within a unit circle
def randcircle():
    u = np.random.rand(2)
    r = np.sqrt(u[0])       # Accounts for probability of being within radius r being proportional to r**2
    theta = u[1]*2.0*np.pi
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.array([x,y])

#==========================================
# Checks a potential new point xnew to see
# if it overlaps any existing circles,
# giving a maximum radius to prevent
# overlap for the new circle
def checkpoint(xnew,x,rad,minmaxrad):
    minrad, maxrad = minmaxrad
    overlap = False    
    maxnewrad = maxrad
    d_edge = 1.0 - np.linalg.norm(xnew)
    if d_edge < maxrad:
        maxnewrad = d_edge
    if d_edge < minrad:
        overlap = True
    if not overlap:
        for i in range(rad.size):
            d = xnew-x[i]
            d = np.linalg.norm(d)
            if d - rad[i] < maxnewrad:
                if d - rad[i] > minrad:
                    maxnewrad = d - rad[i]
                else:
                    overlap = True
                    break
    return overlap,maxnewrad

#==========================================
# Adds a single new circle to x and rad
def addcircle(x,rad,minmaxrad):
    minrad, maxrad = minmaxrad
    overlap = True
    xnew = None
    maxnewrad = maxrad
    while overlap:
        xnew = randcircle()
        overlap, maxnewrad = checkpoint(xnew,x,rad,minmaxrad)
    radnew = (maxnewrad-minrad)*np.random.rand()+minrad
    x = np.vstack((x,xnew))
    rad = np.append(rad,radnew)
    return x,rad

#==========================================
# Fills a unit circle to a fill factor
# maxfillfactor with smaller circles,
# returning the coordinates and radii
def fillunitcircle(minmaxrad,maxfillfactor,singletrial=False):
    x = np.zeros((0,2))
    rad = np.zeros(0)
    fillfactor = 0.0
    if singletrial:
        pbar = tqdm()
    while fillfactor < maxfillfactor:
        x,rad = addcircle(x,rad,minmaxrad)
        m = np.power(rad,2.0)
        fillfactor = np.sum(m)
        if singletrial:
            pbar.update(1)
    return x,rad

#==========================================
# Returns results of a single trial
# Prints a log-log fit of mass vs. number of stars
# m is the power of the fit
# b is log10(the coefficient of the power-law fit)
# Figure 1 is a visual of the trials result
# confirming there are no overlapping of
# out of bounds circles
# Figure 2 shows a histogram of the mass
# distribution proportional to radius**2
def singletrial(minmaxrad,maxfillfactor):
    x,rad = fillunitcircle(minmaxrad,maxfillfactor,singletrial=True)
    mass = np.power(rad,2.0)
    logmass = np.log10(mass)
    n,xedges = np.histogram(logmass,bins=100)
    xmiddles = 0.5*(xedges[0:xedges.size-1]+xedges[1:xedges.size])
    logn = np.log10(np.where(n > 1, n, 1))
    nonnan = np.isfinite(logn)
    m,b = np.polyfit(xmiddles[nonnan],logn[nonnan],1)
    print ("m = " + str(m) + "     b = " + str(b))
    
    plt.figure(1,(8,8))
    plt.clf()
    plt.title('Fill Factor  = ' + str(maxfillfactor))
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.gca().add_patch(plt.Circle((0, 0), 1, color='black', linewidth=1, fill=False))
    for i in range(rad.size):
        xi = x[i,0]
        yi = x[i,1]
        radi = rad[i]
        plt.gca().add_patch(plt.Circle((xi, yi),radi,facecolor=tuple(0.3*np.power(np.random.rand(3),1.0/3.0)+0.7),edgecolor='black',linewidth=0.5))
    plt.show()
    
    plt.figure(2)
    plt.clf()
    plt.hist(logmass,bins=100,log=True)
    plt.title('Mass Distribution')
    plt.xlabel('mass')
    plt.ylabel('# of stars')
    plt.show()

#==========================================
# Returns the results of N trials of circle-packing
# Prints average m, b, num_stars over all trials
# Figure 1 shows the trial results in m-b space
# Useful for identifying outliers and trends
def multipletrials(num_trials,minmaxrad,maxfillfactor):
    all_mass = []
    num_stars = []
    m = np.zeros(num_trials)
    b = np.zeros(num_trials)
    for i in tqdm(range(num_trials)):
        _,rad = fillunitcircle(minmaxrad,maxfillfactor)
        mass = np.power(rad,2.0)
        num_stars.append(len(mass))
        all_mass.append(mass)
        logmass = np.log10(mass)
        n,xedges = np.histogram(logmass,bins=100)
        xmiddles = 0.5*(xedges[0:xedges.size-1]+xedges[1:xedges.size])
        logn = np.log10(np.where(n > 1, n, 1))
        nonnan = np.isfinite(logn)
        m_i,b_i = np.polyfit(xmiddles[nonnan],logn[nonnan],1)
        m[i] = m_i
        b[i] = b_i
    print ('Mean m: ' + str(np.mean(m)) + ' +/- ' + str(np.std(m)))
    print ('Mean b: ' + str(np.mean(b)) + ' +/- ' + str(np.std(b)))
    print ('Average number of stars: ' + str(np.mean(num_stars)) + ' +/- ' + str(np.std(num_stars)))

    plt.figure(1)
    plt.clf()
    plt.plot(10**b,m,'.')
    plt.title('IMF Trials')
    plt.xlabel('Coefficient (10**b)')
    plt.ylabel('Power (m)')
    plt.show()



def main():
    N = 10                 # SET THIS
    maxfillfactor = 0.65     # SET THIS
    minrad = 0.0044         # SET THIS
    maxrad = 0.096          # SET THIS
    minmaxrad = minrad,maxrad
    #multipletrials(N,minmaxrad,maxfillfactor)
    singletrial(minmaxrad,maxfillfactor)
    

main()
