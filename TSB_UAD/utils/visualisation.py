from statsmodels.tsa.stattools import acf
from scipy.signal import argrelextrema
import numpy as np

import matplotlib.patches as mpatches 
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox

from ..vus.utils.metrics import metricor

def plotFig(data, label, score, slidingWindow, fileName, modelName, plotRange=None):
    grader = metricor()

    R_AUC, R_AP, R_fpr, R_tpr, R_prec = grader.RangeAUC(labels=label, score=score, window=slidingWindow, plot_ROC=True) #

    L, fpr, tpr = grader.metric_new(label, score, plot_ROC=True)
    precision, recall, AP = grader.metric_PR(label, score)
    
    range_anomaly = grader.range_convers_new(label)
    
    max_length = len(score)

    if plotRange==None:
        plotRange = [0, max_length]
    
    # plt.plot(data[:max_length], 'k')
    
    fig3 = plt.figure(figsize=(12, 10), constrained_layout=True)
    gs = fig3.add_gridspec(3, 4)
       
    f3_ax1 = fig3.add_subplot(gs[0, :-1])
    plt.tick_params(labelbottom=False)

    plt.plot(data[:max_length],'k')
    for r in range_anomaly:
        if r[0]==r[1]:
            plt.plot(r[0],data[r[0]],'r.')
        else:
            plt.plot(range(r[0],r[1]+1),data[range(r[0],r[1]+1)],'r')
    
    plt.xlim(plotRange)
    
    f3_ax2 = fig3.add_subplot(gs[1, :-1])
    L1 = [ '%.2f' % elem for elem in L]
    plt.plot(score[:max_length])
    plt.hlines(np.mean(score)+3*np.std(score),0,max_length,linestyles='--',color='red')
    plt.ylabel('score')
    
    plt.xlim(plotRange)
        
    # plot the data
    f3_ax3 = fig3.add_subplot(gs[2, :-1])
    index = ( label + 2*(score > (np.mean(score)+3*np.std(score))))
    cf = lambda x: 'k' if x==0 else ('r' if x == 1 else ('g' if x == 2 else 'b') )
    cf = np.vectorize(cf)
    
    color = cf(index[:max_length])
    black_patch = mpatches.Patch(color = 'black', label = 'TN')
    red_patch = mpatches.Patch(color = 'red', label = 'FN')
    green_patch = mpatches.Patch(color = 'green', label = 'FP')
    blue_patch = mpatches.Patch(color = 'blue', label = 'TP')
    plt.scatter(np.arange(max_length), data[:max_length], c=color, marker='.')
    plt.legend(handles = [black_patch, red_patch, green_patch, blue_patch], loc= 'best')
    plt.xlim(plotRange)

    f3_ax4 = fig3.add_subplot(gs[0, -1])
    plt.plot(fpr, tpr)
    plt.plot(R_fpr,R_tpr)
    plt.title('R_AUC='+str(round(R_AUC,3))+', AUC='+str(L1[0]))
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend(['ROC','Range-ROC'])

    extent = full_extent(f3_ax1).transformed(fig3.dpi_scale_trans.inverted())
    fig3.savefig(fileName+'_'+modelName+'_scoring_plot_figure.png', bbox_inches=extent)

    plt.suptitle(fileName + '    window='+str(slidingWindow) +'   '+ modelName
    +'\nAUC='+L1[0]+'     R_AUC='+str(round(R_AUC,2))+'     Precision='+ L1[1]+ '     Recall='+L1[2]+'     F='+L1[3]
    + '     ExistenceReward='+L1[5]+'   OverlapReward='+L1[6]
    +'\nAP='+str(round(AP,2))+'     R_AP='+str(round(R_AP,2))+'     Precision@k='+L1[9]+'     Rprecision='+L1[7] + '     Rrecall='+L1[4] +'    Rf='+L1[8]
    )
    

def full_extent(ax, pad=0.0):
    """Get the full extent of an axes, including axes labels, tick labels, and titles."""
    # For text objects, we need to draw the figure first, otherwise the extents are undefined.
    ax.figure.canvas.draw()
    items = ax.get_xticklabels() + ax.get_yticklabels() 
    # items += [ax, ax.title, ax.xaxis.label, ax.yaxis.label]
    items += [ax, ax.title]
    bbox = Bbox.union([item.get_window_extent() for item in items])

    return bbox.expanded(1.0 + pad, 1.0 + pad)  