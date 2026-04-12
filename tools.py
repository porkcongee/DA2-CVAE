import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import numpy as np
import os

def load_data_1d(filenames, idx=0, type=float, indices=None):
    All_data = []
    if isinstance(filenames, str):
        filenames = [filenames]
    if indices:
        if isinstance(indices, str):
            indices = [indices]
        if len(filenames) != len(indices):
            raise ValueError('The number of filenames and indices should be the same!')
    for index, filename in enumerate(filenames):
        data = []
        isExistfile = os.path.exists(filename)
        
        if isExistfile:
            with open(filename, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    data.append(line.split()[idx])
            print(filename+' is loaded!')
        else:
            raise FileNotFoundError(f'{filename} not exist!')
        data = np.array(data, dtype=type)
        if indices:
            indice = np.load(indices[index])
            data = data[indice]
        All_data.extend(data)
    return np.array(All_data)

def load_data_2d(filenames, idx=[0,1] ,type=float, indices=None):
    All_data1 = []
    All_data2 = []
    if isinstance(filenames, str):
        filenames = [filenames]
    if indices:
        if isinstance(indices, str):
            indices = [indices]
        if len(filenames) != len(indices):
            raise ValueError('The number of filenames and indices should be the same!')
    for index, filename in enumerate(filenames):
        data1 = []
        data2 = []
        isExistfile = os.path.exists(filename)
        
        if isExistfile:
            with open(filename, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    data1.append(line.split()[idx[0]])
                    data2.append(line.split()[idx[1]])
            print(filename+' is loaded!')
        else:
            raise FileNotFoundError(f'{filename} not exist!')
        data1 = np.array(data1, dtype=type)
        data2 = np.array(data2, dtype=type)
        if indices:
            indice = np.load(indices[index])
            data1 = data1[indice]
            data2 = data2[indice]
        All_data1.extend(data1)
        All_data2.extend(data2)
    return np.array(All_data1), np.array(All_data2)


def  plt_par(ax, **kwarg):
    if 'label' in kwarg:
        label = kwarg['label']
        ax.set_xlabel(label[0], fontsize=20)
        ax.set_ylabel(label[1], fontsize=20)
    if 'title' in kwarg:
        ax.set_title(kwarg['title'],fontsize=20)
    if 'tick' in kwarg:
        tick = kwarg['tick']
        ax.set_xticks(tick[0])
        ax.set_xticklabels(tick[1])
        ax.set_yticks(tick[2])
        ax.set_yticklabels(tick[3])
    if 'xtick' in kwarg:
        xtick = kwarg['xtick']
        ax.set_xticks(xtick[0])
        ax.set_xticklabels(xtick[1])
    if 'ytick' in kwarg:
        ytick = kwarg['ytick']
        ax.set_yticks(ytick[0])
        ax.set_yticklabels(ytick[1])
    return ax


def plt_1d(ax,X, **kwarg):
    if 'plotlabel' in kwarg:
        plotlabel = kwarg['plotlabel']
        ax.plot(X,label=plotlabel)
    else:
        ax.plot(X)
    ax = plt_par(ax,**kwarg)
    return ax


def plt_2d(ax,X,Y, **kwarg):
    if 'Point' in kwarg:
        ax.scatter(X,Y, c=kwarg['Point'],s=100,marker=(5,1))  # type: ignore
    else:
        ax.scatter(X,Y, c=range(len(X)),cmap='OrRd')
        ax.scatter(X[0],Y[0], c='b',s=100,marker=(5,1))  # type: ignore
        ax.scatter(X[-1],Y[-1], c='k',s=100,marker=(5,1))  # type: ignore
    if 'cb' in kwarg:
        cb = cm.ScalarMappable(cmap='OrRd')
        cb.set_array(range(len(X)))
        plt.colorbar(cb,label=kwarg['cb'])
    ax = plt_par(ax,**kwarg)
    return ax


def plt_2dhist(ax,fig, X,Y, **kwarg):
    hb = ax.hist2d(X, Y, bins=100, cmap='jet', density=False, norm=colors.LogNorm())
    cb = fig.colorbar(hb[3], ax=ax, label='counts')
    ax = plt_par(ax,**kwarg)
    return fig, ax

if __name__ == '__main__':
    pass