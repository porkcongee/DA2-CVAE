import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from tools import load_data_1d, load_data_2d
from collections import Counter
def get_counts(recon_ktrajs, recon_kdist, dist_weight=True):
    if dist_weight:
        counts = {}
        for c, d in zip(recon_ktrajs, recon_kdist):
            if c in counts:
                counts[c] += 1/d[c]
            else:
                counts[c] = 1/d[c]
    else:
        counts = Counter(recon_ktrajs)
    print(counts)
    return counts
    
def get_weights(counts, n_clusters, S, task, dist_weight=True):
    weight = []
    if dist_weight:
        counts_array = np.array([counts[i] for i in range(n_clusters)])
        ktrajs = np.load(f'{task}/kmeans_{task}_n{n_clusters}.npy')
        kdist = np.load(f'{task}/kdist_{task}_n{n_clusters}.npy')
        for c in range(n_clusters):
            c_file = f'{task}/cluster_n{n_clusters}_{c}_{S}.npy'
            c_select = np.load(c_file)
            c_list = np.where(ktrajs == c)
            c_kdist = kdist[c_list[0],:][c_select,:]
            c_dist_weight = 1 / c_kdist
            c_weight = c_dist_weight @ counts_array
            weight.extend(c_weight)
    else:
        for c in range(n_clusters):
            weight.extend([counts[c]] * S)
    return weight

def plot_kmeans_cluster_test(task):
    inertia = load_data_1d(f'{task}/kmeans_inertia.dat')
    fig, ax = plt.subplots()
    ax.ticklabel_format(style='sci', scilimits=(-1,2),axis='y')
    plt.plot(list(range(2,11)),inertia, color='#33ABC1')
    plt.xlabel('N_clusters')
    plt.ylabel('Inertia')
    plt.tight_layout()
    plt.savefig(f'{task}/fig/kmeans_cluster_test')

def plot_kmeans_raw(task, n_clusters, reduced, ktrajs, my_listedcmap):
    X = 0
    Y = 1
    fig ,ax = plt.subplots()
    sc = plt.scatter(reduced[::1,X],reduced[::1,Y], marker='o', c=ktrajs[::1], cmap=my_listedcmap,alpha=0.05)
    ax.set_xlabel(f'PC{X+1}')
    ax.set_ylabel(f'PC{Y+1}')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    cbar = fig.colorbar(cm.ScalarMappable(norm=mcolors.Normalize(1, n_clusters), cmap=my_listedcmap),ax=ax,ticks=range(1,n_clusters+1))
    cbar.mappable.set_clim(0.5,n_clusters+0.5)
    # plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{task}/fig/kmeans_raw_{task}_n{n_clusters}')

def plot_kmeans_train(task, n_clusters, reduced, ktrajs, my_listedcmap, my_cmap_list, S):
    X = 0
    Y = 1
    fig ,ax = plt.subplots()
    plt.hist2d(reduced[:, X], reduced[:,Y], bins=100, cmap='Greys', norm=mcolors.LogNorm())
    for n in range(n_clusters):
        c_file = f'{task}/cluster_n{n_clusters}_{n}_{S}.npy'
        index = np.load(c_file)
        c_reduced = reduced[np.where(ktrajs == n)[0],:]
        sc = plt.scatter(c_reduced[index,X],c_reduced[index,Y], marker='o', c=my_cmap_list[n],alpha=0.05)
    ax.set_xlabel(f'PC{X+1}')
    ax.set_ylabel(f'PC{Y+1}')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    cbar = fig.colorbar(cm.ScalarMappable(norm=mcolors.Normalize(1, n_clusters), cmap=my_listedcmap),ax=ax,ticks=range(1,n_clusters+1))
    cbar.mappable.set_clim(0.5,n_clusters+0.5)
    # plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{task}/fig/kmeans_train_{task}_n{n_clusters}')

def plot_kmeans_recon(task, task_name, n_clusters, reduced, recon_reduced, recon_ktrajs, my_listedcmap, my_cmap_list):
    X = 0
    Y = 1
    fig ,ax = plt.subplots()
    plt.hist2d(reduced[:, X], reduced[:,Y], bins=100, cmap='Greys', norm=mcolors.LogNorm())
    for n in range(n_clusters):
        index = np.where(recon_ktrajs == n)
        sc = plt.scatter(recon_reduced[index,X],recon_reduced[index,Y], marker='o', c=my_cmap_list[n],alpha=0.05)
    ax.set_xlabel(f'PC{X+1}')
    ax.set_ylabel(f'PC{Y+1}')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    cbar = fig.colorbar(cm.ScalarMappable(norm=mcolors.Normalize(1, n_clusters), cmap=my_listedcmap),ax=ax,ticks=range(1,n_clusters+1))
    cbar.mappable.set_clim(0.5,n_clusters+0.5)
    # plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{task}/fig/kmeans_recon_{task_name}')

def plot_recon_convergence(task, task_name, n_clusters, recon_ktrajs, my_cmap_list):
    C = {}
    for s in [10000,20000,30000,40000,50000]:
        counts = Counter(recon_ktrajs[:s])
        frequency = np.array(list(counts.values()))
        SUM = frequency.sum()
        for f in range(n_clusters):
            if f in C:
                C[f].append(counts[f]/SUM)
            else:
                C[f] = [counts[f]/SUM]
    categories=['10k', '20k', '30k', '40k', '50k']
    fig ,ax = plt.subplots()
    plt.bar(categories, C[0], label='c0', color=my_cmap_list[0])
    B = np.array(C[0])
    for i in range(1,n_clusters):
        plt.bar(categories, C[i], bottom=B, label=f'c{i}', color=my_cmap_list[i])
        B += np.array(C[i])
    plt.ylabel('Proportion')
    plt.xlabel('R-set Size')
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{task}/fig/recon_convergence_{task_name}')

def plot_loss(task, task_name):
    def plot_l(loss, ylabel):
        recon = np.loadtxt(f'{task}/loss/{loss}_{task_name}.txt')
        fig, ax = plt.subplots()
        plt.plot(recon,color='#33ABC1')
        plt.xlabel('Epoch')
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(f'{task}/fig/loss_{loss}_{task_name}')
    for loss, ylabel in [('recon', 'Recon Loss'), ('kl', 'KL Loss'), ('total', 'Total Loss')]:
        plot_l(loss, ylabel)

def hist2fes(hist, temperature):
    epsilon = 1e-10
    prob = hist + epsilon
    prob /= prob.sum()
    fes = -np.log(prob) * 6.02214129 * 1.3806488 * temperature / 1000 * 0.239006
    fes = fes - np.min(fes)
    return fes

def calc_1d_fes(data, label, color, weight, **kwargs):
    hist, bins = np.histogram(data,bins=40,density=True,weights=weight)
    X = [(bins[i]+bins[i+1])/2  for i in range(len(bins)-1)]
    pmf = hist2fes(hist, temperature)
    plt.plot(X,pmf,label=label,color=color, **kwargs)

def calc_2d_fes(data_x, data_y, drange, weight, **kwargs):
    hist, x_edges, y_edges = np.histogram2d(data_x, data_y, bins=30, density=True, range=drange, weights=weight)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    fes = hist2fes(hist, temperature)
    X, Y = np.meshgrid(x_centers, y_centers)
    contour = plt.contourf(X, Y, fes.T, levels=30, cmap=soft_plasma, **kwargs)
    cbar = plt.colorbar(contour)
    cbar.set_label("Free Energy (kcal/mol)", fontsize=20)
    cbar.ax.tick_params(labelsize=12)

def plot_rmsd_1dfes(task, task_name, n_clusters, weight):
    data = load_data_1d([f'{task}/data/c{i}_{task_name}_rmsd.out' for i in range(n_clusters)], 1)
    fig, ax = plt.subplots()
    calc_1d_fes(data, 'reweighted PMF', 'b', weight)
    calc_1d_fes(data, 'original PMF', 'r', None)
    RMSD_D = load_data_1d('2RVD/data/anton_rmsd.dat', 0)
    calc_1d_fes(RMSD_D, 'Anton', 'k', None)
    plt.xlabel(r'RMSD ($\AA$)')
    plt.ylabel(r'PMF (kcal/mol)')
    plt.xticks(np.arange(0, 10, 2),[0, 2, 4, 6, 8], fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylim(-0.5,6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{task}/fig/rmsd_weighted_{task_name}')


def plot_2d_fes(task, task_name, n_clusters, cv, idx_list, weight, label, pic_name, xlim, ylim, **kwargs):
    X, Y= load_data_2d([f'{task}/data/c{i}_{task_name}_{cv}.out' for i in range(n_clusters)], idx_list)

    fig, ax = plt.subplots(figsize=kwargs.get('figsize', None))
    calc_2d_fes(X, Y, (xlim, ylim), None)
    plt.xlabel(label[0])
    plt.ylabel(label[1])
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlim(xlim)
    plt.ylim(ylim)
    if kwargs.get('ticks'):
        ax.set_xticks(kwargs['ticks'])
        ax.set_yticks(kwargs['ticks'])
    if kwargs.get('func'):
        kwargs['func'](ax)
    #plt.legend()
    plt.tight_layout()
    plt.savefig(f'{task}/fig/{pic_name}_raw_{task_name}')

    fig, ax = plt.subplots(figsize=kwargs.get('figsize', None))
    calc_2d_fes(X, Y, (xlim, ylim), weight)
    plt.xlabel(label[0])
    plt.ylabel(label[1])
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlim(xlim)
    plt.ylim(ylim)
    if kwargs.get('ticks'):
        ax.set_xticks(kwargs['ticks'])
        ax.set_yticks(kwargs['ticks'])
    if kwargs.get('func'):
        kwargs['func'](ax)
    #plt.legend()
    plt.tight_layout()
    plt.savefig(f'{task}/fig/{pic_name}_weighted_{task_name}')

def plot_2dfes(task, task_name, n_clusters, weight):
    plot_2d_fes(task, task_name, n_clusters,'hb', [2,1], weight, [r'HB2 $(\AA)$', r'HB1 $(\AA)$'], 'hb_2D', (2, 18), (2, 18), ticks=[2,5,10,15,18])
    def set_ticks(ax):
        ax.set_xticks(np.arange(0, 12, 2),[0, 2, 4, 6, 8, 10])
        # ax.set_yticks(np.arange(0, 12, 2),[0, 2, 4, 6, 8, 10])
    plot_2d_fes(task, task_name, n_clusters,'rmsd', [1,2], weight, [r'$RMSD (\AA)$', r'$R_g (\AA)$'], 'rmsd_2D', (0, 10), (4, 10),func=set_ticks)

if __name__ == '__main__':
    plt.rcParams['savefig.dpi'] = 600
    plt.rcParams['savefig.format'] = 'png'
    plt.rcParams['axes.labelsize'] = 24
    plt.rcParams['xtick.labelsize'] = 20
    plt.rcParams['ytick.labelsize'] = 20
    plt.rcParams['legend.fontsize'] = 16
    
    task = '2RVD'
    n_clusters = 4
    task_affix = None
    if task_affix:
        task_name = f'{task}_{task_affix}_n{n_clusters}'
    else:
        task_name = f'{task}_n{n_clusters}'
    N_confomation_per_cluster = 10000
    my_cmap_list = ['#FAAF76','#A194C6','#5491BB','#F16E65','#255FA7', "#E7E718"][:n_clusters]
    my_listedcmap = mcolors.ListedColormap(my_cmap_list)
    
    reduced = np.load(f'{task}/pca_{task}.npy')
    ktrajs = np.load(f'{task}/kmeans_{task}_n{n_clusters}.npy')
    recon_reduced = np.load(f'{task}/pca_{task_name}_recon.npy')
    recon_ktrajs = np.load(f'{task}/kmeans_{task_name}_recon.npy')
    recon_kdist = np.load(f'{task}/kdist_{task_name}_recon.npy')
    
    os.makedirs(f'{task}/fig', exist_ok=True)
    plot_kmeans_cluster_test(task)
    plot_loss(task, task_name)
    plot_kmeans_raw(task, n_clusters, reduced, ktrajs, my_listedcmap)
    plot_kmeans_train(task, n_clusters, reduced, ktrajs, my_listedcmap, my_cmap_list, N_confomation_per_cluster)
    plot_kmeans_recon(task, task_name, n_clusters, reduced, recon_reduced, recon_ktrajs, my_listedcmap, my_cmap_list)
    plot_recon_convergence(task, task_name, n_clusters, recon_ktrajs, my_cmap_list)
    
    soft_plasma_list = ["#6C39EF", "#BCADE6", "#ADCAE6", '#B3E5FC', '#E0FFFF', "#E8F5DC", '#FFFFE0']
    soft_plasma = mcolors.LinearSegmentedColormap.from_list('soft_plasma', soft_plasma_list, N=256)
    temperature = 340
    weight = get_weights(get_counts(recon_ktrajs, recon_kdist),n_clusters, N_confomation_per_cluster, task)
    plot_rmsd_1dfes(task, task_name, n_clusters,weight)
    plot_2dfes(task, task_name, n_clusters,weight)
