import copy
from argparse import Namespace
import itertools
import random

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

from bax.models.simple_gp import SimpleGp
from bax.alg.algorithms import AverageOutputs
from bax.acq.acqoptimize import AcqOptimizer

import neatplot
neatplot.set_style('fonts')
neatplot.update_rc('font.size', 16)


# Set random seed
seed = 11
random.seed(seed)
np.random.seed(seed)

# Set which acqfn to plot
acq = 'kg'
#acq = 'ei'
#acq = 'es'
#acq = 'kguess'


# Set function
f = lambda x: 2 * np.sin(x[0]) + 0.15 * x[0]

# Set data for model
data = Namespace()
data.x = [[0.8], [2.3], [4.2], [6.65], [9.3]]
data.y = [f(x) for x in data.x]

# Set model as a GP
gp_params = {'ls': 1.5, 'alpha': 1.5, 'sigma': 1e-2, 'n_dimx': 1}
model = SimpleGp(gp_params)
model.set_data(data)

# Set arrays
min_x = 0.0 
max_x = 10.0
n_test = 500
x_test = [[x] for x in np.linspace(min_x, max_x, n_test)]


# Do inference in GP
mu, cov = model.gp_post_wrapper(x_test, data, full_cov=True)
std = np.sqrt(np.diag(cov))

# Sample functions from GP
n_samp = 310
f_sample = model.get_normal_samples(mu, cov, n_samp, full_cov=True)
f_sample = np.array(f_sample).T.reshape(n_samp, n_test)


# Plotting
# --------

# Make arrays
x_plot = np.array(x_test).reshape(-1)
n_std = 3
lcb, ucb =  mu - n_std * std, mu + n_std * std
f_true = np.array([f(x) for x in x_test]).reshape(-1)
data_x = np.array(data.x).reshape(-1)
data_y = np.array(data.y).reshape(-1)

# Make fig, ax
fig, ax = plt.subplots(figsize=(4, 2))

# Plot posterior predictive
ax.fill_between(x_plot, lcb, ucb, color=(0.25, 0.5, 0.75, 0.2))

# Plot true function
ax.plot(x_plot, f_true, '-', c='k', lw=1)

# Plot posterior function samples
n_samp_to_plot = 5
for fs in f_sample[:n_samp_to_plot]:
    ax.plot(x_plot, fs, '-', c=(0.25, 0.5, 0.75, 0.5), lw=1)

# Plot posterior mean
ax.plot(x_plot, mu, '--', c='r', lw=1.0)

# Plot observations
ax.plot(data_x, data_y, 'o', c='k', ms=5)

# Plot Bayes action
plot_extra = True
if acq == 'kg':
    opt_x = x_test[np.argmax(mu)][0]
    ax.plot([opt_x, opt_x], [-5, 5], c='orange')
    ax.text(7.9, -3.25, '$a^*$', c='orange')

    if plot_extra:
        opt_y = f([opt_x])
        ax.plot([-1, 11], [opt_y, opt_y], ':', c='k', lw=1)
        ax.text(5.8, opt_y + 0.4, '$f(a^*)$', c='k', fontdict={'size': 12})

elif acq == 'ei':
    opt_x = data.x[np.argmax(data.y)][0]
    ax.plot([opt_x, opt_x], [-5, 5], c='orange')
    ax.text(2.5, -3.25, '$a^*$', c='orange')

    if plot_extra:
        opt_y = f([opt_x])
        ax.plot([-1, 11], [opt_y, opt_y], ':', c='k', lw=1)
        ax.text(3.0, opt_y + 0.4, '$f(a^*)$', c='k', fontdict={'size': 12})

elif acq == 'es':
    min_y = -3.58
    argmaxes = np.array([x_test[np.argmax(fs)][0] for fs in f_sample]).reshape(-1, 1)
    #ax.plot(argmaxes, np.zeros(n_samp) + min_y, 'x', c='orange') # Plotting argmaxes
    kde = KernelDensity(kernel="gaussian", bandwidth=0.5).fit(argmaxes)
    log_dens = kde.score_samples(np.array(x_test).reshape(-1, 1))
    dens_to_plot = (np.exp(log_dens) * 8.0) + min_y
    dens_bottom_line = min_y * np.ones(len(dens_to_plot))
    ax.fill_between(x_plot, dens_bottom_line, dens_to_plot, color='orange', alpha=0.5)
    ax.plot(x_plot, dens_to_plot, color='orange')
    ax.text(6.1, -3.25, '$a^*$', c='orange')

    # Plot true x* vertical line
    opt_f_true_idx = np.argmax(f_true)
    opt_x_true = x_plot[opt_f_true_idx]
    ax.plot([opt_x_true, opt_x_true], [-5, 5], ':', c='k', lw=1)
    ax.text(8.15, -3.25, '$x^*$', c='k')

    if plot_extra:
        opt_dens = dens_to_plot[opt_f_true_idx]
        ax.plot([-1, 11], [opt_dens, opt_dens], ':', c='k', lw=1)
        ax.text(8.5, opt_dens - 0.8, '$a^*(x^*)$', c='k', fontdict={'size': 12})

elif acq == 'kguess':
    pairs = list(itertools.combinations(np.arange(len(x_test)), 2))
    pairs = random.sample(pairs, 5000)
    pair_scores = [
        np.mean([max(fs[pair[0]], fs[pair[1]]) for fs in f_sample]) for pair in pairs
    ]
    best_idx = pairs[np.argmax(pair_scores)]
    best_x = (x_plot[best_idx[0]], x_plot[best_idx[1]])
    ax.plot([best_x[0], best_x[0]], [-5, 5], c='orange')
    ax.text(best_x[0] + 0.2, -3.2, '$a^*_1$', c='orange')
    ax.plot([best_x[1], best_x[1]], [-5, 5], c='orange')
    ax.text(best_x[1] + 0.2, -3.2, '$a^*_2$', c='orange')

    if plot_extra:
        best_y = (f([best_x[0]]), f([best_x[1]]))
        ax.plot([-1, 11], [best_y[0], best_y[0]], ':', c='k', lw=1)
        ax.text(3.5, best_y[0] - 0.8, '$f(a_1^*)$', c='k', fontdict={'size': 12})
        ax.plot([-1, 11], [best_y[1], best_y[1]], ':', c='k', lw=1)
        ax.text(4.0, best_y[1] + 0.3, '$f(a_2^*)$', c='k', fontdict={'size': 12})

# Add text
if acq == 'kg':
    acq_text = (
        'Action set: $\mathcal{A} = \mathcal{X}$'
        '\nLoss: $\ell(f, a) = -f(a)$, for $a \in \mathcal{A}$'
    )
if acq == 'ei':
    acq_text = (
        'Action set: $\mathcal{A} = \{x_t\}_{t=1}^T$'
        '\nLoss: $\ell(f, a) = -f(a)$, for $a \in \mathcal{A}$'
    )
elif acq == 'es':
    acq_text = (
        'Action set: $\mathcal{A} = \mathcal{P}(\mathcal{X})$'
        '\nLoss: $\ell(f, a) = -\log a(x^*)$,  $a \in \mathcal{A}$'
    )
if acq == 'kguess':
    acq_text = (
        'Action set: $\mathcal{A} = \mathcal{X} \\times \mathcal{X}$'
        #'\nLoss: $\ell(f, a) = f(a_1) \\vee f(a_2)$, '
        #'$a \in \mathcal{A}$'
        '\nLoss: $\ell(f, a) = \min(f(a_1), f(a_2))$'
    )

#ax.text(0, -6.5, acq_text)
ax.text(0, -7.5, acq_text)

# Plot settings
if acq =='kg':
    title = 'Knowledge Gradient'
elif acq =='ei':
    title = 'Expected Improvement'
elif acq == 'es':
    title = 'Entropy Search'
elif acq == 'kguess':
    title = '$k$-Guesses'

#ax.set(xlim=(-0.1, 10.1), ylim=(-3.6, 4.6), xlabel='', ylabel='$y$', title=title)
#ax.text(4.85, -4.9, '$x$') # Custom xlabel positioning

ax.set(
    xlim=(-0.1, 10.1),
    ylim=(-3.6, 4.6),
    xticklabels=[],
    yticklabels=[],
    xticks=[],
    yticks=[],
    title=title
)
ax.text(10.0, -4.9, '$\mathcal{X}$') # Custom x axis label

# Save and show figure
neatplot.save_figure('bayes_action_' + acq, 'pdf')
plt.show()
