"""
BNN with Gaussian prior on weights and biases
"""

import torch
import numpy as np
import matplotlib as mpl
import matplotlib.pylab as plt
import os, sys
import pickle

CWD = os.getcwd()
OUT_DIR = os.path.join(CWD, "output_1d")
FIG_DIR = os.path.join(CWD, "figures_1d")
sys.path.append(os.path.dirname(CWD))

from bnn_spatial.bnn.nets import GaussianNet, BlankNet
from bnn_spatial.bnn.layers.embedding_layer import EmbeddingLayer
from bnn_spatial.utils import util
from bnn_spatial.gp.model import GP
from bnn_spatial.gp import kernels, base
from bnn_spatial.stage1.wasserstein_mapper import MapperWasserstein
from bnn_spatial.utils.rand_generators import GridGenerator
from bnn_spatial.utils.plotting import plot_spread, plot_param_traces, plot_output_traces, plot_output_hist, \
    plot_output_acf, plot_cov_heatmap, plot_lipschitz, plot_rbf, plot_cov_diff
from bnn_spatial.stage2.likelihoods import LikGaussian
from bnn_spatial.stage2.priors import FixedGaussianPrior, OptimGaussianPrior
from bnn_spatial.stage2.bayes_net import BayesNet
from bnn_spatial.metrics.sampling import compute_rhat

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # for handling OMP: Error 15 (2022/11/21)

#plt.rcParams['figure.figsize'] = [14, 7]
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['image.aspect'] = 'auto'
plt.rcParams['font.size'] = 24
plt.rcParams['axes.titlesize'] = 24

#mpl.use('TkAgg')
#mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # set device to GPU if available

# Ensure directories exist (otherwise error occurs)
util.ensure_dir(OUT_DIR)
util.ensure_dir(FIG_DIR)
stage1_file = open(OUT_DIR + "/stage1.file", "wb")
stage1_txt = open(FIG_DIR + "/stage1.txt", "w+")
stage2_file = open(OUT_DIR + "/stage2.file", "wb")
stage2_txt = open(FIG_DIR + "/stage2.txt", "w+")

# Set seed (make results replicable)
util.set_seed(1)

# Specify BNN structure
hidden_dims = [50, 50, 50, 50]  # list of hidden layer dimensions
depth = len(hidden_dims)
embed_width = hidden_dims[0]
transfer_fn = "tanh"  # activation function

to_save = {'hidden_dims': hidden_dims,
           'transfer_fn': transfer_fn}
pickle.dump(to_save, stage1_file)

"""
Fixed BNN prior (non-optimised)
"""

W_rho = 1.  # softplus(W_rho) is initial standard deviation for weights
b_rho = 1.  # softplus(b_rho) is initial standard deviation for biases

n_test = 256  # number of test inputs (must be a multiple of "width" when using embedding layer)
Xmargin = 4  # margin of domain, so that the domain is [-Xmargin : step_size : Xmargin]
Xtest = np.linspace(-Xmargin, Xmargin, n_test).reshape(-1, 1)
Xtest_tensor = torch.from_numpy(Xtest).to(device)
n_plot = 4000  # number of BNN prior and GP samples to use in plots

# Embedding layer output (used as input in stage 2 code)
rbf_ls = 1
embedding_layer = EmbeddingLayer(input_dim=1,
                                 output_dim=embed_width,
                                 domain=Xtest_tensor,
                                 rbf_ls=rbf_ls)
rbf = embedding_layer(Xtest_tensor)

# Plot the RBFs used in the embedding layer
plot_rbf(domain=Xtest,
         net_width=embed_width,
         lenscale=rbf_ls)
plt.savefig(FIG_DIR + '/embedding_RBFs_lenscale{}.png'.format(rbf_ls), bbox_inches='tight')

to_save = {'n_test': n_test,
           'n_plot': n_plot,
           'rbf_ls': rbf_ls}
pickle.dump(to_save, stage1_file)

# Initialize fixed BNN
std_bnn = GaussianNet(input_dim=1,
                      output_dim=1,
                      hidden_dims=hidden_dims,
                      domain=Xtest_tensor,
                      activation_fn=transfer_fn,
                      prior_per='layer',
                      fit_means=False,
                      rbf_ls=rbf_ls).to(device)

# Perform predictions using 'n_plot' number of sampled weights/biases
std_bnn_samples = std_bnn.sample_functions(Xtest_tensor.float(), n_plot).detach().cpu().numpy().squeeze()

# Obtain network parameter values
W_list, b_list = std_bnn.network_parameters()
W_std, b_std = W_list[0], b_list[0]  # all values are identical

# Plot the sampled BNN predictions
plt.figure()
plot_spread(Xtest_tensor, std_bnn_samples, color='xkcd:greenish', n_keep=10)
plt.ylim([-5.5, 5.5])
plt.savefig(FIG_DIR + '/std_prior_preds.png', bbox_inches='tight')

"""
GP prior
"""

# GP kernel hyper-parameters
sn2 = 0.001   # noise variance
leng = 1.0  # lengthscale
ampl = 1.0  # amplitude

# Use radial basis function for kernel
kernel_class = base.Isotropic
kernel_fn = kernels.RBF
kernel = kernel_class(cov=kernel_fn,
                      ampl=ampl,
                      leng=leng,
                      power=2)

to_save = {'sn2': sn2,
           'leng': leng,
           'ampl': ampl,
           'kernel_class': kernel_class,
           'kernel_fn': kernel_fn}
pickle.dump(to_save, stage1_file)

# Initialize GP prior
gp = GP(kern=kernel).to(device)

# Sample functions from GP prior
gp_samples = gp.sample_functions(Xtest_tensor.float(), n_plot+1).detach().cpu().numpy().squeeze()
gp_latent = gp_samples[:, 0].squeeze()  # latent function
gp_samples = gp_samples[:, 1:]  # prior samples

# Plot the GP prior using sampled functions
plt.figure()
plot_spread(Xtest_tensor, gp_samples, n_keep=10)
plt.ylim([-5.5, 5.5])
plt.savefig(FIG_DIR + '/GP_prior_preds.png', bbox_inches='tight')

"""
GP-induced BNN prior
"""

# Initialize BNN to be optimised
opt_bnn = GaussianNet(input_dim=1,
                      output_dim=1,
                      hidden_dims=hidden_dims,
                      domain=Xtest_tensor,
                      activation_fn=transfer_fn,
                      prior_per='layer',
                      fit_means=False,
                      rbf_ls=rbf_ls).to(device)

# Use a grid of n_data points for the measurement set
data_generator = GridGenerator(-4, 4, input_dim=1)

# Perform Wasserstein optimisation to match BNN prior (opt_bnn) with GP prior
mapper_num_iters = 1600
outer_lr = 0.005  # initial learning rate for outer optimiser (wrt psi)
inner_lr = 0.01  # initial learning rate for inner optimiser (wrt theta)
inner_steps = 20  # number of steps for inner optimisation loop
n_measure = n_test
n_stochastic = 4096

to_save = {'mapper_num_iters': mapper_num_iters,
           'outer_lr': outer_lr,
           'inner_lr': inner_lr,
           'inner_steps': inner_steps,
           'n_measure': n_measure,
           'n_stochastic': n_stochastic}
pickle.dump(to_save, stage1_file)

if not os.path.exists(os.path.join(OUT_DIR, "ckpts/it-{}.ckpt".format(mapper_num_iters))):
    mapper = MapperWasserstein(gp, opt_bnn, data_generator,
                               out_dir=OUT_DIR,
                               wasserstein_steps=inner_steps,
                               wasserstein_lr=inner_lr,
                               n_data=n_test,
                               n_gpu=1,
                               gpu_gp=True,
                               starting_steps=inner_steps,
                               starting_lr=inner_lr,
                               continue_training=True)
    w_hist = mapper.optimise(num_iters=mapper_num_iters,
                             n_samples=n_stochastic,  # originally 512
                             lr=outer_lr,
                             print_every=50)  # specify pre-training for Lipschitz neural net
    path = os.path.join(OUT_DIR, "wsr_values.log")
    np.savetxt(path, w_hist, fmt='%.6e')

# Obtain Wasserstein distance values for each iteration
wdist_file = os.path.join(OUT_DIR, "wsr_values.log")
wdist_vals = np.loadtxt(wdist_file)
indices = np.arange(mapper_num_iters)[::1]  # indices for use in plots (::step)

# Plot Wasserstein iterations
plt.figure()
plt.plot(indices, wdist_vals[indices], "-ko", ms=4)
plt.ylabel(r'$W_1(p_{gp}, p_{nn})$')
plt.xlabel('Iteration (Outer Loop)')
plt.savefig(FIG_DIR + '/Wasserstein_steps.png', bbox_inches='tight')

# Plot Wasserstein iterations on log scale
plt.figure()
plt.plot(indices, wdist_vals[indices], "-ko", ms=4)
plt.yscale('log')
plt.ylabel(r'$W_1(p_{gp}, p_{nn})$')
plt.xlabel('Iteration (Outer Loop)')
plt.savefig(FIG_DIR + '/Wasserstein_steps_log.png', bbox_inches='tight')

# Obtain NN gradient norms appearing in Lipschitz loss penalty term
nn_file = os.path.join(OUT_DIR, "f_grad_norms.npy")
f_grad_norms = np.load(nn_file)

# Plot gradient norms to check if unit norm restriction is approximately satisfied (as enforced by the penalty term)
plot_lipschitz(inner_steps=inner_steps,
               outer_steps=mapper_num_iters,
               samples=f_grad_norms,
               type='penalty')
plt.savefig(FIG_DIR + '/Lipschitz_gradient_norm.png', bbox_inches='tight')

# Obtain concatenated parameter gradient norms
param_file = os.path.join(OUT_DIR, "p_grad_norms.npy")
p_grad_norms = np.load(param_file)

# Plot norms of concatenated parameter gradients dL/dp for loss L, weights/biases p, to measure "network change"
plot_lipschitz(inner_steps=inner_steps,
               outer_steps=mapper_num_iters,
               samples=p_grad_norms,
               type='parameter')
plt.savefig(FIG_DIR + '/parameter_gradients_norm.png', bbox_inches='tight')

# Obtain Lipschitz losses
loss_file = os.path.join(OUT_DIR, "lip_losses.npy")
lip_losses = np.load(loss_file)

# Plot Lipschitz losses to assess convergence towards Wasserstein distance (the maximum)
plot_lipschitz(inner_steps=inner_steps,
               outer_steps=mapper_num_iters,
               samples=lip_losses,
               type='loss')
plt.savefig(FIG_DIR + '/Lipschitz_losses.png', bbox_inches='tight')

num_ckpt = mapper_num_iters // 50  # number of 50-step checkpoints with saved data

# Create plot for GP and optimised BNN covariances (start with GP)
K_ss = kernel.K(Xtest_tensor).detach().cpu().numpy()  # GP covariance matrix
plt.figure(num='cov')
gp_cov_mean = 0
for cc in range(n_test // 2):
    gp_cov_mean += K_ss[cc, cc:cc + (n_test // 2)]
gp_cov_mean /= n_test // 2
plt.plot(Xtest[n_test // 2:], gp_cov_mean, 'k-', lw=4, label='GP')

# Plot the sampled predictions for desired checkpoints
plt_checkpoints = np.unique([10/50, 1, 2] + list(range(4, num_ckpt, 4)) + [num_ckpt])
plt_checkpoints.sort()  # ascending order by default
if num_ckpt // 2 not in plt_checkpoints:
    raise Exception('Require mid-checkpoint to be contained in plt_checkpoints')
heatmap_save = np.unique(np.array([2, 4, 8, num_ckpt // 2, num_ckpt]))
if len(heatmap_save) / 2 == len(heatmap_save) // 2:
    heatmap_save = np.insert(heatmap_save, obj=0, values=1)
heatmap_covs = [None] * (len(heatmap_save) + 1)  # add 1 for GP cov matrix

for ckpt in plt_checkpoints:
    # Load optimised BNN from saved checkpoint (dictionary with parameter values)
    ckpt_path = os.path.join(OUT_DIR, "ckpts", "it-{}.ckpt".format(int(50*ckpt)))
    opt_bnn.load_state_dict(torch.load(ckpt_path))

    # Perform predictions using 'n_plot' number of sampled weights/biases (for each layer) - shape (n_data, n_plot)
    opt_bnn_samples = opt_bnn.sample_functions(Xtest_tensor.float(), n_plot).detach().cpu().numpy().squeeze()

    # Obtain network parameter values
    W_list, b_list = opt_bnn.network_parameters()

    # Construct strings with parameter values (accounts for varying network depth)
    W_str = ""
    b_str = ""
    for lvl in range(len(W_list)):
        sigma_W = r'$\sigma_{{w_{}}}$ = {}'.format(lvl+1, W_list[lvl])
        sigma_b = r'$\sigma_{{b_{}}}$ = {}'.format(lvl+1, b_list[lvl])
        if lvl != len(W_list) - 1:
            sigma_W += r', '
            sigma_b += r', '
        W_str += sigma_W
        b_str += sigma_b

    # Plot the optimised BNN at this checkpoint
    plt.figure()
    plot_spread(Xtest_tensor, opt_bnn_samples, color='xkcd:peach', n_keep=10)
    plt.ylim([-5.5, 5.5])
    plt.savefig(FIG_DIR + '/GPiG_prior_preds_step{}.png'.format(int(50*ckpt)), bbox_inches='tight')

    ###################################
    # Empirical covariance computations

    emp_cov0 = np.cov(opt_bnn_samples)  # BNN empirical covariance (zero mean)
    emp_cov = opt_bnn_samples @ opt_bnn_samples.T / (n_plot - 1)  # same as above

    # Note: opt_bnn_samples has rows for spatial inputs, cols for sampled functions

    # Compute average of empirical covariances
    emp_cov_mean = 0
    for cc in range(n_test // 2):
        emp_cov_mean += emp_cov[cc, cc:cc + (n_test//2)]
    emp_cov_mean /= n_test // 2

    # Plot empirical covariance mean and GP covariance
    plt.figure(num='cov')
    plt.plot(Xtest[n_test // 2:], emp_cov_mean, '-', lw=2, label='{}-step BNN'.format(int(50*ckpt)))

    # Store covariance matrices for first and last checkpoints
    if ckpt in heatmap_save:
        ckpt_idx = np.argwhere(ckpt == heatmap_save).squeeze()
        heatmap_covs[ckpt_idx] = emp_cov

# Finalise covariance plot
fig = plt.figure(num='cov')
fig.tight_layout()
fig.set_figwidth(24)
fig.set_figheight(12)
if len(plt_checkpoints) <= 10:  # position legend in top right corner for few checkpoints
    plt.legend(loc='upper right')
else:  # position legend below plot for many checkpoints
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=6)
plt.savefig(FIG_DIR + '/GP_BNN_covariances.png', bbox_inches='tight')

# Plot covariance heatmaps
heatmap_covs[-1] = K_ss  # store GP covariance matrix for comparison
heatmap_titles = ['{}-step BNN'.format(int(50*ckpt)) for ckpt in heatmap_save]
heatmap_titles.append('GP')
plot_cov_heatmap(covs=heatmap_covs,
                 domain=Xtest_tensor,
                 titles=heatmap_titles)
plt.savefig(FIG_DIR + '/GP_BNN_cov_heatmaps.png', bbox_inches='tight')

plot_cov_diff(covs=heatmap_covs[-4:-1],
              gp_cov=heatmap_covs[-1],
              domain=Xtest_tensor,
              titles=heatmap_titles[-4:-1])
plt.savefig(FIG_DIR + '/GP_BNN_cov_diffs.png', bbox_inches='tight')

"""
GP Posterior
"""

# Create data set
n_train = 100  # number of observations
inds1 = np.random.randint(8*Xtest.size//16, 11*Xtest.size//16, size=n_train-5)
inds2 = np.random.randint(3*Xtest.size//16, 4*Xtest.size//16, size=5)
inds = np.concatenate((inds1, inds2))
X = Xtest[inds]  # observational inputs
err_cov = sn2 * np.eye(n_train)  # measurement error cov matrix
err_mean = np.zeros(n_train)  # measurement error mean (zero)
y = gp_latent[inds] + np.random.multivariate_normal(err_mean, err_cov).flatten()  # noisy observations
X_tensor = Xtest_tensor[inds]
y_tensor = torch.from_numpy(y).reshape([-1, 1]).to(device)

# Make predictions
GP.assign_data(gp,
               X=X_tensor,
               Y=y_tensor,
               sn2=sn2)
gp_preds = gp.predict_f_samples(Xtest_tensor, 1000)
gp_preds = gp_preds.detach().cpu().numpy().squeeze()

# Plot the sampled predictions from the GP posterior
plt.figure()
plot_spread(Xtest_tensor, gp_preds, n_keep=10)
plt.plot(X, y, 'ok', zorder=10, ms=10)
plt.ylim([-3.5, 3.5])
plt.savefig(FIG_DIR + '/GP_posterior_preds.png', bbox_inches='tight')

# Save all parameter settings used in stage 1 code
stage1_file.close()
stage1_file = open(OUT_DIR + "/stage1.file", "rb")
while True:
    try:
        stage1_txt.write(str(pickle.load(stage1_file)) + '\n')
    except EOFError:
        break

"""
SGHMC - Fixed BNN posterior
"""

# SGHMC Hyper-parameters (see sampling_configs comments for interpretation)
n_chains = 4
keep_every = 1000
n_samples = 500
n_burn = 3000  # must be multiple of keep_every
n_burn_thin = n_burn // keep_every
n_discarded = 100 - n_burn_thin

sampler = 'adaptive_sghmc'
sghmc_lr = 1e-5
adaptive_lr = sghmc_lr ** 0.5
if sampler == 'sghmc':
    sampler_lr = sghmc_lr
elif sampler == 'adaptive_sghmc':
    sampler_lr = adaptive_lr
else:
    raise Exception('Sampler not supported.')

sampling_configs = {
    "batch_size": 32,  # Mini-batch size
    "num_samples": n_samples,  # Number of samples kept for each chain (default 30)
    "n_discarded": n_discarded,  # Number of samples to discard after burn-in adaptation phase (default 10)
    "num_burn_in_steps": n_burn,  # Number of burn-in adaptation phase steps for adaptive SGHMC (default 2000)
    "keep_every": keep_every,  # Thinning interval (default 200)
    "lr": sampler_lr,  # Step size (1e-4 works well for standard SGHMC, and 1e-2 for adaptive)
    "mdecay": 0.05,  # Momentum coefficient
    "num_chains": n_chains,  # Number of sampling chains (for r_hat, at least 4 suggested)
    "print_every_n_samples": 50,  # Control feedback regularity
}

pickle.dump({'n_train': n_train, 'sampler': sampler}, stage2_file)
pickle.dump(sampling_configs, stage2_file)

# Initialize the prior
prior = FixedGaussianPrior(mu=0, std=1)  # same settings as fixed BNN prior

# Setup likelihood (also used with optimised prior later), and neural network for usage with SGHMC
net = BlankNet(output_dim=1,
               hidden_dims=hidden_dims,
               activation_fn=transfer_fn)
likelihood = LikGaussian(sn2)

# Generate posterior BNN parameter samples with multiple chains of sampling
bayes_net_std = BayesNet(net, likelihood, prior,
                         sampling_method=sampler,
                         n_gpu=0)
bayes_net_std.add_embedding_layer(input_dim=1,
                                  rbf_dim=embed_width,
                                  domain=Xtest_tensor,
                                  rbf_ls=rbf_ls)
std_weights, std_weights_pred = bayes_net_std.sample_multi_chains(X, y, **sampling_configs)

n_samples_all_chains = len(std_weights)  # total number of samples from all chains, including discarded burn-in samples

# Create arrays of MCMC samples (samples cols, parameters rows)
std_chains = np.zeros((2 * depth, n_samples_all_chains))
std_chains_pred = np.zeros((2 * depth, n_chains * n_samples))

# Populate the array containing all samples
for k in range(n_samples_all_chains):
    std_dict = std_weights[k]  # dictionary of name-tensor pairs for network parameters
    if k == 0:
        print('Tensor names (keys) for the fixed BNN')
        print(std_dict.keys())

    W_tensors = []
    b_tensors = []
    for name in std_dict.keys():
        if 'batch_norm' in name:
            continue
        param = std_dict[name]
        if '.W' in name:
            W_tensors.append(param)
        elif '.b' in name:
            b_tensors.append(param)

    # Store weights in even entries, and biases in odd entries
    for ll in range(2 * depth):
        if ll % 2 == 0:
            try:
                std_chains[ll, k] = W_tensors[ll // 2][0, 0]  # use entry [0, 0] of each weight matrix
            except IndexError:
                std_chains[ll, k] = W_tensors[ll // 2][0]
        else:
            std_chains[ll, k] = b_tensors[(ll - 1) // 2][0]  # use entry [0] of each bias vector

# Populate the array only containing samples utilised for predictions
for k in range(n_chains * n_samples):
    std_dict = std_weights_pred[k]  # dictionary of name-tensor pairs for network parameters

    W_tensors = []
    b_tensors = []
    for name in std_dict.keys():
        if 'batch_norm' in name:
            continue
        param = std_dict[name]
        if '.W' in name:
            W_tensors.append(param)
        elif '.b' in name:
            b_tensors.append(param)

    # Store weights in even entries, and biases in odd entries
    for ll in range(2 * depth):
        if ll % 2 == 0:
            try:
                std_chains_pred[ll, k] = W_tensors[ll // 2][0, 0]  # use entry [0, 0] of each weight matrix
            except IndexError:
                std_chains_pred[ll, k] = W_tensors[ll // 2][0]
        else:
            std_chains_pred[ll, k] = b_tensors[(ll - 1) // 2][0]  # use entry [0] of each bias vector

# Construct vector with subplot titles (accounts for varying network depth)
trace_titles = []
for lvl in range(len(W_list)):
    W_l = r'$\mathbf{{W}}_{}$'.format(lvl + 1)
    b_l = r'$\mathbf{{b}}_{}$'.format(lvl + 1)
    trace_titles.append(W_l)
    trace_titles.append(b_l)

# Construct vector of legend entries (accounts for varying chain length)
legend_entries = ['Chain {}'.format(cc+1) for cc in range(n_chains)]

# Trace plots of MCMC iterates
print('Generating plot of MCMC iterates')
plot_param_traces(param_chains=std_chains,
                  n_chains=n_chains,
                  net_depth=depth,
                  n_discarded=n_discarded,
                  n_burn=n_burn_thin,
                  trace_titles=trace_titles,
                  legend_entries=legend_entries)
plt.savefig(FIG_DIR + '/std_chains.png', bbox_inches='tight')

# Make predictions
bnn_std_preds, bnn_std_preds_all = bayes_net_std.predict(Xtest_tensor)

# MCMC convergence diagnostics
r_hat = compute_rhat(bnn_std_preds, sampling_configs["num_chains"])
rhat_mean_fixed = float(r_hat.mean())
rhat_sd_fixed = float(r_hat.std())
print(r"R-hat: mean {:.4f} std {:.4f}".format(rhat_mean_fixed, rhat_sd_fixed))
to_save = {'rhat_mean_fixed': rhat_mean_fixed,
           'rhat_sd_fixed': rhat_sd_fixed}
pickle.dump(to_save, stage2_file)

# Plot the sampled predictions from the fixed BNN posterior
plt.figure()
plot_spread(Xtest_tensor, bnn_std_preds.T, color='xkcd:greenish', n_keep=10)
plt.plot(X, y, 'ok', zorder=10, ms=10)
plt.ylim([-3.5, 3.5])
plt.savefig(FIG_DIR + '/std_posterior_preds.png'.format(depth), bbox_inches='tight')

# Trace plot of network output at specified input point(s)
plot_output_traces(domain=Xtest_tensor,
                   preds=bnn_std_preds_all,
                   n_chains=n_chains,
                   n_discarded=n_discarded,
                   n_burn=n_burn_thin,
                   legend_entries=legend_entries)
plt.savefig(FIG_DIR + '/std_posterior_trace.png', bbox_inches='tight')

# Histogram of network output at same input point(s)
plot_output_hist(domain=Xtest_tensor,
                 preds=bnn_std_preds)
plt.savefig(FIG_DIR + '/std_posterior_hist.png', bbox_inches='tight')

# ACF of network output at same input point(s)
plot_output_acf(domain=Xtest_tensor,
                preds=bnn_std_preds,
                n_chains=n_chains,
                n_samples_kept=n_samples)
plt.savefig(FIG_DIR + '/std_posterior_acf.png', bbox_inches='tight')

"""
SGHMC - GP-induced BNN posterior
"""

# Plot the estimated posteriors for desired 50-step checkpoints
mcmc_checkpoints = [10/50, 2, num_ckpt//2, num_ckpt]
mcmc_checkpoints = [num_ckpt]
for ckpt in mcmc_checkpoints:

    # Load the optimised prior
    if ckpt != num_ckpt // 2:
        ckpt_path = os.path.join(OUT_DIR, "ckpts", "it-{}.ckpt".format(int(50*ckpt)))
    else:
        try:
            ckpt_path = os.path.join(OUT_DIR, "ckpts", "it-{}.ckpt".format(int(50*ckpt)))
        except NotADirectoryError:
            continue
    prior = OptimGaussianPrior(ckpt_path)

    # Obtain network parameter values (for plotting only)
    opt_bnn.load_state_dict(torch.load(ckpt_path))
    W_list, b_list = opt_bnn.network_parameters()

    # Construct strings with parameter values (accounts for varying network depth)
    W_str = ""
    b_str = ""
    for lvl in range(len(W_list)):
        sigma_W = r'$\sigma_{{w_{}}}$ = {}'.format(lvl + 1, W_list[lvl])
        sigma_b = r'$\sigma_{{b_{}}}$ = {}'.format(lvl + 1, b_list[lvl])
        if lvl != len(W_list) - 1:
            sigma_W += r', '
            sigma_b += r', '
        W_str += sigma_W
        b_str += sigma_b

    # Generate posterior BNN parameter samples with multiple chains of sampling
    bayes_net_optim = BayesNet(net, likelihood, prior,
                               sampling_method=sampler,
                               n_gpu=0)
    bayes_net_optim.add_embedding_layer(input_dim=1,
                                        rbf_dim=embed_width,
                                        domain=Xtest_tensor,
                                        rbf_ls=rbf_ls)
    optim_weights, optim_weights_pred = bayes_net_optim.sample_multi_chains(X, y, **sampling_configs)

    # Create arrays of MCMC samples (samples cols, parameters rows)
    optim_chains = np.zeros((2 * depth, n_samples_all_chains))
    optim_chains_pred = np.zeros((2 * depth, n_chains * n_samples))

    # Populate the array containing all samples
    for k in range(n_samples_all_chains):
        optim_dict = optim_weights[k]  # dictionary of name-tensor pairs for network parameters
        if k == 0:
            print('Tensor names (keys) for the GPi-G BNN')
            print(optim_dict.keys())

        W_tensors = []
        b_tensors = []
        for name in optim_dict.keys():
            if 'batch_norm' in name:
                continue
            param = optim_dict[name]
            if '.W' in name:
                W_tensors.append(param)
            elif '.b' in name:
                b_tensors.append(param)

        # Store weights in even entries, and biases in odd entries
        for ll in range(2 * depth):
            if ll % 2 == 0:
                try:
                    optim_chains[ll, k] = W_tensors[ll // 2][0, 0]  # use entry [0, 0] of each weight matrix
                except IndexError:
                    optim_chains[ll, k] = W_tensors[ll // 2][0]
            else:
                optim_chains[ll, k] = b_tensors[(ll - 1) // 2][0]  # use entry [0] of each bias vector

    # Populate the array only containing samples utilised for predictions
    for k in range(n_chains * n_samples):
        optim_dict = optim_weights_pred[k]  # dictionary of name-tensor pairs for network parameters

        W_tensors = []
        b_tensors = []
        for name in optim_dict.keys():
            if 'batch_norm' in name:
                continue
            param = optim_dict[name]
            if '.W' in name:
                W_tensors.append(param)
            elif '.b' in name:
                b_tensors.append(param)

        # Store weights in even entries, and biases in odd entries
        for ll in range(2 * depth):
            if ll % 2 == 0:
                try:
                    optim_chains_pred[ll, k] = W_tensors[ll // 2][0, 0]  # use entry [0, 0] of each weight matrix
                except IndexError:
                    optim_chains_pred[ll, k] = W_tensors[ll // 2][0]
            else:
                optim_chains_pred[ll, k] = b_tensors[(ll - 1) // 2][0]  # use entry [0] of each bias vector

    # Make predictions
    bnn_optim_preds, bnn_optim_preds_all = bayes_net_optim.predict(Xtest_tensor)  # preds have rows samples, cols traces

    # MCMC convergence diagnostics
    r_hat = compute_rhat(bnn_optim_preds, sampling_configs["num_chains"])
    rhat_mean_opt = float(r_hat.mean())
    rhat_sd_opt = float(r_hat.std())
    print(r"R-hat: mean {:.4f} std {:.4f}".format(rhat_mean_opt, rhat_sd_opt))
    to_save = {'rhat_mean_opt': rhat_mean_opt,
               'rhat_sd_opt': rhat_sd_opt}
    pickle.dump(to_save, stage2_file)

    if ckpt == num_ckpt:

        #################
        # Parameter plots

        # Trace plots of MCMC iterates
        plot_param_traces(param_chains=optim_chains,
                          n_chains=n_chains,
                          net_depth=depth,
                          n_discarded=n_discarded,
                          n_burn=n_burn_thin,
                          trace_titles=trace_titles,
                          legend_entries=legend_entries)
        plt.savefig(FIG_DIR + '/optim_chains.png', bbox_inches='tight')

        ######################
        # Network output plots

        # Trace plot of network output at specified input point(s)
        plot_output_traces(domain=Xtest_tensor,
                           preds=bnn_optim_preds_all,
                           n_chains=n_chains,
                           n_discarded=n_discarded,
                           n_burn=n_burn_thin,
                           legend_entries=legend_entries)
        plt.savefig(FIG_DIR + '/GPiG_posterior_trace.png', bbox_inches='tight')

        # Histogram of network output at same input point(s)
        plot_output_hist(domain=Xtest_tensor,
                         preds=bnn_optim_preds)
        plt.savefig(FIG_DIR + '/GPiG_posterior_hist.png', bbox_inches='tight')

        # ACF of network output at same input point(s)
        plot_output_acf(domain=Xtest_tensor,
                        preds=bnn_optim_preds,
                        n_chains=n_chains,
                        n_samples_kept=n_samples)
        plt.savefig(FIG_DIR + '/GPiG_posterior_acf.png', bbox_inches='tight')

    # Plot the sampled predictions from the GP-induced BNN posterior
    plt.figure()
    plot_spread(Xtest_tensor, bnn_optim_preds.T, color='xkcd:peach', n_keep=10)
    plt.plot(X, y, 'ok', zorder=10, ms=10)
    plt.ylim([-3.5, 3.5])
    plt.savefig(FIG_DIR + '/GPiG_posterior_preds_step{}.png'.format(int(50*ckpt)), bbox_inches='tight')

# Save all parameter settings used in stage 2 code
stage2_file.close()
stage2_file = open(OUT_DIR + "/stage2.file", "rb")
while True:
    try:
        stage2_txt.write(str(pickle.load(stage2_file)) + '\n')
    except EOFError:
        break



