"""
BNN with Gaussian prior on weights and biases
"""

import torch
import numpy as np
import matplotlib as mpl
import matplotlib.pylab as plt
import os, sys
import pickle

CWD = sys.path[0]  # directory containing script
OUT_DIR = os.path.join(CWD, "output_2d_naive")
FIG_DIR = os.path.join(CWD, "figures_2d_naive")
sys.path.append(os.path.dirname(CWD))

from bnn_spatial.bnn.nets import GaussianNet, BlankNet
from bnn_spatial.utils import util
from bnn_spatial.gp.model import GP
from bnn_spatial.gp import kernels, base
from bnn_spatial.utils.rand_generators import GridGenerator
from bnn_spatial.utils.plotting import plot_param_traces, plot_output_traces, plot_output_hist, plot_output_acf, \
        plot_mean_sd, plot_rbf, plot_samples_2d, plot_lipschitz, plot_cov_heatmap, plot_cov_nonstat, plot_cov_nonstat_diff
from bnn_spatial.stage1.wasserstein_mapper import MapperWasserstein
from bnn_spatial.stage2.likelihoods import LikGaussian
from bnn_spatial.stage2.priors import FixedGaussianPrior, OptimGaussianPrior
from bnn_spatial.stage2.bayes_net import BayesNet
from bnn_spatial.metrics.sampling import compute_rhat
from bnn_spatial.metrics.prediction import rmspe, perc_coverage, interval_score

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # for handling OMP: Error 15 (2022/11/21)

#plt.rcParams['figure.figsize'] = [14, 7]
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['image.aspect'] = 'auto'
plt.rcParams['font.size'] = 24
plt.rcParams['axes.titlesize'] = 24
#mpl.use('TkAgg')
#mpl.rcParams['text.usetex'] = True
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
hidden_dims = [40, 40, 40]  # list of hidden layer dimensions
depth = len(hidden_dims)
embed_width = hidden_dims[0]
transfer_fn = "tanh"  # activation function

to_save = {'hidden_dims': hidden_dims,
           'transfer_fn': transfer_fn}
pickle.dump(to_save, stage1_file)

"""
Fixed BNN prior (non-optimised)
"""

n_test_h = 64  # width of input matrix
n_test_v = 64  # height of input matrix
margin_h = 4  # horizontal margin of domain
margin_v = 4  # vertical margin of domain
test_range_h = np.linspace(-margin_h, margin_h, n_test_h)
test_range_v = np.linspace(-margin_v, margin_v, n_test_v)
X1, X2 = np.meshgrid(test_range_h, test_range_v)  # Cartesian indexing (xy) is default
test_array = np.vstack((X1.flatten(), X2.flatten())).T  # one col for each of X1 and X2, shape (n_test_h*n_test_v, 2)
test_tensor = torch.from_numpy(test_array).to(device).float()
n_plot = 1000  # number of samples to use in plots (for BNN priors and GP only)

n_test = test_range = None
if n_test_h == n_test_v:
    n_test = n_test_h
    test_range = test_range_h

# Note: the flatten operation (row-wise, default C-style) can be reversed by .reshape(n_test, -1)

to_save = {'n_test_h': n_test_h,
           'n_test_v': n_test_v,
           'n_plot': n_plot}
pickle.dump(to_save, stage1_file)

# Initialize fixed BNN
std_bnn = GaussianNet(input_dim=2,
                      output_dim=1,
                      activation_fn=transfer_fn,
                      hidden_dims=hidden_dims,
                      prior_per='layer',
                      fit_means=False).to(device)

# Perform predictions using n_plot sets of sampled weights/biases
std_bnn_samples = std_bnn.sample_functions(test_tensor, n_plot).detach().cpu().numpy().squeeze()
# shape of tensor is (n_test ** 2, n_plot); rows for inputs, cols for samples

# Resize array of predictions
std_bnn_samples_ = np.zeros((n_plot, n_test_v, n_test_h))
for ss in range(n_plot):
    sample = std_bnn_samples[:, ss].T
    sample_ = sample.reshape(n_test_v, -1)
    std_bnn_samples_[ss, :, :] = sample_

# Obtain mean and std dev of samples
std_bnn_mean = np.mean(std_bnn_samples_, axis=0)
std_bnn_sd = np.std(std_bnn_samples_, axis=0)

# Plot 4-by-4 panels of samples
plot_samples_2d(samples=std_bnn_samples_,
                extent=[-margin_h, margin_h, -margin_v, margin_v])
plt.savefig(FIG_DIR + '/std_prior_samples.png', bbox_inches='tight')

"""
GP prior
"""

# GP kernel hyperparameters
sn2 = 0.001   # measurement error variance
leng = 1.0  # lengthscale
ampl = 1.0  # amplitude

# Specify GP kernel
kernel_class = base.Isotropic
kernel = kernel_class(cov=kernels.RBF,
                      ampl=ampl,
                      leng=leng,
                      power=2)

to_save = {'sn2': sn2,
           'leng': leng,
           'ampl': ampl,
           'kernel': kernel}
pickle.dump(to_save, stage1_file)

# Initialize GP prior
gp = GP(kern=kernel).to(device)

# Sample functions from GP prior
gp_samples = gp.sample_functions(test_tensor, n_plot + 1).detach().cpu().numpy().squeeze()

# Resize array of predictions
gp_samples_ = np.zeros((n_plot, n_test_v, n_test_h))
for ss in range(n_plot + 1):
    sample = gp_samples[:, ss].T
    sample_ = sample.reshape(n_test_v, -1)
    if ss < n_plot:
        gp_samples_[ss, :, :] = sample_

# Use the final sample as the latent function (simulated from GP)
gp_latent = sample
gp_latent_ = sample_

# Obtain mean and std dev of samples
gp_mean = np.mean(gp_samples_, axis=0)
gp_sd = np.std(gp_samples_, axis=0)

# Plot 4-by-4 panels of samples
plot_samples_2d(samples=gp_samples_,
                extent=[-margin_h, margin_h, -margin_v, margin_v])
plt.savefig(FIG_DIR + '/GP_prior_samples.png', bbox_inches='tight')

"""
GP-induced BNN prior
"""

# Initialize BNN to be optimised
opt_bnn = GaussianNet(input_dim=2,
                      output_dim=1,
                      activation_fn=transfer_fn,
                      hidden_dims=hidden_dims,
                      prior_per='layer',
                      fit_means=False).to(device)

# Use a grid of `n_data` points for the measurement set (`n_data` specified in MapperWasserstein)
data_generator = GridGenerator(-4, 4, input_dim=2)

# Perform Wasserstein optimisation to match BNN prior (opt_bnn) with GP prior
mapper_num_iters = 2000
outer_lr = 0.003  # initial learning rate for outer optimiser (wrt psi)
inner_lr = 0.01  # initial learning rate for inner optimiser (wrt theta)
inner_steps = 50  # number of steps for inner optimisation loop
n_measure = n_test_h*n_test_v
n_stochastic = 1024

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
                               n_data=n_measure,
                               n_gpu=1,
                               gpu_gp=True,
                               save_memory=False,
                               continue_training=True)  # specify pretraining of Lipschitz network
    w_hist = mapper.optimise(num_iters=mapper_num_iters,
                             n_samples=n_stochastic,  # originally 512
                             lr=outer_lr,
                             print_every=10)
    path = os.path.join(OUT_DIR, "wsr_values.log")
    np.savetxt(path, w_hist, fmt='%.6e')

# Obtain Wasserstein distance values for each iteration
wdist_file = os.path.join(OUT_DIR, "wsr_values.log")
wdist_vals = np.loadtxt(wdist_file)
indices = np.arange(mapper_num_iters-1)[::1]  # indices for use in plots (::step)

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

# Plot Lipschitz gradient norms to check if unit norm is approximately satisfied (as enforced by the penalty term)
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

# Note: In test_tensor, 1st index (x1) changes fastest

# Compute GP covariances, first horizontal slices (fixing x2 value)
gp_covs = [None] * n_test_v
for i in range(n_test_v):
    x_inputs = test_tensor[i*n_test_v:(i+1)*n_test_v, :]  # take all (*, test_range[i]) inputs
    gp_covs[i] = kernel.K(x_inputs).detach().cpu().numpy()  # GP covariance matrix for the i-th slice
K_h = np.mean(gp_covs, axis=0)

# Do the same for vertical slices (fixing x1 value)
gp_covs = [None] * n_test_h
for j in range(n_test_h):
    x_inputs = test_tensor[j::n_test_v, :]  # take all (test_range[j], *) inputs
    gp_covs[j] = kernel.K(x_inputs).detach().cpu().numpy()  # GP covariance matrix for the j-th slice
K_v = np.mean(gp_covs, axis=0)

# Compute GP covariance function values (when n_test_h = n_test_v)
if n_test is not None:
    gp_cov_h = gp_cov_v = 0
    for cc in range(n_test // 2):
        gp_cov_h += K_h[cc, cc:cc + (n_test // 2)]
        gp_cov_v += K_v[cc, cc:cc + (n_test // 2)]
    gp_cov_h /= n_test // 2
    gp_cov_v /= n_test // 2
    gp_cov_mean = (gp_cov_v + gp_cov_h) / 2

    # Plot the GP covariance (overlaid by optimised BNN covariances later)
    plt.figure(num='cov')
    plt.plot(test_range[n_test // 2:], gp_cov_mean, 'k-', lw=4, label='GP')

# Averaged covariance matrix estimates
K_avg = (K_h + K_v) / 2
heatmap_gp_covs = [K_h, K_v, K_avg]

# Plot heatmaps for GP covariance matrices
heatmap_titles = ['Horizontal', 'Vertical', 'Overall']
plot_cov_heatmap(covs=heatmap_gp_covs,
                 titles=heatmap_titles,
                 domain=test_tensor)
plt.savefig(FIG_DIR + '/GP_slice_cov_heatmaps.png', bbox_inches='tight')

# Plot the sampled predictions for desired checkpoints
plt_checkpoints = np.unique([10/50, 1, 2] + list(range(4, num_ckpt, 4)) + [num_ckpt])
plt_checkpoints.sort()  # ascending order by default
if num_ckpt // 2 not in plt_checkpoints:
    raise Exception('Require mid-checkpoint to be contained in plt_checkpoints')
heatmap_save = np.unique(np.array([2, 4, 8, num_ckpt // 2, num_ckpt]))
if len(heatmap_save) / 2 == len(heatmap_save) // 2:
    heatmap_save = np.insert(heatmap_save, obj=0, values=1)
heatmap_covs = [None] * (len(heatmap_save) + 1)  # add 1 for GP cov matrix
opt_bnn_mean0 = None
opt_bnn_sd0 = None

for ckpt in plt_checkpoints:

    # Load optimised BNN from saved checkpoint (dictionary with parameter values)
    ckpt_path = os.path.join(OUT_DIR, "ckpts", "it-{}.ckpt".format(int(50*ckpt)))
    opt_bnn.load_state_dict(torch.load(ckpt_path))

    # Perform predictions using n_plot sets of sampled weights/biases - rows inputs, cols samples
    opt_bnn_samples = opt_bnn.sample_functions(test_tensor, n_plot).detach().cpu().numpy().squeeze()

    # Resize array of predictions
    opt_bnn_samples_ = np.zeros((n_plot, n_test_v, n_test_h))
    for ss in range(n_plot):
        sample = opt_bnn_samples[:, ss].T
        sample_ = sample.reshape(n_test_v, -1)
        opt_bnn_samples_[ss, :, :] = sample_

    if (len(plt_checkpoints) > 1) & (ckpt == plt_checkpoints[0]):
        # Obtain mean and std dev of samples
        opt_bnn_mean0 = np.mean(opt_bnn_samples_, axis=0).squeeze()
        opt_bnn_sd0 = np.std(opt_bnn_samples_, axis=0).squeeze()

    # Obtain mean and std dev of samples
    opt_bnn_mean = np.mean(opt_bnn_samples_, axis=0).squeeze()
    opt_bnn_sd = np.std(opt_bnn_samples_, axis=0).squeeze()

    if (len(plt_checkpoints) > 1) & (ckpt < num_ckpt) & (ckpt > plt_checkpoints[0]):
        # Plot the optimised BNN at this checkpoint
        plot_mean_sd(mean_grid=opt_bnn_mean,
                     sd_grid=opt_bnn_sd,
                     domain=test_tensor)
        plt.savefig(FIG_DIR + '/GPiG_prior_step{}.png'.format(int(50*ckpt)), bbox_inches='tight')

    # Plot 4-by-4 panels of samples
    plot_samples_2d(samples=opt_bnn_samples_,
                    extent=[-margin_h, margin_h, -margin_v, margin_v])
    plt.savefig(FIG_DIR + '/GPiG_prior_samples_step{}.png'.format(int(50*ckpt)), bbox_inches='tight')

    ###################################
    # Empirical covariance computations

    # Vertical slices first
    bnn_covs = [None] * n_test_h
    for i in range(n_test_h):
        opt_bnn_slice = opt_bnn_samples_[:, :, i]  # shape is (n_plot, n_test_v)
        bnn_covs[i] = opt_bnn_slice.T @ opt_bnn_slice / (n_plot - 1)  # cov matrix for the i-th vertical slice
    K_v_bnn = np.mean(bnn_covs, axis=0)  # average sample cov matrices over all vertical slices

    # Now horizontal slices
    bnn_covs = [None] * n_test_v
    for j in range(n_test_v):
        opt_bnn_slice = opt_bnn_samples_[:, j, :]  # shape is (n_plot, n_test_h)
        bnn_covs[j] = opt_bnn_slice.T @ opt_bnn_slice / (n_plot - 1)  # cov matrix for the j-th horizontal slice
    K_h_bnn = np.mean(bnn_covs, axis=0)  # average sample cov matrices over all horizontal slices

    # Compute covariance function values (when n_test_h = n_test_v)
    if n_test is not None and ckpt in heatmap_save:
        bnn_cov_h = bnn_cov_v = 0
        for cc in range(n_test // 2):
            bnn_cov_h += K_h_bnn[cc, cc:cc + (n_test // 2)]
            bnn_cov_v += K_v_bnn[cc, cc:cc + (n_test // 2)]
        bnn_cov_h /= n_test // 2
        bnn_cov_v /= n_test // 2
        bnn_cov_mean = (bnn_cov_v + bnn_cov_h) / 2

        # Plot the optimised BNN covariance function
        plt.figure(num='cov')
        plt.plot(test_range[n_test//2:], bnn_cov_mean, '-', lw=2, label='{}-step BNN'.format(int(50 * ckpt)))

    # Obtain overall average estimates
    K_avg_bnn = (K_h_bnn + K_v_bnn) / 2

    # Store covariance matrices for first and last checkpoints, and plot comparisons based on slices averaged
    if ckpt in heatmap_save:
        ckpt_idx = np.argwhere(ckpt == heatmap_save).squeeze()
        heatmap_covs[ckpt_idx] = K_avg_bnn

        # Plot heatmaps for BNN covariance matrices at select checkpoints
        heatmap_bnn_covs = [K_h_bnn, K_v_bnn, K_avg_bnn]
        heatmap_titles = ['Horizontal', 'Vertical', 'Overall']
        plot_cov_heatmap(covs=heatmap_bnn_covs,
                         titles=heatmap_titles,
                         domain=test_tensor)
        plt.savefig(FIG_DIR + '/BNN_slice_cov_heatmaps_step{}.png'.format(int(50 * ckpt)), bbox_inches='tight')

# Finalise covariance function plot (when n_test_h = n_test_v)
if n_test is not None:
    fig = plt.figure(num='cov')
    fig.tight_layout()
    fig.set_figwidth(24)
    fig.set_figheight(12)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=6, fontsize='large')
    plt.savefig(FIG_DIR + '/GP_BNN_covariances.png', bbox_inches='tight')

# Plot covariance heatmaps
heatmap_covs[-1] = K_avg  # store GP covariance matrix for comparison
heatmap_titles = ['{}-step BNN'.format(int(50*ckpt)) for ckpt in heatmap_save]
heatmap_titles.append('GP')
plot_cov_heatmap(covs=heatmap_covs,
                 titles=heatmap_titles,
                 domain=test_tensor)
plt.savefig(FIG_DIR + '/GP_BNN_cov_heatmaps.png', bbox_inches='tight')

# Compute covariance matrices between all inputs
gp_big_cov_mx = kernel.K(test_tensor).detach().cpu().numpy()
bnn_big_cov_mx = np.cov(opt_bnn_samples) # final step BNN covariances
std_big_cov_mx = np.cov(std_bnn_samples) # final step BNN covariances

# Plot GP prior covariances, between uniformly selected points and the rest of the grid
cov_min, cov_max = plot_cov_nonstat(cov=gp_big_cov_mx,
                                    domain=test_tensor)
plt.savefig(FIG_DIR + '/GP_stat_cov_heatmaps.png', bbox_inches='tight')

# Do the same for GPi-G BNN covariances (using same points)
plot_cov_nonstat(cov=bnn_big_cov_mx,
                 domain=test_tensor,
                 cov_min=cov_min,
                 cov_max=cov_max)
plt.savefig(FIG_DIR + '/BNN_stat_cov_heatmaps.png', bbox_inches='tight')

# Do the same for fixed BNN covariances (using same points)
plot_cov_nonstat(cov=std_big_cov_mx,
                 domain=test_tensor)
plt.savefig(FIG_DIR + '/BNN_std_stat_cov_heatmaps.png', bbox_inches='tight')

# Plot differences of nonstationary covariances (K_GP - K_BNN)
plot_cov_nonstat_diff(cov=bnn_big_cov_mx,
                      gp_cov=gp_big_cov_mx,
                      domain=test_tensor)
plt.savefig(FIG_DIR + '/BNN_GP_stat_cov_diffs.png', bbox_inches='tight')

# Obtain min/max for SD and mean, for consistent value ranges in plots
sd_min_list = [np.min(gp_sd), np.min(std_bnn_sd), np.min(opt_bnn_sd)]
sd_max_list = [np.max(gp_sd), np.max(std_bnn_sd), np.max(opt_bnn_sd)]
mean_min_list = [np.min(gp_mean), np.min(std_bnn_mean), np.min(opt_bnn_mean)]
mean_max_list = [np.max(gp_mean), np.max(std_bnn_mean), np.max(opt_bnn_mean)]
if opt_bnn_sd0 is not None:
    sd_min_list.append(np.min(opt_bnn_sd0))
    sd_max_list.append(np.max(opt_bnn_sd0))
if opt_bnn_mean0 is not None:
    mean_min_list.append(np.min(opt_bnn_mean0))
    mean_max_list.append(np.max(opt_bnn_mean0))
sd_min0, sd_max0 = min(sd_min_list), max(sd_max_list)
mean_min0, mean_max0 = min(mean_min_list), max(mean_max_list)

# Plot the GP prior using sampled functions
plot_mean_sd(mean_grid=gp_mean,
             sd_grid=gp_sd,
             domain=test_tensor,
             sd_range=[sd_min0, sd_max0],
             mean_range=[mean_min0, mean_max0])
plt.savefig(FIG_DIR + '/GP_prior.png', bbox_inches='tight')

# Plot the sampled BNN predictions
plot_mean_sd(mean_grid=std_bnn_mean,
             sd_grid=std_bnn_sd,
             domain=test_tensor,
             sd_range=[sd_min0, sd_max0],
             mean_range=[mean_min0, mean_max0])
plt.savefig(FIG_DIR + '/std_prior.png', bbox_inches='tight')

if (opt_bnn_sd0 is not None) & (opt_bnn_mean0 is not None):
    # Plot an early GPiG prior using sampled functions
    plot_mean_sd(mean_grid=opt_bnn_mean0,
                 sd_grid=opt_bnn_sd0,
                 domain=test_tensor,
                 sd_range=[sd_min0, sd_max0],
                 mean_range=[mean_min0, mean_max0])
    plt.savefig(FIG_DIR + '/GPiG_prior_step{}.png'.format(int(plt_checkpoints[0]*50)), bbox_inches='tight')

# Plot the final GPiG prior using sampled functions
plot_mean_sd(mean_grid=opt_bnn_mean,
             sd_grid=opt_bnn_sd,
             domain=test_tensor,
             sd_range=[sd_min0, sd_max0],
             mean_range=[mean_min0, mean_max0])
plt.savefig(FIG_DIR + '/GPiG_prior.png', bbox_inches='tight')

"""
GP Posterior
"""

# Create data set
n_train = 100
inds = np.random.randint(0, test_array.shape[0], size=n_train)
X = test_array[inds, :]
err_cov = sn2 * np.eye(n_train)
err_mean = np.zeros(n_train)
y = gp_latent[inds] + np.random.multivariate_normal(err_mean, err_cov).flatten()
X_tensor = test_tensor[inds, :]
y_tensor = torch.from_numpy(y).reshape([-1, 1]).to(device)

# Plot latent function (to compare with sample mean plots)
plt.figure()
plt.contourf(X1, X2, gp_latent_, levels=256, cmap='Spectral_r', origin='lower')
plt.axis('scaled')
plt.colorbar(pad=0.01)
plt.savefig(FIG_DIR + '/latent_function.png', bbox_inches='tight')

# Make predictions
GP.assign_data(gp,
               X=X_tensor,
               Y=y_tensor,
               sn2=sn2)
gp_preds = gp.predict_f_samples(test_tensor, n_plot).detach().cpu().numpy().squeeze()

# Resize array of predictions
gp_preds_ = np.zeros((n_plot, n_test_v, n_test_h))
for ss in range(n_plot):
    sample = gp_preds[:, ss].T
    sample_ = sample.reshape(n_test_v, -1)
    gp_preds_[ss, :, :] = sample_

# Obtain mean and std dev of samples
gp_pred_mean = np.mean(gp_preds_, axis=0)
gp_pred_sd = np.std(gp_preds_, axis=0)

# Plot 4-by-4 panels of samples
plot_samples_2d(samples=gp_preds_,
                extent=[-margin_h, margin_h, -margin_v, margin_v])
plt.savefig(FIG_DIR + '/GP_posterior_samples.png', bbox_inches='tight')

# Save parameter settings used for stage 1 code
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
n_samples = 200
n_burn = 3000  # must be multiple of keep_every
n_burn_thin = n_burn // keep_every
n_discarded = 100 - n_burn_thin

sampler = 'adaptive_sghmc'
sghmc_lr = 1e-4
adaptive_lr = sghmc_lr ** 0.5
if sampler == 'sghmc':
    sampler_lr = sghmc_lr
elif sampler == 'adaptive_sghmc':
    sampler_lr = adaptive_lr
else:
    raise Exception('Only `sghmc` and `adaptive_sghmc` samplers are supported.')

sampling_configs = {
    "batch_size": 32,  # Mini-batch size
    "num_samples": n_samples,  # Number of samples kept for each chain (default 30)
    "n_discarded": n_discarded,  # Number of samples to discard after burn-in adaptation phase (default 10)
    "num_burn_in_steps": n_burn,  # Number of burn-in steps for adaptive SGHMC (default 2000)
    "keep_every": keep_every,  # Thinning interval (default 200)
    "lr": sampler_lr,  # Step size (1e-4 works well for standard SGHMC, and 1e-2 for adaptive)
    "mdecay": 0.05,  # Momentum coefficient
    "num_chains": n_chains,  # Number of Markov chains (for r_hat, at least 4 suggested)
    "print_every_n_samples": 50,  # Control feedback regularity
}

pickle.dump({'n_train': n_train, 'sampler': sampler, 'sampler_lr': sampler_lr}, stage2_file)
pickle.dump(sampling_configs, stage2_file)

# Initialize the prior and likelihood
prior = FixedGaussianPrior(mu=0, std=1)  # same settings as fixed BNN prior
likelihood = LikGaussian(sn2)

# Set up neural network for usage with SGHMC
net = BlankNet(output_dim=1,
               input_dim=2,
               hidden_dims=hidden_dims,
               activation_fn=transfer_fn).to(device)

# Compare named parameters of the neural networks
print('GaussianNet has named parameters:\n{}'.format([name for name, param in opt_bnn.named_parameters()]))
print('BlankNet has named parameters:\n{}'.format([name for name, param in net.named_parameters()]))

# Generate posterior BNN parameter samples with multiple chains of sampling
bayes_net_std = BayesNet(net, likelihood, prior,
                         sampling_method=sampler,
                         n_gpu=0)
std_weights, std_weights_pred = bayes_net_std.sample_multi_chains(X, y, **sampling_configs)

n_samples_all_chains = len(std_weights)  # total number of samples from all chains, including discarded burn-in samples
n_samples_pred_chains = len(std_weights_pred)
n_samples_none = len(list(filter(lambda item: item is None, std_weights)))
n_samples_pred_none = len(list(filter(lambda item: item is None, std_weights_pred)))
print('Sample quantities')
print('Total samples including discarded: {}\n'
      'None values in total samples: {}\n'
      'Samples used for prediction: {}\n'
      'None values in samples for prediction : {}\n'
      'Samples per chain * Number of chains : {}'
      .format(n_samples_all_chains, n_samples_none, n_samples_pred_chains, n_samples_pred_none, n_samples * n_chains))

# Create arrays of MCMC samples (samples cols, parameters rows)
std_chains = np.zeros((2 * depth, n_samples_all_chains))
std_chains_pred = np.zeros((2 * depth, n_chains * n_samples))

# Populate the array containing all samples
for k in range(n_samples_all_chains):
    std_dict = std_weights[k]  # dictionary of name-tensor pairs for network parameters
    W_tensors = list(std_dict.values())[0::2]
    b_tensors = list(std_dict.values())[1::2]
    for ll in range(2 * depth):
        if ll % 2 == 0:
            W_idx = tuple([0] * W_tensors[ll // 2].dim())
            std_chains[ll, k] = W_tensors[ll // 2][W_idx]  # use first weight entry
        else:
            b_idx = tuple([0] * b_tensors[(ll - 1) // 2].dim())
            std_chains[ll, k] = b_tensors[(ll - 1) // 2][b_idx]  # use first bias entry

# Populate the array only containing samples utilised for predictions
for k in range(n_samples * n_chains):
    std_dict = std_weights_pred[k]  # dictionary of name-tensor pairs for network parameters
    W_tensors = list(std_dict.values())[0::2]
    b_tensors = list(std_dict.values())[1::2]
    for ll in range(2 * depth):
        if ll % 2 == 0:
            W_idx = tuple([0] * W_tensors[ll // 2].dim())
            std_chains_pred[ll, k] = W_tensors[ll // 2][W_idx]  # use first weight entry
        else:
            b_idx = tuple([0] * b_tensors[(ll - 1) // 2].dim())
            std_chains_pred[ll, k] = b_tensors[(ll - 1) // 2][b_idx]  # use first bias entry

# Construct vector with subplot titles (accounts for varying network depth)
trace_titles = []
for lvl in range(depth):
    W_l = r'$\mathbf{{W}}_{}$'.format(lvl + 1)
    b_l = r'$\mathbf{{b}}_{}$'.format(lvl + 1)
    trace_titles.append(W_l)
    trace_titles.append(b_l)

# Construct vector of legend entries (accounts for varying chain length)
legend_entries = ['Chain {}'.format(cc+1) for cc in range(n_chains)]

# Trace plots of MCMC iterates
print('Plotting parameter traces (fixed BNN)')
plot_param_traces(param_chains=std_chains,
                  n_chains=n_chains,
                  net_depth=depth,
                  n_discarded=n_discarded,
                  n_burn=n_burn_thin,
                  trace_titles=trace_titles,
                  legend_entries=legend_entries)
plt.savefig(FIG_DIR + '/std_chains.png', bbox_inches='tight')

# Make predictions
bnn_std_preds, bnn_std_preds_all = bayes_net_std.predict(test_tensor)

# MCMC convergence diagnostics
r_hat = compute_rhat(bnn_std_preds, sampling_configs["num_chains"])
rhat_mean_fixed = float(r_hat.mean())
rhat_sd_fixed = float(r_hat.std())
print(r"R-hat: mean {:.4f} std {:.4f}".format(rhat_mean_fixed, rhat_sd_fixed))

to_save = {'rhat_mean_fixed': rhat_mean_fixed,
           'rhat_sd_fixed': rhat_sd_fixed}
pickle.dump(to_save, stage2_file)

# Resize array of predictions
bnn_std_preds_ = np.zeros((n_chains * n_samples, n_test_v, n_test_h))
for ss in range(n_chains * n_samples):
    sample = bnn_std_preds[ss, :]
    sample_ = sample.reshape(n_test_v, -1)
    bnn_std_preds_[ss, :, :] = sample_

# Obtain mean and std dev of samples
std_pred_mean = np.mean(bnn_std_preds_, axis=0)
std_pred_sd = np.std(bnn_std_preds_, axis=0)

# Plot 4-by-4 panels of samples
plot_samples_2d(samples=bnn_std_preds_,
                extent=[-margin_h, margin_h, -margin_v, margin_v])
plt.savefig(FIG_DIR + '/std_posterior_samples.png', bbox_inches='tight')

# Trace plot of network output at specified input point(s)
print('Plotting output traces (fixed BNN)')
plot_output_traces(domain=test_tensor,
                   preds=bnn_std_preds_all,
                   n_chains=n_chains,
                   n_discarded=n_discarded,
                   n_burn=n_burn_thin,
                   legend_entries=legend_entries)
plt.savefig(FIG_DIR + '/std_posterior_trace.png', bbox_inches='tight')

# Histogram of network output at same input point(s)
plot_output_hist(domain=test_tensor, preds=bnn_std_preds)
plt.savefig(FIG_DIR + '/std_posterior_hist.png', bbox_inches='tight')

# ACF of network output at same input point(s)
print('Plotting output ACF (fixed BNN)')
plot_output_acf(domain=test_tensor,
                preds=bnn_std_preds,
                n_chains=n_chains,
                n_samples_kept=n_samples)
plt.savefig(FIG_DIR + '/std_posterior_acf.png', bbox_inches='tight')

"""
SGHMC - GP-induced BNN posterior
"""

# Plot the estimated posteriors for desired checkpoints
mcmc_checkpoints = [num_ckpt]
for ckpt in mcmc_checkpoints:

    # Load the optimized prior
    ckpt_path = os.path.join(OUT_DIR, "ckpts", "it-{}.ckpt".format(int(50*ckpt)))
    prior = OptimGaussianPrior(ckpt_path)  # use optimised prior on each parameter

    # Obtain network parameter values (for plotting only)
    #opt_bnn.load_state_dict(torch.load(ckpt_path))

    # Generate posterior BNN parameter samples with multiple chains of sampling
    bayes_net_optim = BayesNet(net, likelihood, prior,
                               sampling_method=sampler,
                               n_gpu=0)
    optim_weights, optim_weights_pred = bayes_net_optim.sample_multi_chains(X, y, **sampling_configs)

    # Create arrays of MCMC samples (samples cols, parameters rows)
    optim_chains = np.zeros((2 * depth, n_samples_all_chains))
    optim_chains_pred = np.zeros((2 * depth, n_chains * n_samples))

    # Populate the array containing all samples
    for k in range(n_samples_all_chains):
        optim_dict = optim_weights[k]  # dictionary of name-tensor pairs for network parameters
        W_tensors = list(optim_dict.values())[0::2]
        b_tensors = list(optim_dict.values())[1::2]
        for ll in range(2 * depth):
            if ll % 2 == 0:
                W_idx = tuple([0] * W_tensors[ll // 2].dim())
                optim_chains[ll, k] = W_tensors[ll // 2][W_idx]  # use first weight entry
            else:
                b_idx = tuple([0] * b_tensors[(ll - 1) // 2].dim())
                optim_chains[ll, k] = b_tensors[(ll - 1) // 2][b_idx]  # use first bias entry

    # Populate the array only containing samples utilised for predictions
    for k in range(n_chains * n_samples):
        optim_dict = optim_weights_pred[k]  # dictionary of name-tensor pairs for network parameters
        W_tensors = list(optim_dict.values())[0::2]
        b_tensors = list(optim_dict.values())[1::2]
        for ll in range(2 * depth):
            if ll % 2 == 0:
                W_idx = tuple([0] * W_tensors[ll // 2].dim())
                optim_chains_pred[ll, k] = W_tensors[ll // 2][W_idx]  # use first weight entry
            else:
                b_idx = tuple([0] * b_tensors[(ll - 1) // 2].dim())
                optim_chains_pred[ll, k] = b_tensors[(ll - 1) // 2][b_idx]  # use first bias entry

    # Make predictions
    bnn_optim_preds, bnn_optim_preds_all = bayes_net_optim.predict(test_tensor)  # rows samples, cols traces

    # MCMC convergence diagnostics
    r_hat = compute_rhat(bnn_optim_preds, sampling_configs["num_chains"])
    rhat_mean_opt = float(r_hat.mean())
    rhat_sd_opt = float(r_hat.std())
    print(r"R-hat: mean {:.4f} std {:.4f}".format(rhat_mean_opt, rhat_sd_opt))

    if ckpt == num_ckpt:
        to_save = {'rhat_mean_opt': rhat_mean_opt,
                   'rhat_sd_opt': rhat_sd_opt}
        pickle.dump(to_save, stage2_file)

    # Resize array of predictions
    bnn_optim_preds_ = np.zeros((n_chains * n_samples, n_test_v, n_test_h))
    for ss in range(n_chains * n_samples):
        sample = bnn_optim_preds[ss, :]
        sample_ = sample.reshape(n_test_v, -1)
        bnn_optim_preds_[ss, :, :] = sample_

    # Obtain mean and std dev of samples
    opt_pred_mean = np.mean(bnn_optim_preds_, axis=0).squeeze()
    opt_pred_sd = np.std(bnn_optim_preds_, axis=0).squeeze()

    if ckpt == num_ckpt:

        #################
        # Parameter plots

        # Trace plots of MCMC iterates
        print('Plotting parameter traces (optimised BNN)')
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
        print('Plotting output traces (optimised BNN)')
        plot_output_traces(domain=test_tensor,
                           preds=bnn_optim_preds_all,
                           n_chains=n_chains,
                           n_discarded=n_discarded,
                           n_burn=n_burn_thin,
                           legend_entries=legend_entries)
        plt.savefig(FIG_DIR + '/GPiG_posterior_trace.png', bbox_inches='tight')

        # Histogram of network output at same input point(s)
        plot_output_hist(domain=test_tensor, preds=bnn_optim_preds)
        plt.savefig(FIG_DIR + '/GPiG_posterior_hist.png', bbox_inches='tight')

        # ACF of network output at same input point(s)
        print('Plotting output ACF (optimised BNN)')
        plot_output_acf(domain=test_tensor,
                        preds=bnn_optim_preds,
                        n_chains=n_chains,
                        n_samples_kept=n_samples)
        plt.savefig(FIG_DIR + '/GPiG_posterior_acf.png', bbox_inches='tight')

    if (len(mcmc_checkpoints) > 1) & (ckpt < num_ckpt):
        # Plot the sampled predictions from the GP-induced BNN posterior (at each checkpoint)
        plot_mean_sd(mean_grid=opt_pred_mean,
                     sd_grid=opt_pred_sd,
                     domain=test_tensor,
                     obs=X)
        plt.savefig(FIG_DIR + '/GPiG_posterior_step{}.png'.format(int(50*ckpt)), bbox_inches='tight')

    # Plot 4-by-4 panels of samples
    plot_samples_2d(samples=bnn_optim_preds_,
                    extent=[-margin_h, margin_h, -margin_v, margin_v])
    plt.savefig(FIG_DIR + '/GPiG_posterior_samples_step{}.png'.format(int(50 * ckpt)), bbox_inches='tight')

# Obtain min/max for SD and mean, for consistent value ranges in plots
sd_min_list = [np.min(gp_pred_sd), np.min(std_pred_sd), np.min(opt_pred_sd)]
sd_max_list = [np.max(gp_pred_sd), np.max(std_pred_sd), np.max(opt_pred_sd)]
mean_min_list = [np.min(gp_pred_mean), np.min(std_pred_mean), np.min(opt_pred_mean)]
mean_max_list = [np.max(gp_pred_mean), np.max(std_pred_mean), np.max(opt_pred_mean)]
sd_min, sd_max = min(sd_min_list), max(sd_max_list)
mean_min, mean_max = min(mean_min_list), max(mean_max_list)

# Plot the sampled predictions from the GP posterior
plot_mean_sd(mean_grid=gp_pred_mean,
             sd_grid=gp_pred_sd,
             domain=test_tensor,
             obs=X,
             sd_range=[sd_min, sd_max],
             mean_range=[mean_min, mean_max])
plt.savefig(FIG_DIR + '/GP_posterior.png', bbox_inches='tight')

# Plot the sampled predictions from the fixed BNN posterior
plot_mean_sd(mean_grid=std_pred_mean,
             sd_grid=std_pred_sd,
             domain=test_tensor,
             obs=X,
             sd_range=[sd_min, sd_max],
             mean_range=[mean_min, mean_max])
plt.savefig(FIG_DIR + '/std_posterior.png', bbox_inches='tight')

# Plot the sampled predictions from the GP-induced BNN posterior (at each checkpoint)
plot_mean_sd(mean_grid=opt_pred_mean,
             sd_grid=opt_pred_sd,
             domain=test_tensor,
             obs=X,
             sd_range=[sd_min, sd_max],
             mean_range=[mean_min, mean_max])
plt.savefig(FIG_DIR + '/GPiG_posterior.png', bbox_inches='tight')

# Save all parameter settings for stage 1 code
stage2_file.close()
stage2_file = open(OUT_DIR + "/stage2.file", "rb")
while True:
    try:
        stage2_txt.write(str(pickle.load(stage2_file)) + '\n')
    except EOFError:
        break



