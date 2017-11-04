from collections import OrderedDict
import tensorflow as tf
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
from matplotlib.pyplot import *
import seaborn as sns
from tqdm import tqdm
ds = tf.contrib.distributions
slim = tf.contrib.slim
graph_replace = tf.contrib.graph_editor.graph_replace
 
from keras.optimizers import Adam
 
 
 
 
def sample_mog(batch_size, n_mixture=8, std=0.02, radius=2.0):
    thetas = np.linspace(0, 2 * np.pi * (n_mixture-1)/float(n_mixture), n_mixture)
    xs, ys = radius * np.sin(thetas), radius * np.cos(thetas)
    cat = ds.Categorical(tf.zeros(n_mixture))
    comps = [ds.MultivariateNormalDiag([xi, yi], [std, std]) for xi, yi in zip(xs.ravel(), ys.ravel())]
    data = ds.Mixture(cat, comps)
    return data.sample(batch_size)
 
 
 
params = dict(
    batch_size=512,
    disc_learning_rate=1e-4,
    gen_learning_rate=1e-4,
    beta1=0.5,
    epsilon=1e-8,
    max_iter=100001,
    viz_every=1000,
    z_dim=256,
    x_dim=2,
    unrolling_steps=0,
    regularizer_weight=0.5,
)


 

def extract_step_dict(update_ops):
    """Extract variables and their new values from Assign and AssignAdd ops.
    
    Args:
        update_ops: list of Assign and AssignAdd ops, typically computed using Keras' opt.get_updates()

    Returns:
        dict mapping from variables to their step updates
    """
    name_to_var = {v.name: v for v in tf.global_variables()}
    updates = OrderedDict()
    for update in update_ops:
        var_name = update.op.inputs[0].name
        var = name_to_var[var_name]
        value = update.op.inputs[1]
        if update.op.type == 'Assign':
            updates[var] = value - var
        elif update.op.type == 'AssignAdd':
            updates[var] =  value
        else:
            raise ValueError("Update op type (%s) must be of type Assign or AssignAdd"%update_op.op.type)
    return updates


 
 
def generator(z, output_dim=2, n_hidden=128, n_layer=2):
    with tf.variable_scope("generator"):
        h = slim.stack(z, slim.fully_connected, [n_hidden] * n_layer, activation_fn=tf.nn.relu)
        x = slim.fully_connected(h, output_dim, activation_fn=None)
    return x
 
def discriminator(x, n_hidden=128, n_layer=1, reuse=False):
    with tf.variable_scope("discriminator", reuse=reuse):
        h = slim.stack(tf.divide(x,4.0), slim.fully_connected, [n_hidden] * n_layer, activation_fn=tf.nn.relu)
        log_d = slim.fully_connected(h, 1, activation_fn=None)
    return log_d
 
tf.reset_default_graph()
 
data = sample_mog(params['batch_size'])
 
noise = ds.Normal(tf.zeros(params['z_dim']), 
                  tf.ones(params['z_dim'])).sample(params['batch_size'])
# Construct generator and discriminator nets
with slim.arg_scope([slim.fully_connected], weights_initializer=tf.orthogonal_initializer(gain=0.8)):
    samples = generator(noise, output_dim=params['x_dim'])
    real_score = discriminator(data)
    fake_score = discriminator(samples, reuse=True)
     
# D maximizes this, G minimizes this + a regularizer
V = -tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=real_score, labels=tf.ones_like(real_score)) +
    tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_score, labels=tf.zeros_like(fake_score)))
 


gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "generator")
disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "discriminator")
 
# Vanilla discriminator update
d_opt = Adam(lr=params['disc_learning_rate'], beta_1=params['beta1'], epsilon=params['epsilon'])
d_updates = d_opt.get_updates(disc_vars, [], -V)
d_train_op = tf.group(*d_updates, name="d_train_op")
step_dict = extract_step_dict(d_updates)

# Penalty
d_gradient_norm_sq = tf.square(tf.global_norm(tf.gradients(V, disc_vars)))


g_opt = Adam(lr=params['gen_learning_rate'], beta_1=params['beta1'], epsilon=params['epsilon'])
g_updates = g_opt.get_updates(gen_vars, [], V +params['regularizer_weight']*d_gradient_norm_sq)
g_train_op = tf.group(*g_updates, name="g_train_op")


# Or alternatively:
'''
d_opt = tf.train.AdamOptimizer(params['disc_learning_rate'], beta1=params['beta1'], epsilon=params['epsilon'])
g_opt = tf.train.AdamOptimizer(params['gen_learning_rate'], beta1=params['beta1'], epsilon=params['epsilon'])
d_train_op = d_opt.minimize(-V, var_list=disc_vars)
g_train_op = g_opt.minimize(V+params['regularizer_weight']*d_gradient_norm_sq, var_list=gen_vars)
'''



norm_d = tf.global_norm(tf.gradients(V, disc_vars))
norm_g = tf.global_norm(tf.gradients(V, gen_vars))
 
sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
sess.run(tf.global_variables_initializer())
 

xmax = 3
fs = []
frames = []
np_samples = []
ds = [] # first gradients 
gs = []
n_batches_viz = 10
viz_every = params['viz_every']
for i in tqdm(range(params['max_iter'])):
    sess.run(d_train_op)
    sess.run(g_train_op)
    d, g, f, = sess.run([norm_d, norm_g, V])
    fs.append(f)    
    ds.append(d)
    gs.append(g)
    if i % viz_every == 0:
        np_samples.append(np.vstack([sess.run(samples) for _ in range(n_batches_viz)]))
        xx, yy = sess.run([samples, data])
        fig = figure(figsize=(5,5))
        scatter(xx[:, 0], xx[:, 1], edgecolor='none')
        scatter(yy[:, 0], yy[:, 1], c='g', edgecolor='none')
        axis('off')

        fig.savefig('fig'+str(i)+'.pdf') 
        close(fig)
 

np.savetxt("d_norm.out",np.array(ds))
np.savetxt("g_norm.out",np.array(gs))
fig = figure()        
ax = subplot(111)
ax.set_ylabel('Discriminator Gradient L2 Norm')
ax.set_xlabel('Iteration')
plot(range(len(ds)), ds)
fig.savefig('d_norm.pdf')
fig = figure()
ax = subplot(111)
plot(range(len(gs)), gs)
ax.set_ylabel('Generator Gradient L2 Norm')
ax.set_xlabel('Iteration')
fig.savefig('g_norm.pdf')




np_samples_ = np_samples[::1]
cols = len(np_samples_)
bg_color  = sns.color_palette('Greens', n_colors=256)[0]
fig=figure(figsize=(2*cols, 2))
for i, samps in enumerate(np_samples_):
    if i == 0:
        ax = subplot(1,cols,1)
    else:
        subplot(1,cols,i+1, sharex=ax, sharey=ax)
    ax2 = sns.kdeplot(samps[:, 0], samps[:, 1], shade=True, cmap='Greens', n_levels=20, clip=[[-xmax,xmax]]*2)
    ax2.set_axis_bgcolor(bg_color)
    xticks([]); yticks([])
    title('step %d'%(i*viz_every))
ax.set_ylabel('%d unrolling steps'%params['unrolling_steps'])
gcf().tight_layout()
fig.savefig('series.pdf')

np.savetxt("loss.out",np.array(fs))
fig=figure()
fs = np.array(fs)
plot(range(len(fs)),fs)
ax = subplot(111)
ax.set_ylabel('Loss')
ax.set_xlabel('Iteration')
fig.savefig('loss.pdf')
