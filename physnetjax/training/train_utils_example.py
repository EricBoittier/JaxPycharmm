# based on an example from Thorben Frank 
# https://github.com/EricBoittier/euclidean_fast_attention/commit/c348a37e2fdb04d4b1c94635da24ab5dadae4a83#diff-2e2af370f7a8ba0009c6dcc806435fa1180ba7ccd1fb95df9f1de6dcda36b834
 import jraph
 import jax
 import jax.numpy as jnp
 import optax
 
 from clu import metrics
 from flax import struct as flax_struct
 
 
 @flax_struct.dataclass
 class EvalMetrics(metrics.Collection):
     loss: metrics.Average.from_output('loss')
     energy_mse: metrics.Average.from_output('energy_mse')
     forces_mse: metrics.Average.from_output('forces_mse')
 
 
 @flax_struct.dataclass
 class TrainMetrics(metrics.Collection):
     loss: metrics.Average.from_output('loss')
     grad_norm: metrics.Average.from_output('grad_norm')
     energy_mse: metrics.Average.from_output('energy_mse')
     forces_mse: metrics.Average.from_output('forces_mse')
 
 
 def graph_mse_loss(y, y_label, batch_segments, graph_mask, scale):
     del batch_segments
 
     assert y.shape == y_label.shape
 
     full_mask = ~jnp.isnan(
         y_label
     ) & jnp.expand_dims(
         graph_mask, [y_label.ndim - 1 - o for o in range(0, y_label.ndim - 1)]
     )
     denominator = full_mask.sum().astype(y.dtype)
     mse = (
             jnp.sum(
                 2 * scale * optax.l2_loss(
                     jnp.where(full_mask, y, 0).reshape(-1),
                     jnp.where(full_mask, y_label, 0).reshape(-1),
                 )
             )
             / denominator
     )
     return mse
 
 
 def node_mse_loss(y, y_label, batch_segments, graph_mask, scale):
 
     assert y.shape == y_label.shape
 
     num_graphs = graph_mask.sum().astype(y.dtype)  # ()
 
     squared = 2 * optax.l2_loss(
         predictions=y,
         targets=y_label,
     )  # same shape as y
 
     # sum up the l2_losses for node properties along the non-leading dimension. For e.g. scalar node quantities
     # this does not have any effect, but e.g. for vectorial and tensorial node properties one averages over all
     # additional non-leading dimension. E.g. for forces this corresponds to taking mean over x, y, z component.
     node_mean_squared = squared.reshape(len(squared), -1).mean(axis=-1)  # (num_nodes)
 
     per_graph_mse = jraph.segment_mean(
         data=node_mean_squared,
         segment_ids=batch_segments,
         num_segments=len(graph_mask)
     )  # (num_graphs)
 
     # Set contributions from padding graphs to zero.
     per_graph_mse = jnp.where(
         graph_mask,
         per_graph_mse,
         jnp.asarray(0., dtype=per_graph_mse.dtype)
     )  # (num_graphs)
 
     # Calculate mean and scale.
     mse = scale * jnp.sum(per_graph_mse) / num_graphs  # ()
 
     return mse
 
 
 def make_loss_fn(model, energy_weight, forces_weight):
     """Make loss function, given a model and energy and force weights.
 
   Args:
     model: A `flax.linen.Module`.
     energy_weight: Energy weight in the loss function.
     forces_weight: Force weight in the loss function.
 
   Returns:
 
   """
 
     def loss_fn(params, batch):
         energy, forces = model.apply(
             params,
             positions=batch['positions'],
             atomic_numbers=batch['atomic_numbers'],
             atomic_dipoles=batch['atomic_dipoles'],
             src_idx=batch['src_idx'],
             dst_idx=batch['dst_idx'],
             batch_segments=batch['batch_segments'],
             graph_mask=batch['graph_mask'],
         )
         energy_label, forces_label = batch['energy'], batch['forces']
         graph_mask, node_mask = batch['graph_mask'], batch['node_mask']
 
         if not (len(node_mask) == len(forces_label) == len(forces)):
             raise ValueError(
                 '`node_mask`, `forces_label` and `forces` must have the same length.'
                 f' They have shapes {node_mask.shape}, {forces_label.shape} and'
                 f' {forces.shape}.'
             )
         if not (energy.shape == energy_label.shape == graph_mask.shape):
             raise ValueError(
                 '`energy`, `energy_label` and `graph_mask` must have the same shape.'
                 f' They have shapes {energy.shape}, {energy_label.shape} and'
                 f' {graph_mask.shape}.'
             )
 
         energy_mse = graph_mse_loss(
             y=energy,
             y_label=energy_label,
             batch_segments=batch['batch_segments'],
             graph_mask=graph_mask,
             scale=1.
         )
 
         forces_mse = node_mse_loss(
             y=forces,
             y_label=forces_label,
             batch_segments=batch['batch_segments'],
             graph_mask=graph_mask,
             scale=1.
         )
 
         loss = energy_weight * energy_mse + forces_weight * forces_mse
         return loss, (energy_mse, forces_mse)
 
     return loss_fn
 
 
 def make_inference_fn(params, model):
     """Make inference function, given a model and parameter tree.
 
   Args:
     params: PyTree of parameters.
     model: FLAX model.
 
   Returns:
     Inference function, that returns energy and forces.
   """
 
     def inference_fn(batch):
         energy, forces = model.apply(
             params,
             positions=batch['positions'],
             atomic_numbers=batch['atomic_numbers'],
             atomic_dipoles=batch['atomic_dipoles'],
             src_idx=batch['src_idx'],
             dst_idx=batch['dst_idx'],
             batch_segments=batch['batch_segments'],
             graph_mask=batch['graph_mask'],
         )
         return energy, forces
 
     return inference_fn
 
 
 def make_train_step_fn(loss_fn, optimizer):
     """Make train step function, given a loss function and `optax.optimizer`.
 
     Args:
       loss_fn: Loss function.
       optimizer: Optimizer.
 
     Returns:
       Train step function, that takes `params`, `opt_state` and `batch` and
       returns updated `params`, `opt_state` and `metrics.Collection`.
     """
 
     def train_step_fn(params, opt_state, batch):
         (loss, (energy_mse, forces_mse)), grad = jax.value_and_grad(
             loss_fn, has_aux=True
         )(params, batch)
         updates, opt_state = optimizer.update(grad, opt_state, params=params)
         params = optax.apply_updates(params, updates)
         grad_norm = optax.global_norm(grad)
 
         mtrcs = {
             'loss': loss,
             'energy_mse': energy_mse,
             'forces_mse': forces_mse,
             'grad_norm': grad_norm,
         }
 
         metrics_update = TrainMetrics.single_from_model_output(**mtrcs)
 
         return params, opt_state, metrics_update
 
     return train_step_fn
 
 
 def make_eval_step_fn(loss_fn):
     """Make eval step function, given a loss function.
 
     Args:
       loss_fn: Loss function.
 
    Returns:
       Evaluation function, that takes `params` and `batch` as input and returns
       `metrics.Collection`.
    """
 
     def eval_step_fn(params, batch):
         loss, (energy_mse, forces_mse) = loss_fn(params, batch)
         mtrcs = {'loss': loss, 'energy_mse': energy_mse, 'forces_mse': forces_mse}
         metrics_update = EvalMetrics.single_from_model_output(**mtrcs)
         return metrics_update
 
     return eval_step_fn
 
 
 def mean_absolute_error(a, b, msk):
     return jnp.mean(jnp.abs(a[msk].reshape(-1) - b[msk].reshape(-1)))
 
 
 def mean_squared_error(a, b, msk):
     return jnp.mean(jnp.square(a[msk].reshape(-1) - b[msk].reshape(-1)))
 
 
 @jax.jit
 def collect_metrics(mtrcs):
     return jax.tree_map(lambda *args: jnp.mean(jnp.stack(args)), *mtrcs)