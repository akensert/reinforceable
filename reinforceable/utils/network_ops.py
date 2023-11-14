import tensorflow as tf


def compute_shared_regularization_loss(
    variable_set_a: list[tf.Variable],
    variable_set_b: list[tf.Variable],
    coefs: tuple[float, float, float],
    regularization_type: str,
) -> float:

    '''Computes regularization loss for shared variables.'''

    if regularization_type.lower().startswith('l1'):
        penalty_fn = _l1_reg_penalty
    else:
        penalty_fn = _l2_reg_penalty

    kernels_a = [v for v in variable_set_a if 'kernel' in v.name]
    kernels_b = [v for v in variable_set_b if 'kernel' in v.name]

    shared_kernels, unshared_kernels_a, unshared_kernels_b = (
        _separate_variable_sets(kernels_a, kernels_b)
    )

    return tf.add_n(
          [penalty_fn(v, coefs[0]) for v in shared_kernels] 
        + [penalty_fn(v, coefs[1]) for v in unshared_kernels_a] 
        + [penalty_fn(v, coefs[2]) for v in unshared_kernels_b]
    )

def _l1_reg_penalty(x: tf.Variable, weight: float) -> float:
    return tf.reduce_sum(tf.abs(x)) * weight

def _l2_reg_penalty(x: tf.Variable, weight: float) -> float:
    return tf.reduce_sum(tf.square(x)) * weight

def _separate_variable_sets(
    variable_set_a: list[tf.Variable], 
    variable_set_b: list[tf.Variable]
) -> tuple[list[tf.Variable], list[tf.Variable], list[tf.Variable]]:
    '''Separates variables into three sets: shared, unshared_a, unshared_b.'''
    var_set_a = set([x.ref() for x in variable_set_a])
    var_set_b = set([x.ref() for x in variable_set_b])
    shared_var_set = var_set_a.intersection(var_set_b)
    unshared_var_set_a = var_set_a.difference(var_set_b)
    unshared_var_set_b = var_set_b.difference(var_set_a)
    shared_var_list = list(x.deref() for x in shared_var_set)
    unshared_var_list_a = list(x.deref() for x in unshared_var_set_a)
    unshared_var_list_b = list(x.deref() for x in unshared_var_set_b)
    return shared_var_list, unshared_var_list_a, unshared_var_list_b