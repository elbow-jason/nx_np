defmodule NxNp.Random.Weibull do
#     defp min(key, scale, concentration, shape=(), dtype=dtypes.float_):
#     """Sample from a Weibull distribution.
#     The scipy counterpart is `scipy.stats.weibull_min`.
#     Args:
#       key: a PRNGKey key.
#       scale: The scale parameter of the distribution.
#       concentration: The concentration parameter of the distribution.
#       shape: The shape added to the parameters loc and scale broadcastable shape.
#       dtype: The type used for samples.
#     Returns:
#       A jnp.array of samples.
#     """
#   if not dtypes.issubdtype(dtype, np.floating):
#     raise ValueError(f"dtype argument to `weibull_min` must be a float "
#                      f"dtype, got {dtype}")
#   dtype = dtypes.canonicalize_dtype(dtype)
#   shape = core.canonicalize_shape(shape)
#   return _weibull_min(key, scale, concentration, shape, dtype)


# @partial(jit, static_argnums=(1, 2, 3, 4))
# def _weibull_min(key, scale, concentration, shape, dtype):
#   random_uniform = uniform(
#       key=key, shape=shape, minval=0, maxval=1, dtype=dtype)

#   # Inverse weibull CDF.
#   return jnp.power(-jnp.log1p(-random_uniform), 1.0/concentration) * scale

end