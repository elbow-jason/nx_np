defmodule NxNp.Random.PRNG do
  import Nx.Defn

  alias NxNp.Util

  @max_u32 Nx.tensor(0xFFFFFFFF, type: {:u, 32})

  def nyi(rest \\ ""), do: raise "not yet implemented #{rest}"

  # @defn_compiler {EXLA, max_signed_type: {:s, 64}}
  # defn new_keypair(seed) do    
  #   seed = Nx.as_type(seed, {:s, 64})
  #   k1 =
  #     seed
  #     |> Nx.right_shift(32)
  #     |> Nx.as_type({:u, 32})
  #     |> Nx.reshape({1})

  #   k2 =
  #     seed
  #     |> Nx.bitwise_and(@max_u32)
  #     |> Nx.as_type({:u, 32})
  #     |> Nx.reshape({1})

  #   # Nx.concatenate([k1, k2])
  #   res = Nx.concatenate([k1, k2])
  #   transform(res, &IO.inspect/1)
  #   res
  # end

  def new_keypair(seed) do
    <<k1::32, k2::32>> = <<seed::64>>
    Nx.tensor([k1, k2], type: {:u, 32})
  end

  def shape_product(s) do
    s
    |> Tuple.to_list()
    |> Enum.sum()
  end

  # def uniform(key, opts \\ []) do
  #   # shape = Keyword.get(opts, :shape, {})
  #   # shape_prod = shape_product(shape)
  # end

  # @doc """
  # Sample uniform random values in [minval, maxval) with given shape/dtype.
  # Args:
  #   key: a PRNGKey used as the random key.
  #   shape: optional, a tuple of nonnegative integers representing the result
  #     shape. Default ().
  #   dtype: optional, a float dtype for the returned values (default float64 if
  #     jax_enable_x64 is true, otherwise float32).
  #   minval: optional, a minimum (inclusive) value broadcast-compatible with shape for the range (default 0).
  #   maxval: optional, a maximum (exclusive) value broadcast-compatible with shape for the range (default 1).
  # Returns:
  #   A random array with the specified shape and dtype.
  # """
  defnp do_uniform(_key, opts \\ []) do
    opts = keyword!(opts, shape: {}, dtype: {:f, 64}, minval: 0, maxval: 0)
    # TODO: figure out how to require a float type
    # TODO: figure out how to check shapes are broadcast-compatible

    shape = opts[:shape]
    _minval =
      opts[:minval]
      |> Nx.tensor(opts[:dtype])
      |> Nx.broadcast(shape)

    _maxval =
      opts[:maxval]
      |> Nx.tensor(opts[:dtype])
      |> Nx.broadcast(shape)
    
    transform(0, fn _ -> nyi("do_uniform/2") end)

  #   finfo = jnp.finfo(dtype)
  #   nbits, nmant = finfo.bits, finfo.nmant

  #   if nbits not in (16, 32, 64):
  #     raise TypeError("uniform only accepts 32- or 64-bit dtypes.")

  #   bits = _random_bits(key, nbits, shape)

  #   # The strategy here is to randomize only the mantissa bits with an exponent of
  #   # 1 (after applying the bias), then shift and scale to the desired range. The
  #   # bit-level transformation we use relies on Numpy and XLA having bit-for-bit
  #   # equivalent float representations, which might not be true on all platforms.
  #   float_bits = lax.bitwise_or(
  #       lax.shift_right_logical(bits, np.array(nbits - nmant, lax.dtype(bits))),
  #       np.array(1., dtype).view(_UINT_DTYPES[nbits]))
  #   floats = lax.bitcast_convert_type(float_bits, dtype) - np.array(1., dtype)
  #   lax.max(
  #       minval,
  #       lax.reshape(floats * (maxval - minval) + minval, shape))
  end

  # defp shape_prod(shape) do
  #   shape
  #   |> Tuple.to_list()
  #   |> Enum.reduce(1, fn dim, acc -> dim * acc end)
  # end

  defn calc_max_count(bit_width, shape_prod) do
    Nx.ceil(bit_width * shape_prod / 32)
  end

  def random_bits({k1, k2}, bit_width, shape_prod) do
    max_count = calc_max_count(bit_width, shape_prod)
    {nblocks, remainder} = Util.divmod(max_count, @max_u32)
    _bits = if Nx.equal(nblocks, 0) and not Nx.equal(remainder, 0) do
      rem_seq = Nx.iota({remainder})
      threefry_2x32({k1, k2}, rem_seq)
    else
      nyi("random_bits with blocks or no remainder")
    end
    # if not nblocks:
    #   bits = threefry_2x32(key, lax.iota(np.uint32, rem))
    # else:
    #   *subkeys, last_key = split(key, nblocks + 1)
    #   blocks = [threefry_2x32(k, lax.iota(np.uint32, jnp.iinfo(np.uint32).max))
    #             for k in subkeys]
    #   last = threefry_2x32(last_key, lax.iota(np.uint32, rem))
    #   bits = lax.concatenate(blocks + [last], 0)

    # dtype = _UINT_DTYPES[bit_width]
    # if bit_width == 64:
    #   bits = [lax.convert_element_type(x, dtype) for x in jnp.split(bits, 2)]
    #   bits = lax.shift_left(bits[0], dtype(32)) | bits[1]
    # elif bit_width in [8, 16]:
    #   # this is essentially bits.view(dtype)[:size]
    #   bits = lax.bitwise_and(
    #     np.uint32(np.iinfo(dtype).max),
    #     lax.shift_right_logical(
    #       lax.broadcast(bits, (1,)),
    #       lax.mul(
    #         np.uint32(bit_width),
    #         lax.broadcasted_iota(np.uint32, (32 // bit_width, 1), 0)
    #       )
    #     )
    #   )
    #   bits = lax.reshape(bits, (np.uint32(max_count * 32 // bit_width),), (1, 0))
    #   bits = lax.convert_element_type(bits, dtype)[:size]
    # return lax.r
  end

  defp split_rank1(ary, indices) do
    max_i = Nx.size(ary) - 1
    indices
    |> indices_to_pairs(0)
    |> Enum.flat_map(fn
      {start, limit} ->
        start = min(start, max_i)
        limit = min(limit, max_i)
        slice_rank1(ary, start, limit, max_i)
      start when is_integer(start) ->
        start = min(start, max_i)
        slice_rank1(ary, start, max_i, max_i)
    end)
  end

  defp slice_rank1(_ary, same, same, _) do
    []
  end
  defp slice_rank1(_ary, start, _, max_i) when start >= max_i do
    []
  end
  defp slice_rank1(ary, start, limit, max_i) when limit <= max_i do
    [Nx.slice(ary, [start], [limit])]
  end

  defp slice_rank1(ary, start, limit, max_i) when limit > max_i do
    [Nx.slice(ary, [start], [max_i])]
  end

  defp indices_to_pairs([], prev) do
    [prev]
  end

  defp indices_to_pairs([head | tail], prev) do
    [{prev, head} | indices_to_pairs(tail, head)]
  end

  def np_split(ary, indices_or_sections, opts \\ []) do
    axis = Keyword.get(opts, :axis, 0)
    
    case {Nx.rank(ary), axis} do
      {1, 0} -> split_rank1(ary, indices_or_sections)
      _ -> nyi("np_split only handles rank 1 tensors on axis 0")
    end
  end
  @doc """
  Apply the Threefry 2x32 hash.
  Args:
    keypair: a pair of 32bit unsigned integers used for the key.
    count: an array of dtype uint32 used for the counts.
  Returns:
    An array of dtype uint32 with the same shape as `count`.
  """
  def threefry_2x32({_k1, _k2}, count) do
    # TODO: require k1, k2, and count to be {:u, 32}
    odd_size =
      count
      |> Nx.size()
      |> Nx.remainder(2)

    _x = if Nx.equal(odd_size, 0) do
      nyi("threefry_2x32 count was even")
    else
      nyi("threefry_2x32 count was odd")
    end


  #   if not lax.dtype(key1) == lax.dtype(key2) == lax.dtype(count) == np.uint32:
  #     msg = "threefry_2x32 requires uint32 arguments, got {}"
  #     raise TypeError(msg.format([lax.dtype(x) for x in [key1, key2, count]]))

  #   odd_size = count.size % 2
  #   if odd_size:
  #     x = list(jnp.split(jnp.concatenate([count.ravel(), np.uint32([0])]), 2))
  #   else:
  #     x = list(jnp.split(count.ravel(), 2))

  #   x = threefry2x32_p.bind(key1, key2, x[0], x[1])
  #   out = jnp.concatenate(x)
  #   assert out.dtype == np.uint32
  #   return lax.reshape(out[:-1] if odd_size else out, count.shape)
  end


  defn ravel(a) do
    Nx.reshape(a, {Nx.size(a)})
  end

  def new_keys(key, num) do
    n = num * 2
    counts = Nx.iota({n})
    counts = Nx.as_type(counts, {:u, 32})
    {k0, k1} = key_to_tuple(key)
    threefry_2x32({k0, k1}, counts)
  end

  defn key_to_tuple(key) do
    k0 = Nx.slice(key, [0], [1])
    k1 = Nx.slice(key, [1], [2])
    {Nx.reshape(k0, {1}), Nx.reshape(k1, {1})}
  end

  def split({k0, k1}, num) do
    {k0, k1}
    |> new_keys(num)
    |> Nx.reshape({num, 2})
  end

  @hash_num Nx.tensor(0x1BD11BDA, type: {:u, 32})

  defnp gen_hash_key({k0, k1}) do
    k0
    |> Nx.bitwise_xor(k1)
    |> Nx.bitwise_xor(@hash_num)
  end

  @doc """
  Apply the Threefry 2x32 hash.
  """
  defn threefry2x32_lowering({k0, k1}, {x0, x1}) do
    x0 = Nx.as_type(x0, {:u, 32})
    x1 = Nx.as_type(x1, {:u, 32})

    k2 = gen_hash_key({k0, k1})

    x0 = x0 + k0
    x1 = x1 + k1

    # round1
    {x0, x1} = apply_rotation0({x0, x1})
          
    x0 = x0 + k1
    x1 = x1 + k2 + 1

    # round2
    {x0, x1} = apply_rotation1({x0, x1})
      
    x0 = x0 + k2
    x1 = x1 + k0 + 2

    # round 3
    {x0, x1} = apply_rotation0({x0, x1})

    x0 = x0 + k0
    x1 = x1 + k1 + 3

    # round 4
    {x0, x1} = apply_rotation1({x0, x1})
    
    x0 = x0 + k1
    x1 = x1 + k2 + 4

    {x0, x1} = apply_rotation0({x0, x1})
    
    x0 = x0 + k2
    x1 = x1 + k0 + 5

    {x0, x1}
  end

  defnp apply_round({v0, v1}, rot) do
    v0 = v0 + v1
    v1 = rotate_left(v1, rot)
    v1 = Nx.bitwise_xor(v0, v1)
    {v0, v1}
  end

  defn rotate_left(x, d) do
    nbits = Util.sizeof_val(d)
    x
    |> Nx.left_shift(d)
    |> Nx.bitwise_or(Nx.right_shift(x, nbits - d))
    |> Nx.as_type({:u, 32})
  end

  defn u32(val) do
    val
    |> Nx.tensor()
    |> Nx.as_type({:u, 32})
  end

  defnp apply_rotation0({x0, x1}) do
    # transform(x0, &IO.inspect/1)
    # transform(x1, &IO.inspect/1)
    {x0, x1}
    |> apply_round(u32(13))
    |> apply_round(u32(15))
    |> apply_round(u32(26))
    |> apply_round(u32(6))
  end

  defnp apply_rotation1({x0, x1}) do
    {x0, x1}
    |> apply_round(u32(17))
    |> apply_round(u32(29))
    |> apply_round(u32(16))
    |> apply_round(u32(24))
  end
end
