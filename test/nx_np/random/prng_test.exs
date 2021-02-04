defmodule NxNp.Random.PRNGTest do
  use ExUnit.Case

  alias NxNp.Random.PRNG

  defp u32(val) do
    Nx.tensor(val, type: {:u, 32}) 
  end

  describe "threefry2x32_lowering/2" do
    test "works" do
      # In [193]: k0
      # Out[193]: DeviceArray(0, dtype=uint32)

      # In [194]: k1
      # Out[194]: DeviceArray(2884901888, dtype=uint32)

      # In [195]: random._threefry2x32_lowering(k0, k1, jnp.uint32(10), jnp.uint32(0), use_rolled_loops=False)
      # Out[195]: (DeviceArray(1667764360, dtype=uint32), DeviceArray(3696259546, dtype=uint32))
      k0 = u32(0)
      k1 = u32(2884901888)
      x0 = u32(10)
      x1 = u32(0)
      {t0, t1} = PRNG.threefry2x32_lowering({k0, k1}, {x0, x1})
      assert Nx.type(t0) == {:u, 32}
      assert Nx.type(t1) == {:u, 32}

      v0 = Nx.to_scalar(t0)
      v1 = Nx.to_scalar(t1)
      assert v0 == 1667764360
      assert v1 == 3696259546
    end
  end

  describe "new_keypair/1" do
    test "works" do
      key = PRNG.new_keypair(-7055057591812841014)
      expected = Nx.tensor([2652333695, 3792871882], type: {:u, 32})
      assert key == expected
    end
  end
end