defmodule NxNp.Util do
  import Nx.Defn


  defn sizeof_mantissa(x) do
    s = transform(Nx.type(x), fn x ->
      x
      |> do_sizeof_mantissa()
      |> Nx.tensor(type: {:u, 64})
    end)
    0 + s
  end

  defn sizeof_val(x) do
    x
    |> Nx.type()
    |> elem(1)
    |> Nx.tensor(type: {:u, 64})
  end

  defp do_sizeof_mantissa({:f, 64}), do: 52
  defp do_sizeof_mantissa({:f, 32}), do: 23
  defp do_sizeof_mantissa({:f, 16}), do: 10
  defp do_sizeof_mantissa({:bf, 16}), do: 10
  defp do_sizeof_mantissa(t), do: raise "Type #{inspect(t)} has no mantissa"

  defn divmod(n, d) do
    {Nx.divide(n, d), Nx.remainder(n, d)}
  end
 
  # defn require_int_type(x) do
  #   transform(x, fn x ->
  #     case Nx.type(x) do
  #       {:s, _} ->
  #         x
  #       {:u, _} ->
  #         x
  #       got ->
  #         raise """
  #         A int type is required - got: #{inspect(got)}
  #         """
  #     end
  #   end)
  # end

  # defn require_float_type(x) do
  #   transform(x, fn x ->
  #     case Nx.type(x) do
  #       {:f, _} ->
  #         x
  #       got ->
  #         raise """
  #         A float type is required - got: #{inspect(got)}
  #         """
  #     end
  #   end)
  # end
end