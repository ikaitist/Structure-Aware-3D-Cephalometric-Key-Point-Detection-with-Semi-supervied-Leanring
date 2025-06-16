import torch
from typing import Union, Sequence, Tuple
import itertools
import operator

# 最大值索引函数
def unravel_index(indices: torch.Tensor, shape: Union[int, Sequence[int]]) -> Tuple[torch.Tensor, ...]:
    """
    Converts a tensor of flat indices into a tuple of coordinate tensors
    that index into an arbitrary tensor of the specified shape.

    Args:
        indices (torch.Tensor): A tensor containing flattened indices.
        shape (int or sequence of ints): The shape of the tensor to index into.

    Returns:
        Tuple[torch.Tensor, ...]: A tuple of tensors representing the multi-dimensional indices.
    """
    # Ensure `shape` is a tuple or torch.Size
    if isinstance(shape, int):
        shape = (shape,)
    elif not isinstance(shape, (Sequence, torch.Size)):
        raise TypeError(f"Expected `shape` to be an int or sequence of ints, but got {type(shape)}")

    shape = torch.Size(shape)  # Convert to torch.Size for consistency

    # Check for valid shape dimensions
    if any(dim <= 0 for dim in shape):
        raise ValueError(f"All dimensions in `shape` must be positive, but got {shape}")

    # Compute coefficients for each dimension
    coefs = list(
        reversed(
            list(
                itertools.accumulate(
                    reversed(shape[1:] + (1,)), func=operator.mul
                )
            )
        )
    )

    # Compute multi-dimensional indices
    indices = indices.unsqueeze(-1)  # Add an extra dimension for broadcasting
    unravelled = indices // torch.tensor(coefs, device=indices.device) % torch.tensor(shape, device=indices.device)

    return unravelled# Split the last dimension into separate tensors



tensor = torch.tensor([[0.5836, 0.2372, 0.2647],
                       [0.2485, 0.8687, 0.8687],
                       [0.5871, 0.8405, 0.9265]])
index =  torch.argmax(tensor)
max_index = unravel_index(index,tensor.shape)


print("最大值:",max_index)  # .item() 将张量转换为 Python 标量


