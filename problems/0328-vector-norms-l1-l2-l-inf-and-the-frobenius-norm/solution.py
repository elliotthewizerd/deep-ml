import torch

def compute_norm(arr: torch.Tensor, norm_type: str) -> float:
    """
    Compute the specified norm of the input tensor.

    'l1', 'l2' and 'linf' are entrywise norms and accept a 1D or 2D tensor.
    'frobenius' is a matrix norm and must raise a ValueError if arr is not 2D.

    Args:
        arr: Input tensor (1D or 2D)
        norm_type: Type of norm ('l1', 'l2', 'linf', or 'frobenius')

    Returns:
        The computed norm as a float
    """
    # Your code here
    if norm_type == "l1":
        v = torch.linalg.vector_norm(arr, ord = 1)
        return v.item()
    elif norm_type == "l2":
        v = torch.linalg.vector_norm(arr, ord = 2)
        return v.item()
    elif norm_type == "linf":
        v = torch.linalg.vector_norm(arr, ord = float('inf'))
        return v.item()
    elif norm_type == "frobenius":
        if arr.ndim != 2:
            raise ValueError("ValueError")
        else : 
            v = torch.linalg.matrix_norm(arr, ord = "fro")
            return v.item()
    pass
