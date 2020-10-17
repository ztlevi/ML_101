import torch

float_tensor = torch.tensor([-1.0, 0.0, 1.0, 2.0])
q_made_per_tensor = torch.quantize_per_tensor(float_tensor, 0.1, 10, torch.quint8)

# Dequantize
dequantized_tensor = q_made_per_tensor.dequantize()

# Quantized Tensor supports slicing like usual Tensors do
s = q_made_per_tensor[2]  # a quantized Tensor of with same scale and zero_point
# that contains the values of the 2nd row of the original quantized Tensor
# same as q_made_per_tensor[2, :]

# Assignment
q_made_per_tensor[0] = 3.5  # quantize 3.5 and store the int value in quantized Tensor

# Copy
# we can copy from a quantized Tensor of the same size and dtype
# but different scale and zero_point
scale1, zero_point1 = 1e-1, 0
scale2, zero_point2 = 1, -1
q1 = torch._empty_affine_quantized(
    [2, 3], scale=scale1, zero_point=zero_point1, dtype=torch.qint8
)
q2 = torch._empty_affine_quantized(
    [2, 3], scale=scale2, zero_point=zero_point2, dtype=torch.qint8
)
q2.copy_(q1)

# Permutation
q1.transpose(0, 1)  # see https://pytorch.org/docs/stable/torch.html#torch.transpose
q1.permute([1, 0])  # https://pytorch.org/docs/stable/tensors.html#torch.Tensor.permute
q1.contiguous()  # Convert to contiguous Tensor

# Serialization and Deserialization
import tempfile

with tempfile.NamedTemporaryFile() as f:
    torch.save(q2, f)
    f.seek(0)
    q3 = torch.load(f)
