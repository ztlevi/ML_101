import torch

float_tensor = torch.tensor([-1.0, 0.0, 1.0, 2.0])
q = torch.quantize_per_tensor(float_tensor, 0.1, 10, torch.quint8)

# Check size of Tensor
q.numel(), q.size()

# Check whether the tensor is quantized
q.is_quantized

# Get the scale of the quantized Tensor, only works for affine quantized tensor
q.q_scale()

# Get the zero_point of quantized Tensor
q.q_zero_point()

# get the underlying integer representation of the quantized Tensor
# int_repr() returns a Tensor of the corresponding data type of the quantized data type
# e.g.for quint8 Tensor it returns a uint8 Tensor while preserving the MemoryFormat when possible
q.int_repr()

# If a quantized Tensor is a scalar we can print the value:
# item() will dequantize the current tensor and return a Scalar of float
q[0].item()

# printing
print(q)
# tensor([0.0026, 0.0059, 0.0063, 0.0084, 0.0009, 0.0020, 0.0094, 0.0019, 0.0079,
#         0.0094], size=(10,), dtype=torch.quint8,
#        quantization_scheme=torch.per_tensor_affine, scale=0.0001, zero_point=2)

# indexing
print(q[0])  # q[0] is a quantized Tensor with one value
# tensor(0.0026, size=(), dtype=torch.quint8,
#        quantization_scheme=torch.per_tensor_affine, scale=0.0001, zero_point=2)
