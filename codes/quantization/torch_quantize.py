import torch

# https://github.com/pytorch/pytorch/wiki/Introducing-Quantized-Tensor

# Quantization(x, scale, zero_point) = round(x/scale + zero_point)

# 1. Get a quantized Tensor by quantizing unquantized float Tensors
float_tensor = torch.randn(2, 2, 3)

# Help on built-in function quantize_per_tensor:
# quantize_per_tensor(...)
#     quantize_per_tensor(input, scale, zero_point, dtype) -> Tensor

#     Converts a float tensor to quantized tensor with given scale and zero point.

#     Arguments:
#         input (Tensor): float tensor to quantize
#         scale (float): scale to apply in quantization formula
#         zero_point (int): offset in integer value that maps to float zero
#         dtype (:class:`torch.dtype`): the desired data type of returned tensor.
#             Has to be one of the quantized dtypes: ``torch.quint8``, ``torch.qint8``, ``torch.qint32``

#     Returns:
#         Tensor: A newly quantized tensor

#     Example::

#         >>> torch.quantize_per_tensor(torch.tensor([-1.0, 0.0, 1.0, 2.0]), 0.1, 10, torch.quint8)
#         tensor([-1.,  0.,  1.,  2.], size=(4,), dtype=torch.quint8,
#                quantization_scheme=torch.per_tensor_affine, scale=0.1, zero_point=10)
#         >>> torch.quantize_per_tensor(torch.tensor([-1.0, 0.0, 1.0, 2.0]), 0.1, 10, torch.quint8).int_repr()
#         tensor([ 0, 10, 20, 30], dtype=torch.uint8)

scale, zero_point = 0.01, 2
dtype = torch.qint8
float_tensorq_per_tensor = torch.quantize_per_tensor(float_tensor, scale, zero_point, dtype).int_repr()
print("float_tensor: ", float_tensor)
# >>> float_tensor:  tensor([[[-0.0687,  1.1466, -0.0254],
#          [-0.9382, -2.5390,  0.7466]],
#         [[ 0.2182,  0.8028,  0.2443],
#          [-1.0790, -0.7884, -0.1219]]])
print("float_tensorq_per_tensor: ", float_tensorq_per_tensor)
# >>> float_tensorq_per_tensor:  tensor([[[  -5,  117,   -1],
#          [ -92, -128,   77]],
#         [[  24,   82,   26],
#          [-106,  -77,  -10]]], dtype=torch.int8)
print()

float_tensor = torch.tensor([-1.0, 0.0, 1.0, 2.0])
int_tensor = torch.quantize_per_tensor(float_tensor, 0.1, 10, torch.quint8).int_repr()
print("float_tensor: ", float_tensor)
# >>> float_tensor:  tensor([-1.,  0.,  1.,  2.])
print("int_tensor: ", int_tensor)
# >>> int_tensor:  tensor([ 0, 10, 20, 30], dtype=torch.uint8)
print()

# Help on built-in function quantize_per_channel:
# quantize_per_channel(...)
#     quantize_per_channel(input, scales, zero_points, axis, dtype) -> Tensor

#     Converts a float tensor to per-channel quantized tensor with given scales and zero points.

#     Arguments:
#         input (Tensor): float tensor to quantize
#         scales (Tensor): float 1D tensor of scales to use, size should match ``input.size(axis)``
#         zero_points (int): integer 1D tensor of offset to use, size should match ``input.size(axis)``
#         axis (int): dimension on which apply per-channel quantization
#         dtype (:class:`torch.dtype`): the desired data type of returned tensor.
#             Has to be one of the quantized dtypes: ``torch.quint8``, ``torch.qint8``, ``torch.qint32``

#     Returns:
#         Tensor: A newly quantized tensor

#     Example::

#         >>> x = torch.tensor([[-1.0, 0.0], [1.0, 2.0]])
#         >>> torch.quantize_per_channel(x, torch.tensor([0.1, 0.01]), torch.tensor([10, 0]), 0, torch.quint8)
#         tensor([[-1.,  0.],
#                 [ 1.,  2.]], size=(2, 2), dtype=torch.quint8,
#                quantization_scheme=torch.per_channel_affine,
#                scale=tensor([0.1000, 0.0100], dtype=torch.float64),
#                zero_point=tensor([10,  0]), axis=0)
#         >>> torch.quantize_per_channel(x, torch.tensor([0.1, 0.01]), torch.tensor([10, 0]), 0, torch.quint8).int_
#         tensor([[  0,  10],
#                 [100, 200]], dtype=torch.uint8)

# we also support per channel quantization
float_tensor = torch.randn(2, 2, 3)
scales = torch.tensor([1e-1, 1e-2, 1e-3])
zero_points = torch.tensor([-1, 0, 1])
channel_axis = 2
dtype = torch.qint32
q_per_channel = torch.quantize_per_channel(float_tensor, scales, zero_points, axis=channel_axis, dtype=dtype).int_repr()
print("float_tensor: ", float_tensor)
# >>> float_tensor:  tensor([[[ 0.7369,  1.0839, -0.9761],
#          [-2.2165, -0.1486, -0.6503]],
#         [[-0.5460,  0.1936, -0.7754],
#          [-0.9335, -0.2668, -0.0625]]])
print("q_per_channel: ", q_per_channel)
# >>> q_per_channel:  tensor([[[   6,  108, -975],
#          [ -23,  -15, -649]],
#         [[  -6,   19, -774],
#          [ -10,  -27,  -61]]], dtype=torch.int32)
print()

# 2. Create a quantized Tensor directly from empty_quantized functions
# Note that _empty_affine_quantized is a private API, we will replace it
# something like torch.empty_quantized_tensor(sizes, quantizer) in the future
q = torch._empty_affine_quantized([10], scale=scale, zero_point=zero_point, dtype=dtype).int_repr()
print(q)

# >>> tensor([2053701646,        101,          0,          0,          0,          0,
#           60826480,          1, 1293628728,      32653], dtype=torch.int32)

# 3. Create a quantized Tensor by assembling int Tensors and quantization parameters
# Note that _per_tensor_affine_qtensor is a private API, we will replace it with
# something like torch.form_tensor(int_tensor, quantizer) in the future
int_tensor = torch.randint(0, 100, size=(10,), dtype=torch.uint8)

# The data type will be torch.quint8, which is the corresponding type
# of torch.uint8, we have following correspondance between torch int types and
# torch quantized int types:
# - torch.uint8 -> torch.quint8
# - torch.int8 -> torch.qint8
# - torch.int32 -> torch.qint32
q = torch._make_per_tensor_quantized_tensor(int_tensor, scale, zero_point).int_repr()  # Note no `dtype`

print(q)
# >>> tensor([37, 89, 98, 47, 58, 52, 81, 62, 55, 86], dtype=torch.uint8)
