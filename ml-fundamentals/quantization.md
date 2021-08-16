# Quantization

quantize: [code](https://github.com/ztlevi/ML_101/blob/master/codes/quantization/torch_quantize.py)

* Converts a float tensor to quantized tensor with given scale and zero point.
* Quantization\(x, scale, zero\_point\) = round\(x/scale + zero\_point\)

dequantize: [code](https://github.com/ztlevi/ML_101/blob/master/codes/quantization/torch_dequantize.py)

* Mapping the quantized tensor back to the original scale and pivot

## Reference

* [https://www.h-schmidt.net/FloatConverter/IEEE754.html](https://www.h-schmidt.net/FloatConverter/IEEE754.html)

