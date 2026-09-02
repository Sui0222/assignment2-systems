### Mixed Precision Accumulation
- for fp32 to fp32, it costs 10.0001
- for fp16 to fp16, it costs 9.9531
- for fp32 to fp16, it costs 10.0021
- for fp16 to fp32 and to sum, it costs 10.0021
- 
> The mathematically exact result is 1000*0.01=10, while FP32 accumulation procudes 10.0001 because 0.01 and the intermediate sums cannot be represented exactly in binary floating point. Direct FP16 accumulation produces 9.9531 because every intermediate result is rounded up to accumulate. Using FP32 accumulator with FP16 inputs produces 10.0021 in both casting cases: FP32 prevents repeated low-precision accumulation errors, but it cannot recover the precision lost when 0.01 was initially quantized to FP16
