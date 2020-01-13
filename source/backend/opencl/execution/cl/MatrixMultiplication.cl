__kernel void MatrixMultiplication(__global float *a,
                        __global float *b,
                        __global float *c,
                        __private const int4 imgSize,
                        __private const int4 strides_batch,
                        __private const int4 strides
                        )
{
    int batch = get_global_id(0);
	int2 pos = (int2)(get_global_id(1), get_global_id(2));
    int y = pos.x;
    int x = pos.y;
	
    int2 imgInfo = imgSize.xy;
    int k = imgSize.z;

    int input0Stride = strides_batch.x;
    int input1Stride = strides_batch.y;
    int outputStride = strides_batch.z;

    int aw = strides.x;
    int bw = strides.y;
    int cw = strides.z;

	if(batch >= imgSize.w || y >= imgInfo.x || x >= imgInfo.y)
	{
        return;
    }

    float sum = 0.0f;
    for (int i = 0; i < k; ++i) {
        sum += a[input0Stride*batch + y*aw+i]*b[input1Stride*batch + x+i*bw];
    }
    
    c[outputStride*batch + y * cw+x] = sum;
}
