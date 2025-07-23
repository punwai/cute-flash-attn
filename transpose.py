
# transpose kernel

from typing import Tuple
import cutlass.cute as cute
from cutlass.utils import pipeline

# pipelines
# pipeline.producer_acquire()
# - 
# pipeline.producer_commit()
# pipeline.consumer_wait()
# pipeline.consumer_release()

# A simple transpose kernel written in cute.
class TransposeSimple:
    def __init__(
        self,
        cta_tiler: Tuple[int, int, int],
    ):
        self.cta_tiler = cta_tiler

        consumer_pipe = pipeline.make_pipeline_state(
            type=pipeline.PipelineUserType.Consumer,
            stages=2,
        )
        producer_pipe = pipeline.make_pipeline_state(
            type=pipeline.PipelineUserType.Producer,
            stages=2,
        )

        pipeline.PipelineTmaAsync

        # 1. Make a copy atom from global -> smem
        pipeline.producer_acquire(
            
        )

        # 2. Pad the tensors and 

        # 3. Pad the tensors and 


    @cute.jit
    def __call__(
        self,
        m: cute.Tensor,
        mK: cute.Tensor,
    ):
        pass

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
    ):
        pass
