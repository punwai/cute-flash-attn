from cutlass.utils.pipeline import PipelineAsync
import cutlass.cute as cute
import cutlass
from cutlass import Int32
from cutlass.utils.pipeline import PipelineUserType, _PipelineOp

class PipelineTmaAsyncNoCluster(PipelineAsync):
    @cute.jit 
    def __init__(self, num_stages: Int32):
        self.num_stages = num_stages
        @cute.struct
        class SharedStorage:
            mainloop_pipeline_array_ptr: cute.struct.MemRange[
                cutlass.Int64, 
                self.num_stages * 2
            ]
        
        PipelineAsync._make_sync_object_array(
            smem_ptr,
            self.num_stages,
        )

    @staticmethod
    def create(
        barrier_storage: cute.Pointer,
        num_stages: Int32,
        producer_group: cute.CooperativeGroup,
        consumer_group: cute.CooperativeGroup,
        tx_count: int,
        init_wait: cutlass.ConstExpr[bool] = True,
    ):
        """
        :param barrier_storage: Pointer to the barrier storage.
        :param num_stages: Number of stages.
        :param producer_group: Producer group.
        :param consumer_group: Consumer group.
        :param tx_count: Transaction count.
        :param init_wait: Whether to initialize the pipeline with a wait.
        :return: PipelineTmaAsyncNoCluster instance.
        """
        producer_type = _PipelineOp.TmaLoad