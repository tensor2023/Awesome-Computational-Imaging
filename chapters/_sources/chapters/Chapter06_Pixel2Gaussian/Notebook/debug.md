发生异常: AttributeError
module 'gsplat.cuda' has no attribute 'project_gaussians_2d_forward'
  File "/home/xqgao/2025/MIT/code/GS/ContinuousSR-main/models/gaussian.py", line 256, in query_output
    weighted_cholesky, H, W, self.tile_bounds)

            out_img = rasterize_gaussians_sum(xys, depths, radii, conics, num_tiles_hit,
                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xqgao/2025/MIT/code/GS/ContinuousSR-main/models/gaussian.py", line 271, in forward
    image = self.query_output(inp,scale)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xqgao/2025/MIT/code/GS/ContinuousSR-main/demo.py", line 38, in <module>
    pred = model(img.unsqueeze(0), scale).squeeze(0)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: module 'gsplat.cuda' has no attribute 'project_gaussians_2d_forward'

去https://github.com/ChrisDong-THU/gsplat/blob/679b24cb8bae01935399c4685d74a69fbe19b347/gsplat/gsplat/cuda/csrc/bindings.h#L285重新安装这个包

验证安装：python -c "from gsplat import cuda; print(dir(cuda))"
