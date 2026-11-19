# 直接IR场景的原单元测试命令

在仓库根目录执行。需要编译包含Ascend后端的TileLang；这些场景无需NPU。无驱动环境先将环境变量 `TORCH_DEVICE_BACKEND_AUTOLOAD` 设为 `0`。

原有参数化用例已覆盖38个直接IR场景，本次为这些入口增加输入IR、下沉IR或目标诊断以及验收结果输出。另新增T05-IR复现有效K96未拦截的问题。

## 按测试函数运行

```bash
python -m pytest -q -s testing/ascend/transform/test_ascend_lower_gemm.py::test_invalid_binding_and_slice_are_rejected
python -m pytest -q -s testing/ascend/transform/test_ascend_lower_gemm.py::test_matching_leading_indices_and_compact_regions
python -m pytest -q -s testing/ascend/transform/test_ascend_lower_gemm.py::test_mad_uses_region_geometry_without_shrinking_storage
python -m pytest -q -s testing/ascend/transform/test_ascend_lower_gemm.py::test_blockscaled_effective_k96_with_matching_uint8_scales_is_rejected
python -m pytest -q -s testing/ascend/transform/test_ascend_copy_bounds.py::test_mx_copy_pads_only_the_unwritten_k_tail
python -m pytest -q -s testing/ascend/transform/test_ascend_copy_bounds.py::test_mx_padding_rejects_an_incomplete_physical_scale_group
```

## 每个参数场景单独执行

以下节点名来自实际pytest收集结果，逐行命令只运行一个场景。T05-IR要求拒绝有效K96；当前未拒绝时必须FAIL，不使用xfail。其他38项与独立测试目录中的编号一一对应。

| 对应单元 | 固定参数 | 独立执行命令 |
| --- | --- | --- |
| S14 | `{"case": "wrong-data", "diagnostic": "is bound to"}` | `python -m pytest -q -s 'testing/ascend/transform/test_ascend_lower_gemm.py::test_invalid_binding_and_slice_are_rejected[wrong-data-is bound to]'` |
| S15 | `{"case": "missing", "diagnostic": "no allocation binding"}` | `python -m pytest -q -s 'testing/ascend/transform/test_ascend_lower_gemm.py::test_invalid_binding_and_slice_are_rejected[missing-no allocation binding]'` |
| S16 | `{"case": "scope", "diagnostic": "matching L0A/L0B SF scope"}` | `python -m pytest -q -s 'testing/ascend/transform/test_ascend_lower_gemm.py::test_invalid_binding_and_slice_are_rejected[scope-matching L0A/L0B SF scope]'` |
| S17 | `{"case": "leading", "diagnostic": "same leading index"}` | `python -m pytest -q -s 'testing/ascend/transform/test_ascend_lower_gemm.py::test_invalid_binding_and_slice_are_rejected[leading-same leading index]'` |
| S18 | `{"case": "origin", "diagnostic": "zero-origin"}` | `python -m pytest -q -s 'testing/ascend/transform/test_ascend_lower_gemm.py::test_invalid_binding_and_slice_are_rejected[origin-zero-origin]'` |
| S19 | `{"case": "k", "diagnostic": "compact data K extent"}` | `python -m pytest -q -s 'testing/ascend/transform/test_ascend_lower_gemm.py::test_invalid_binding_and_slice_are_rejected[k-compact data K extent]'` |
| S20 | `{"case": "duplicate", "diagnostic": "multiple SF handles"}` | `python -m pytest -q -s 'testing/ascend/transform/test_ascend_lower_gemm.py::test_invalid_binding_and_slice_are_rejected[duplicate-multiple SF handles]'` |
| S21 | `{"case": "valid"}` | `python -m pytest -q -s 'testing/ascend/transform/test_ascend_lower_gemm.py::test_matching_leading_indices_and_compact_regions[valid]'` |
| S26 | `{"case": "compact"}` | `python -m pytest -q -s 'testing/ascend/transform/test_ascend_lower_gemm.py::test_matching_leading_indices_and_compact_regions[compact]'` |
| T05-IR | `{}` | `python -m pytest -q -s 'testing/ascend/transform/test_ascend_lower_gemm.py::test_blockscaled_effective_k96_with_matching_uint8_scales_is_rejected'` |
| T06 | `{"symbolic": true, "blockscaled": false}` | `python -m pytest -q -s 'testing/ascend/transform/test_ascend_lower_gemm.py::test_mad_uses_region_geometry_without_shrinking_storage[dynamic-region-mad]'` |
| T07 | `{"symbolic": true, "blockscaled": true}` | `python -m pytest -q -s 'testing/ascend/transform/test_ascend_lower_gemm.py::test_mad_uses_region_geometry_without_shrinking_storage[dynamic-region-blockscaled-mad]'` |
| T16 | `{"explicit_task": false, "dtype": "float8_e4m3fn", "layout": "k-contiguous", "actual_k": 65, "tail": [96, 32]}` | `python -m pytest -q -s 'testing/ascend/transform/test_ascend_copy_bounds.py::test_mx_copy_pads_only_the_unwritten_k_tail[implicit-task-float8_e4m3fn-k-contiguous-65-tail0]'` |
| T17 | `{"explicit_task": false, "dtype": "float8_e4m3fn", "layout": "mn-contiguous", "actual_k": 65, "tail": [65, 63]}` | `python -m pytest -q -s 'testing/ascend/transform/test_ascend_copy_bounds.py::test_mx_copy_pads_only_the_unwritten_k_tail[implicit-task-float8_e4m3fn-mn-contiguous-65-tail1]'` |
| T18 | `{"explicit_task": false, "dtype": "float8_e4m3fn", "layout": "transpose-to-k-contiguous", "actual_k": 65, "tail": [96, 32]}` | `python -m pytest -q -s 'testing/ascend/transform/test_ascend_copy_bounds.py::test_mx_copy_pads_only_the_unwritten_k_tail[implicit-task-float8_e4m3fn-transpose-to-k-contiguous-65-tail2]'` |
| T19 | `{"explicit_task": false, "dtype": "float8_e4m3fn", "layout": "k-contiguous", "actual_k": 33, "tail": null}` | `python -m pytest -q -s 'testing/ascend/transform/test_ascend_copy_bounds.py::test_mx_copy_pads_only_the_unwritten_k_tail[implicit-task-float8_e4m3fn-k-contiguous-33-None]'` |
| T20 | `{"explicit_task": false, "dtype": "float8_e4m3fn", "layout": "mn-contiguous", "actual_k": 64, "tail": null}` | `python -m pytest -q -s 'testing/ascend/transform/test_ascend_copy_bounds.py::test_mx_copy_pads_only_the_unwritten_k_tail[implicit-task-float8_e4m3fn-mn-contiguous-64-None]'` |
| T21 | `{"explicit_task": false, "dtype": "float8_e4m3fn", "layout": "transpose-to-k-contiguous", "actual_k": 33, "tail": null}` | `python -m pytest -q -s 'testing/ascend/transform/test_ascend_copy_bounds.py::test_mx_copy_pads_only_the_unwritten_k_tail[implicit-task-float8_e4m3fn-transpose-to-k-contiguous-33-None]'` |
| T22 | `{"explicit_task": false, "dtype": "float4_e2m1fn", "layout": "k-contiguous", "actual_k": 4, "tail": null}` | `python -m pytest -q -s 'testing/ascend/transform/test_ascend_copy_bounds.py::test_mx_copy_pads_only_the_unwritten_k_tail[implicit-task-float4_e2m1fn-k-contiguous-4-None]'` |
| T23 | `{"explicit_task": false, "dtype": "float4_e2m1fn", "layout": "k-contiguous", "actual_k": 34, "tail": null}` | `python -m pytest -q -s 'testing/ascend/transform/test_ascend_copy_bounds.py::test_mx_copy_pads_only_the_unwritten_k_tail[implicit-task-float4_e2m1fn-k-contiguous-34-None]'` |
| T24 | `{"explicit_task": false, "dtype": "float4_e2m1fn", "layout": "mn-contiguous", "actual_k": 4, "tail": [4, 60]}` | `python -m pytest -q -s 'testing/ascend/transform/test_ascend_copy_bounds.py::test_mx_copy_pads_only_the_unwritten_k_tail[implicit-task-float4_e2m1fn-mn-contiguous-4-tail8]'` |
| T25 | `{"explicit_task": false, "dtype": "float4_e2m1fn", "layout": "mn-contiguous", "actual_k": 34, "tail": [34, 30]}` | `python -m pytest -q -s 'testing/ascend/transform/test_ascend_copy_bounds.py::test_mx_copy_pads_only_the_unwritten_k_tail[implicit-task-float4_e2m1fn-mn-contiguous-34-tail9]'` |
| T26 | `{"explicit_task": false, "dtype": "float4_e2m1fn", "layout": "mn-contiguous", "actual_k": 66, "tail": [66, 62]}` | `python -m pytest -q -s 'testing/ascend/transform/test_ascend_copy_bounds.py::test_mx_copy_pads_only_the_unwritten_k_tail[implicit-task-float4_e2m1fn-mn-contiguous-66-tail10]'` |
| T27 | `{"explicit_task": false, "dtype": "float4_e2m1fn", "layout": "mn-contiguous", "actual_k": 64, "tail": null}` | `python -m pytest -q -s 'testing/ascend/transform/test_ascend_copy_bounds.py::test_mx_copy_pads_only_the_unwritten_k_tail[implicit-task-float4_e2m1fn-mn-contiguous-64-None]'` |
| T28 | `{"explicit_task": false, "dtype": "float4_e2m1fn", "layout": "transpose-to-k-contiguous", "actual_k": 66, "tail": null}` | `python -m pytest -q -s 'testing/ascend/transform/test_ascend_copy_bounds.py::test_mx_copy_pads_only_the_unwritten_k_tail[implicit-task-float4_e2m1fn-transpose-to-k-contiguous-66-None]'` |
| T29 | `{"explicit_task": true, "dtype": "float8_e4m3fn", "layout": "k-contiguous", "actual_k": 65, "tail": [96, 32]}` | `python -m pytest -q -s 'testing/ascend/transform/test_ascend_copy_bounds.py::test_mx_copy_pads_only_the_unwritten_k_tail[explicit-task-float8_e4m3fn-k-contiguous-65-tail0]'` |
| T30 | `{"explicit_task": true, "dtype": "float8_e4m3fn", "layout": "mn-contiguous", "actual_k": 65, "tail": [65, 63]}` | `python -m pytest -q -s 'testing/ascend/transform/test_ascend_copy_bounds.py::test_mx_copy_pads_only_the_unwritten_k_tail[explicit-task-float8_e4m3fn-mn-contiguous-65-tail1]'` |
| T31 | `{"explicit_task": true, "dtype": "float8_e4m3fn", "layout": "transpose-to-k-contiguous", "actual_k": 65, "tail": [96, 32]}` | `python -m pytest -q -s 'testing/ascend/transform/test_ascend_copy_bounds.py::test_mx_copy_pads_only_the_unwritten_k_tail[explicit-task-float8_e4m3fn-transpose-to-k-contiguous-65-tail2]'` |
| T32 | `{"explicit_task": true, "dtype": "float8_e4m3fn", "layout": "k-contiguous", "actual_k": 33, "tail": null}` | `python -m pytest -q -s 'testing/ascend/transform/test_ascend_copy_bounds.py::test_mx_copy_pads_only_the_unwritten_k_tail[explicit-task-float8_e4m3fn-k-contiguous-33-None]'` |
| T33 | `{"explicit_task": true, "dtype": "float8_e4m3fn", "layout": "mn-contiguous", "actual_k": 64, "tail": null}` | `python -m pytest -q -s 'testing/ascend/transform/test_ascend_copy_bounds.py::test_mx_copy_pads_only_the_unwritten_k_tail[explicit-task-float8_e4m3fn-mn-contiguous-64-None]'` |
| T34 | `{"explicit_task": true, "dtype": "float8_e4m3fn", "layout": "transpose-to-k-contiguous", "actual_k": 33, "tail": null}` | `python -m pytest -q -s 'testing/ascend/transform/test_ascend_copy_bounds.py::test_mx_copy_pads_only_the_unwritten_k_tail[explicit-task-float8_e4m3fn-transpose-to-k-contiguous-33-None]'` |
| T35 | `{"explicit_task": true, "dtype": "float4_e2m1fn", "layout": "k-contiguous", "actual_k": 4, "tail": null}` | `python -m pytest -q -s 'testing/ascend/transform/test_ascend_copy_bounds.py::test_mx_copy_pads_only_the_unwritten_k_tail[explicit-task-float4_e2m1fn-k-contiguous-4-None]'` |
| T36 | `{"explicit_task": true, "dtype": "float4_e2m1fn", "layout": "k-contiguous", "actual_k": 34, "tail": null}` | `python -m pytest -q -s 'testing/ascend/transform/test_ascend_copy_bounds.py::test_mx_copy_pads_only_the_unwritten_k_tail[explicit-task-float4_e2m1fn-k-contiguous-34-None]'` |
| T37 | `{"explicit_task": true, "dtype": "float4_e2m1fn", "layout": "mn-contiguous", "actual_k": 4, "tail": [4, 60]}` | `python -m pytest -q -s 'testing/ascend/transform/test_ascend_copy_bounds.py::test_mx_copy_pads_only_the_unwritten_k_tail[explicit-task-float4_e2m1fn-mn-contiguous-4-tail8]'` |
| T38 | `{"explicit_task": true, "dtype": "float4_e2m1fn", "layout": "mn-contiguous", "actual_k": 34, "tail": [34, 30]}` | `python -m pytest -q -s 'testing/ascend/transform/test_ascend_copy_bounds.py::test_mx_copy_pads_only_the_unwritten_k_tail[explicit-task-float4_e2m1fn-mn-contiguous-34-tail9]'` |
| T39 | `{"explicit_task": true, "dtype": "float4_e2m1fn", "layout": "mn-contiguous", "actual_k": 66, "tail": [66, 62]}` | `python -m pytest -q -s 'testing/ascend/transform/test_ascend_copy_bounds.py::test_mx_copy_pads_only_the_unwritten_k_tail[explicit-task-float4_e2m1fn-mn-contiguous-66-tail10]'` |
| T40 | `{"explicit_task": true, "dtype": "float4_e2m1fn", "layout": "mn-contiguous", "actual_k": 64, "tail": null}` | `python -m pytest -q -s 'testing/ascend/transform/test_ascend_copy_bounds.py::test_mx_copy_pads_only_the_unwritten_k_tail[explicit-task-float4_e2m1fn-mn-contiguous-64-None]'` |
| T41 | `{"explicit_task": true, "dtype": "float4_e2m1fn", "layout": "transpose-to-k-contiguous", "actual_k": 66, "tail": null}` | `python -m pytest -q -s 'testing/ascend/transform/test_ascend_copy_bounds.py::test_mx_copy_pads_only_the_unwritten_k_tail[explicit-task-float4_e2m1fn-transpose-to-k-contiguous-66-None]'` |
| T42 | `{}` | `python -m pytest -q -s 'testing/ascend/transform/test_ascend_copy_bounds.py::test_mx_padding_rejects_an_incomplete_physical_scale_group'` |
