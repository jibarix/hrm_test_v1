
import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims

import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config
torch._dynamo.config.automatic_dynamic_shapes = False
torch._inductor.config.max_autotune = True
torch._inductor.config.triton.cudagraphs = True
torch._functorch.config.debug_partitioner = True
torch._functorch.config.unlift_effect_tokens = True



isolate_fails_code_str = None



# torch version: 2.5.1+cu124
# torch cuda version: 12.4
# torch git version: a8d6afb511a69687bbb2b7e88a3cf67917e1697e


# CUDA Info: 
# nvcc: NVIDIA (R) Cuda compiler driver 
# Copyright (c) 2005-2024 NVIDIA Corporation 
# Built on Wed_Oct_30_01:18:48_Pacific_Daylight_Time_2024 
# Cuda compilation tools, release 12.6, V12.6.85 
# Build cuda_12.6.r12.6/compiler.35059454_0 

# GPU Hardware Info: 
# NVIDIA GeForce RTX 3070 Ti Laptop GPU : 1 


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    
    
    def forward(self, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53):
        view = torch.ops.aten.view.default(primals_1, [-1, 1, 1])
        where = torch.ops.aten.where.self(view, primals_2, primals_3);  primals_2 = primals_3 = None
        where_1 = torch.ops.aten.where.self(view, primals_4, primals_5);  view = primals_4 = primals_5 = None
        full_default = torch.ops.aten.full.default([], 0, dtype = torch.int32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2 = torch.ops.aten.where.self(primals_1, full_default, primals_6);  full_default = primals_6 = None
        view_2 = torch.ops.aten.view.default(primals_1, [-1, 1])
        where_3 = torch.ops.aten.where.self(view_2, primals_8, primals_7);  primals_8 = primals_7 = None
        where_4 = torch.ops.aten.where.self(view_2, primals_10, primals_9);  view_2 = primals_10 = primals_9 = None
        view_4 = torch.ops.aten.view.default(primals_1, [-1]);  primals_1 = None
        where_5 = torch.ops.aten.where.self(view_4, primals_12, primals_11);  view_4 = primals_12 = primals_11 = None
        convert_element_type = torch.ops.prims.convert_element_type.default(primals_15, torch.bfloat16)
        embedding = torch.ops.aten.embedding.default(convert_element_type, where_3);  convert_element_type = None
        index = torch.ops.aten.index.Tensor(primals_17, [where_5])
        convert_element_type_1 = torch.ops.prims.convert_element_type.default(index, torch.bfloat16);  index = None
        view_5 = torch.ops.aten.view.default(convert_element_type_1, [-1, 1, 128]);  convert_element_type_1 = None
        cat = torch.ops.aten.cat.default([view_5, embedding], -2);  view_5 = embedding = None
        mul = torch.ops.aten.mul.Tensor(cat, 11.313708498984761);  cat = None
        add = torch.ops.aten.add.Tensor(where, mul)
        add_1 = torch.ops.aten.add.Tensor(where_1, add);  where_1 = None
        convert_element_type_2 = torch.ops.prims.convert_element_type.default(primals_19, torch.bfloat16)
        permute = torch.ops.aten.permute.default(convert_element_type_2, [1, 0]);  convert_element_type_2 = None
        view_6 = torch.ops.aten.view.default(add_1, [115328, 128])
        mm = torch.ops.aten.mm.default(view_6, permute);  view_6 = None
        view_7 = torch.ops.aten.view.default(mm, [128, 901, 384]);  mm = None
        view_8 = torch.ops.aten.view.default(view_7, [128, 901, 6, 64]);  view_7 = None
        slice_3 = torch.ops.aten.slice.Tensor(view_8, 2, 0, 2)
        slice_6 = torch.ops.aten.slice.Tensor(view_8, 2, 2, 4)
        slice_9 = torch.ops.aten.slice.Tensor(view_8, 2, 4, 9223372036854775807);  view_8 = None
        convert_element_type_5 = torch.ops.prims.convert_element_type.default(slice_3, torch.float32);  slice_3 = None
        convert_element_type_6 = torch.ops.prims.convert_element_type.default(slice_6, torch.float32);  slice_6 = None
        unsqueeze = torch.ops.aten.unsqueeze.default(primals_13, -2)
        mul_1 = torch.ops.aten.mul.Tensor(convert_element_type_5, unsqueeze)
        slice_10 = torch.ops.aten.slice.Tensor(convert_element_type_5, 3, 0, 32)
        slice_11 = torch.ops.aten.slice.Tensor(convert_element_type_5, 3, 32, 9223372036854775807);  convert_element_type_5 = None
        neg = torch.ops.aten.neg.default(slice_11);  slice_11 = None
        cat_1 = torch.ops.aten.cat.default([neg, slice_10], -1);  neg = slice_10 = None
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(primals_14, -2)
        mul_2 = torch.ops.aten.mul.Tensor(cat_1, unsqueeze_1);  cat_1 = None
        add_2 = torch.ops.aten.add.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
        mul_3 = torch.ops.aten.mul.Tensor(convert_element_type_6, unsqueeze)
        slice_12 = torch.ops.aten.slice.Tensor(convert_element_type_6, 3, 0, 32)
        slice_13 = torch.ops.aten.slice.Tensor(convert_element_type_6, 3, 32, 9223372036854775807);  convert_element_type_6 = None
        neg_1 = torch.ops.aten.neg.default(slice_13);  slice_13 = None
        cat_2 = torch.ops.aten.cat.default([neg_1, slice_12], -1);  neg_1 = slice_12 = None
        mul_4 = torch.ops.aten.mul.Tensor(cat_2, unsqueeze_1);  cat_2 = None
        add_3 = torch.ops.aten.add.Tensor(mul_3, mul_4);  mul_3 = mul_4 = None
        convert_element_type_7 = torch.ops.prims.convert_element_type.default(add_2, torch.bfloat16);  add_2 = None
        convert_element_type_8 = torch.ops.prims.convert_element_type.default(add_3, torch.bfloat16);  add_3 = None
        _flash_attn_forward = torch.ops.flash_attn._flash_attn_forward.default(convert_element_type_7, convert_element_type_8, slice_9, 0.0, 0.125, False, -1, -1, 0.0, None, False);  convert_element_type_7 = convert_element_type_8 = slice_9 = None
        getitem = _flash_attn_forward[0];  _flash_attn_forward = None
        view_9 = torch.ops.aten.view.default(getitem, [128, 901, 128]);  getitem = None
        convert_element_type_9 = torch.ops.prims.convert_element_type.default(primals_20, torch.bfloat16)
        permute_1 = torch.ops.aten.permute.default(convert_element_type_9, [1, 0]);  convert_element_type_9 = None
        view_10 = torch.ops.aten.view.default(view_9, [115328, 128]);  view_9 = None
        mm_1 = torch.ops.aten.mm.default(view_10, permute_1);  view_10 = None
        view_11 = torch.ops.aten.view.default(mm_1, [128, 901, 128]);  mm_1 = None
        add_4 = torch.ops.aten.add.Tensor(add_1, view_11);  add_1 = view_11 = None
        convert_element_type_12 = torch.ops.prims.convert_element_type.default(add_4, torch.float32);  add_4 = None
        pow_1 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_12, 2)
        mean = torch.ops.aten.mean.dim(pow_1, [-1], True);  pow_1 = None
        add_5 = torch.ops.aten.add.Tensor(mean, 1e-05);  mean = None
        rsqrt = torch.ops.aten.rsqrt.default(add_5);  add_5 = None
        mul_5 = torch.ops.aten.mul.Tensor(convert_element_type_12, rsqrt);  convert_element_type_12 = rsqrt = None
        convert_element_type_13 = torch.ops.prims.convert_element_type.default(mul_5, torch.bfloat16);  mul_5 = None
        convert_element_type_14 = torch.ops.prims.convert_element_type.default(primals_21, torch.bfloat16)
        permute_2 = torch.ops.aten.permute.default(convert_element_type_14, [1, 0]);  convert_element_type_14 = None
        view_12 = torch.ops.aten.view.default(convert_element_type_13, [115328, 128])
        mm_2 = torch.ops.aten.mm.default(view_12, permute_2);  view_12 = None
        view_13 = torch.ops.aten.view.default(mm_2, [128, 901, 1024]);  mm_2 = None
        split = torch.ops.aten.split.Tensor(view_13, 512, -1);  view_13 = None
        getitem_4 = split[0]
        getitem_5 = split[1];  split = None
        convert_element_type_17 = torch.ops.prims.convert_element_type.default(getitem_4, torch.float32);  getitem_4 = None
        sigmoid = torch.ops.aten.sigmoid.default(convert_element_type_17)
        mul_6 = torch.ops.aten.mul.Tensor(convert_element_type_17, sigmoid);  convert_element_type_17 = sigmoid = None
        convert_element_type_18 = torch.ops.prims.convert_element_type.default(mul_6, torch.bfloat16);  mul_6 = None
        mul_7 = torch.ops.aten.mul.Tensor(convert_element_type_18, getitem_5);  convert_element_type_18 = getitem_5 = None
        convert_element_type_19 = torch.ops.prims.convert_element_type.default(primals_22, torch.bfloat16)
        permute_3 = torch.ops.aten.permute.default(convert_element_type_19, [1, 0]);  convert_element_type_19 = None
        view_14 = torch.ops.aten.view.default(mul_7, [115328, 512]);  mul_7 = None
        mm_3 = torch.ops.aten.mm.default(view_14, permute_3);  view_14 = None
        view_15 = torch.ops.aten.view.default(mm_3, [128, 901, 128]);  mm_3 = None
        add_6 = torch.ops.aten.add.Tensor(convert_element_type_13, view_15);  convert_element_type_13 = view_15 = None
        convert_element_type_22 = torch.ops.prims.convert_element_type.default(add_6, torch.float32);  add_6 = None
        pow_2 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_22, 2)
        mean_1 = torch.ops.aten.mean.dim(pow_2, [-1], True);  pow_2 = None
        add_7 = torch.ops.aten.add.Tensor(mean_1, 1e-05);  mean_1 = None
        rsqrt_1 = torch.ops.aten.rsqrt.default(add_7);  add_7 = None
        mul_8 = torch.ops.aten.mul.Tensor(convert_element_type_22, rsqrt_1);  convert_element_type_22 = rsqrt_1 = None
        convert_element_type_23 = torch.ops.prims.convert_element_type.default(mul_8, torch.bfloat16);  mul_8 = None
        convert_element_type_24 = torch.ops.prims.convert_element_type.default(primals_23, torch.bfloat16)
        permute_4 = torch.ops.aten.permute.default(convert_element_type_24, [1, 0]);  convert_element_type_24 = None
        view_16 = torch.ops.aten.view.default(convert_element_type_23, [115328, 128])
        mm_4 = torch.ops.aten.mm.default(view_16, permute_4);  view_16 = None
        view_17 = torch.ops.aten.view.default(mm_4, [128, 901, 384]);  mm_4 = None
        view_18 = torch.ops.aten.view.default(view_17, [128, 901, 6, 64]);  view_17 = None
        slice_16 = torch.ops.aten.slice.Tensor(view_18, 2, 0, 2)
        slice_19 = torch.ops.aten.slice.Tensor(view_18, 2, 2, 4)
        slice_22 = torch.ops.aten.slice.Tensor(view_18, 2, 4, 9223372036854775807);  view_18 = None
        convert_element_type_27 = torch.ops.prims.convert_element_type.default(slice_16, torch.float32);  slice_16 = None
        convert_element_type_28 = torch.ops.prims.convert_element_type.default(slice_19, torch.float32);  slice_19 = None
        mul_9 = torch.ops.aten.mul.Tensor(convert_element_type_27, unsqueeze)
        slice_23 = torch.ops.aten.slice.Tensor(convert_element_type_27, 3, 0, 32)
        slice_24 = torch.ops.aten.slice.Tensor(convert_element_type_27, 3, 32, 9223372036854775807);  convert_element_type_27 = None
        neg_2 = torch.ops.aten.neg.default(slice_24);  slice_24 = None
        cat_3 = torch.ops.aten.cat.default([neg_2, slice_23], -1);  neg_2 = slice_23 = None
        mul_10 = torch.ops.aten.mul.Tensor(cat_3, unsqueeze_1);  cat_3 = None
        add_8 = torch.ops.aten.add.Tensor(mul_9, mul_10);  mul_9 = mul_10 = None
        mul_11 = torch.ops.aten.mul.Tensor(convert_element_type_28, unsqueeze)
        slice_25 = torch.ops.aten.slice.Tensor(convert_element_type_28, 3, 0, 32)
        slice_26 = torch.ops.aten.slice.Tensor(convert_element_type_28, 3, 32, 9223372036854775807);  convert_element_type_28 = None
        neg_3 = torch.ops.aten.neg.default(slice_26);  slice_26 = None
        cat_4 = torch.ops.aten.cat.default([neg_3, slice_25], -1);  neg_3 = slice_25 = None
        mul_12 = torch.ops.aten.mul.Tensor(cat_4, unsqueeze_1);  cat_4 = None
        add_9 = torch.ops.aten.add.Tensor(mul_11, mul_12);  mul_11 = mul_12 = None
        convert_element_type_29 = torch.ops.prims.convert_element_type.default(add_8, torch.bfloat16);  add_8 = None
        convert_element_type_30 = torch.ops.prims.convert_element_type.default(add_9, torch.bfloat16);  add_9 = None
        _flash_attn_forward_1 = torch.ops.flash_attn._flash_attn_forward.default(convert_element_type_29, convert_element_type_30, slice_22, 0.0, 0.125, False, -1, -1, 0.0, None, False);  convert_element_type_29 = convert_element_type_30 = slice_22 = None
        getitem_6 = _flash_attn_forward_1[0];  _flash_attn_forward_1 = None
        view_19 = torch.ops.aten.view.default(getitem_6, [128, 901, 128]);  getitem_6 = None
        convert_element_type_31 = torch.ops.prims.convert_element_type.default(primals_24, torch.bfloat16)
        permute_5 = torch.ops.aten.permute.default(convert_element_type_31, [1, 0]);  convert_element_type_31 = None
        view_20 = torch.ops.aten.view.default(view_19, [115328, 128]);  view_19 = None
        mm_5 = torch.ops.aten.mm.default(view_20, permute_5);  view_20 = None
        view_21 = torch.ops.aten.view.default(mm_5, [128, 901, 128]);  mm_5 = None
        add_10 = torch.ops.aten.add.Tensor(convert_element_type_23, view_21);  convert_element_type_23 = view_21 = None
        convert_element_type_34 = torch.ops.prims.convert_element_type.default(add_10, torch.float32);  add_10 = None
        pow_3 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_34, 2)
        mean_2 = torch.ops.aten.mean.dim(pow_3, [-1], True);  pow_3 = None
        add_11 = torch.ops.aten.add.Tensor(mean_2, 1e-05);  mean_2 = None
        rsqrt_2 = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
        mul_13 = torch.ops.aten.mul.Tensor(convert_element_type_34, rsqrt_2);  convert_element_type_34 = rsqrt_2 = None
        convert_element_type_35 = torch.ops.prims.convert_element_type.default(mul_13, torch.bfloat16);  mul_13 = None
        convert_element_type_36 = torch.ops.prims.convert_element_type.default(primals_25, torch.bfloat16)
        permute_6 = torch.ops.aten.permute.default(convert_element_type_36, [1, 0]);  convert_element_type_36 = None
        view_22 = torch.ops.aten.view.default(convert_element_type_35, [115328, 128])
        mm_6 = torch.ops.aten.mm.default(view_22, permute_6);  view_22 = None
        view_23 = torch.ops.aten.view.default(mm_6, [128, 901, 1024]);  mm_6 = None
        split_1 = torch.ops.aten.split.Tensor(view_23, 512, -1);  view_23 = None
        getitem_10 = split_1[0]
        getitem_11 = split_1[1];  split_1 = None
        convert_element_type_39 = torch.ops.prims.convert_element_type.default(getitem_10, torch.float32);  getitem_10 = None
        sigmoid_1 = torch.ops.aten.sigmoid.default(convert_element_type_39)
        mul_14 = torch.ops.aten.mul.Tensor(convert_element_type_39, sigmoid_1);  convert_element_type_39 = sigmoid_1 = None
        convert_element_type_40 = torch.ops.prims.convert_element_type.default(mul_14, torch.bfloat16);  mul_14 = None
        mul_15 = torch.ops.aten.mul.Tensor(convert_element_type_40, getitem_11);  convert_element_type_40 = getitem_11 = None
        convert_element_type_41 = torch.ops.prims.convert_element_type.default(primals_26, torch.bfloat16)
        permute_7 = torch.ops.aten.permute.default(convert_element_type_41, [1, 0]);  convert_element_type_41 = None
        view_24 = torch.ops.aten.view.default(mul_15, [115328, 512]);  mul_15 = None
        mm_7 = torch.ops.aten.mm.default(view_24, permute_7);  view_24 = None
        view_25 = torch.ops.aten.view.default(mm_7, [128, 901, 128]);  mm_7 = None
        add_12 = torch.ops.aten.add.Tensor(convert_element_type_35, view_25);  convert_element_type_35 = view_25 = None
        convert_element_type_44 = torch.ops.prims.convert_element_type.default(add_12, torch.float32);  add_12 = None
        pow_4 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_44, 2)
        mean_3 = torch.ops.aten.mean.dim(pow_4, [-1], True);  pow_4 = None
        add_13 = torch.ops.aten.add.Tensor(mean_3, 1e-05);  mean_3 = None
        rsqrt_3 = torch.ops.aten.rsqrt.default(add_13);  add_13 = None
        mul_16 = torch.ops.aten.mul.Tensor(convert_element_type_44, rsqrt_3);  convert_element_type_44 = rsqrt_3 = None
        convert_element_type_45 = torch.ops.prims.convert_element_type.default(mul_16, torch.bfloat16);  mul_16 = None
        convert_element_type_46 = torch.ops.prims.convert_element_type.default(primals_27, torch.bfloat16)
        permute_8 = torch.ops.aten.permute.default(convert_element_type_46, [1, 0]);  convert_element_type_46 = None
        view_26 = torch.ops.aten.view.default(convert_element_type_45, [115328, 128])
        mm_8 = torch.ops.aten.mm.default(view_26, permute_8);  view_26 = None
        view_27 = torch.ops.aten.view.default(mm_8, [128, 901, 384]);  mm_8 = None
        view_28 = torch.ops.aten.view.default(view_27, [128, 901, 6, 64]);  view_27 = None
        slice_29 = torch.ops.aten.slice.Tensor(view_28, 2, 0, 2)
        slice_32 = torch.ops.aten.slice.Tensor(view_28, 2, 2, 4)
        slice_35 = torch.ops.aten.slice.Tensor(view_28, 2, 4, 9223372036854775807);  view_28 = None
        convert_element_type_49 = torch.ops.prims.convert_element_type.default(slice_29, torch.float32);  slice_29 = None
        convert_element_type_50 = torch.ops.prims.convert_element_type.default(slice_32, torch.float32);  slice_32 = None
        mul_17 = torch.ops.aten.mul.Tensor(convert_element_type_49, unsqueeze)
        slice_36 = torch.ops.aten.slice.Tensor(convert_element_type_49, 3, 0, 32)
        slice_37 = torch.ops.aten.slice.Tensor(convert_element_type_49, 3, 32, 9223372036854775807);  convert_element_type_49 = None
        neg_4 = torch.ops.aten.neg.default(slice_37);  slice_37 = None
        cat_5 = torch.ops.aten.cat.default([neg_4, slice_36], -1);  neg_4 = slice_36 = None
        mul_18 = torch.ops.aten.mul.Tensor(cat_5, unsqueeze_1);  cat_5 = None
        add_14 = torch.ops.aten.add.Tensor(mul_17, mul_18);  mul_17 = mul_18 = None
        mul_19 = torch.ops.aten.mul.Tensor(convert_element_type_50, unsqueeze)
        slice_38 = torch.ops.aten.slice.Tensor(convert_element_type_50, 3, 0, 32)
        slice_39 = torch.ops.aten.slice.Tensor(convert_element_type_50, 3, 32, 9223372036854775807);  convert_element_type_50 = None
        neg_5 = torch.ops.aten.neg.default(slice_39);  slice_39 = None
        cat_6 = torch.ops.aten.cat.default([neg_5, slice_38], -1);  neg_5 = slice_38 = None
        mul_20 = torch.ops.aten.mul.Tensor(cat_6, unsqueeze_1);  cat_6 = None
        add_15 = torch.ops.aten.add.Tensor(mul_19, mul_20);  mul_19 = mul_20 = None
        convert_element_type_51 = torch.ops.prims.convert_element_type.default(add_14, torch.bfloat16);  add_14 = None
        convert_element_type_52 = torch.ops.prims.convert_element_type.default(add_15, torch.bfloat16);  add_15 = None
        _flash_attn_forward_2 = torch.ops.flash_attn._flash_attn_forward.default(convert_element_type_51, convert_element_type_52, slice_35, 0.0, 0.125, False, -1, -1, 0.0, None, False);  convert_element_type_51 = convert_element_type_52 = slice_35 = None
        getitem_12 = _flash_attn_forward_2[0];  _flash_attn_forward_2 = None
        view_29 = torch.ops.aten.view.default(getitem_12, [128, 901, 128]);  getitem_12 = None
        convert_element_type_53 = torch.ops.prims.convert_element_type.default(primals_28, torch.bfloat16)
        permute_9 = torch.ops.aten.permute.default(convert_element_type_53, [1, 0]);  convert_element_type_53 = None
        view_30 = torch.ops.aten.view.default(view_29, [115328, 128]);  view_29 = None
        mm_9 = torch.ops.aten.mm.default(view_30, permute_9);  view_30 = None
        view_31 = torch.ops.aten.view.default(mm_9, [128, 901, 128]);  mm_9 = None
        add_16 = torch.ops.aten.add.Tensor(convert_element_type_45, view_31);  convert_element_type_45 = view_31 = None
        convert_element_type_56 = torch.ops.prims.convert_element_type.default(add_16, torch.float32);  add_16 = None
        pow_5 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_56, 2)
        mean_4 = torch.ops.aten.mean.dim(pow_5, [-1], True);  pow_5 = None
        add_17 = torch.ops.aten.add.Tensor(mean_4, 1e-05);  mean_4 = None
        rsqrt_4 = torch.ops.aten.rsqrt.default(add_17);  add_17 = None
        mul_21 = torch.ops.aten.mul.Tensor(convert_element_type_56, rsqrt_4);  convert_element_type_56 = rsqrt_4 = None
        convert_element_type_57 = torch.ops.prims.convert_element_type.default(mul_21, torch.bfloat16);  mul_21 = None
        convert_element_type_58 = torch.ops.prims.convert_element_type.default(primals_29, torch.bfloat16)
        permute_10 = torch.ops.aten.permute.default(convert_element_type_58, [1, 0]);  convert_element_type_58 = None
        view_32 = torch.ops.aten.view.default(convert_element_type_57, [115328, 128])
        mm_10 = torch.ops.aten.mm.default(view_32, permute_10);  view_32 = None
        view_33 = torch.ops.aten.view.default(mm_10, [128, 901, 1024]);  mm_10 = None
        split_2 = torch.ops.aten.split.Tensor(view_33, 512, -1);  view_33 = None
        getitem_16 = split_2[0]
        getitem_17 = split_2[1];  split_2 = None
        convert_element_type_61 = torch.ops.prims.convert_element_type.default(getitem_16, torch.float32);  getitem_16 = None
        sigmoid_2 = torch.ops.aten.sigmoid.default(convert_element_type_61)
        mul_22 = torch.ops.aten.mul.Tensor(convert_element_type_61, sigmoid_2);  convert_element_type_61 = sigmoid_2 = None
        convert_element_type_62 = torch.ops.prims.convert_element_type.default(mul_22, torch.bfloat16);  mul_22 = None
        mul_23 = torch.ops.aten.mul.Tensor(convert_element_type_62, getitem_17);  convert_element_type_62 = getitem_17 = None
        convert_element_type_63 = torch.ops.prims.convert_element_type.default(primals_30, torch.bfloat16)
        permute_11 = torch.ops.aten.permute.default(convert_element_type_63, [1, 0]);  convert_element_type_63 = None
        view_34 = torch.ops.aten.view.default(mul_23, [115328, 512]);  mul_23 = None
        mm_11 = torch.ops.aten.mm.default(view_34, permute_11);  view_34 = None
        view_35 = torch.ops.aten.view.default(mm_11, [128, 901, 128]);  mm_11 = None
        add_18 = torch.ops.aten.add.Tensor(convert_element_type_57, view_35);  convert_element_type_57 = view_35 = None
        convert_element_type_66 = torch.ops.prims.convert_element_type.default(add_18, torch.float32);  add_18 = None
        pow_6 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_66, 2)
        mean_5 = torch.ops.aten.mean.dim(pow_6, [-1], True);  pow_6 = None
        add_19 = torch.ops.aten.add.Tensor(mean_5, 1e-05);  mean_5 = None
        rsqrt_5 = torch.ops.aten.rsqrt.default(add_19);  add_19 = None
        mul_24 = torch.ops.aten.mul.Tensor(convert_element_type_66, rsqrt_5);  convert_element_type_66 = rsqrt_5 = None
        convert_element_type_67 = torch.ops.prims.convert_element_type.default(mul_24, torch.bfloat16);  mul_24 = None
        convert_element_type_68 = torch.ops.prims.convert_element_type.default(primals_31, torch.bfloat16)
        permute_12 = torch.ops.aten.permute.default(convert_element_type_68, [1, 0]);  convert_element_type_68 = None
        view_36 = torch.ops.aten.view.default(convert_element_type_67, [115328, 128])
        mm_12 = torch.ops.aten.mm.default(view_36, permute_12);  view_36 = None
        view_37 = torch.ops.aten.view.default(mm_12, [128, 901, 384]);  mm_12 = None
        view_38 = torch.ops.aten.view.default(view_37, [128, 901, 6, 64]);  view_37 = None
        slice_42 = torch.ops.aten.slice.Tensor(view_38, 2, 0, 2)
        slice_45 = torch.ops.aten.slice.Tensor(view_38, 2, 2, 4)
        slice_48 = torch.ops.aten.slice.Tensor(view_38, 2, 4, 9223372036854775807);  view_38 = None
        convert_element_type_71 = torch.ops.prims.convert_element_type.default(slice_42, torch.float32);  slice_42 = None
        convert_element_type_72 = torch.ops.prims.convert_element_type.default(slice_45, torch.float32);  slice_45 = None
        mul_25 = torch.ops.aten.mul.Tensor(convert_element_type_71, unsqueeze)
        slice_49 = torch.ops.aten.slice.Tensor(convert_element_type_71, 3, 0, 32)
        slice_50 = torch.ops.aten.slice.Tensor(convert_element_type_71, 3, 32, 9223372036854775807);  convert_element_type_71 = None
        neg_6 = torch.ops.aten.neg.default(slice_50);  slice_50 = None
        cat_7 = torch.ops.aten.cat.default([neg_6, slice_49], -1);  neg_6 = slice_49 = None
        mul_26 = torch.ops.aten.mul.Tensor(cat_7, unsqueeze_1);  cat_7 = None
        add_20 = torch.ops.aten.add.Tensor(mul_25, mul_26);  mul_25 = mul_26 = None
        mul_27 = torch.ops.aten.mul.Tensor(convert_element_type_72, unsqueeze)
        slice_51 = torch.ops.aten.slice.Tensor(convert_element_type_72, 3, 0, 32)
        slice_52 = torch.ops.aten.slice.Tensor(convert_element_type_72, 3, 32, 9223372036854775807);  convert_element_type_72 = None
        neg_7 = torch.ops.aten.neg.default(slice_52);  slice_52 = None
        cat_8 = torch.ops.aten.cat.default([neg_7, slice_51], -1);  neg_7 = slice_51 = None
        mul_28 = torch.ops.aten.mul.Tensor(cat_8, unsqueeze_1);  cat_8 = None
        add_21 = torch.ops.aten.add.Tensor(mul_27, mul_28);  mul_27 = mul_28 = None
        convert_element_type_73 = torch.ops.prims.convert_element_type.default(add_20, torch.bfloat16);  add_20 = None
        convert_element_type_74 = torch.ops.prims.convert_element_type.default(add_21, torch.bfloat16);  add_21 = None
        _flash_attn_forward_3 = torch.ops.flash_attn._flash_attn_forward.default(convert_element_type_73, convert_element_type_74, slice_48, 0.0, 0.125, False, -1, -1, 0.0, None, False);  convert_element_type_73 = convert_element_type_74 = slice_48 = None
        getitem_18 = _flash_attn_forward_3[0];  _flash_attn_forward_3 = None
        view_39 = torch.ops.aten.view.default(getitem_18, [128, 901, 128]);  getitem_18 = None
        convert_element_type_75 = torch.ops.prims.convert_element_type.default(primals_32, torch.bfloat16)
        permute_13 = torch.ops.aten.permute.default(convert_element_type_75, [1, 0]);  convert_element_type_75 = None
        view_40 = torch.ops.aten.view.default(view_39, [115328, 128]);  view_39 = None
        mm_13 = torch.ops.aten.mm.default(view_40, permute_13);  view_40 = None
        view_41 = torch.ops.aten.view.default(mm_13, [128, 901, 128]);  mm_13 = None
        add_22 = torch.ops.aten.add.Tensor(convert_element_type_67, view_41);  convert_element_type_67 = view_41 = None
        convert_element_type_78 = torch.ops.prims.convert_element_type.default(add_22, torch.float32);  add_22 = None
        pow_7 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_78, 2)
        mean_6 = torch.ops.aten.mean.dim(pow_7, [-1], True);  pow_7 = None
        add_23 = torch.ops.aten.add.Tensor(mean_6, 1e-05);  mean_6 = None
        rsqrt_6 = torch.ops.aten.rsqrt.default(add_23);  add_23 = None
        mul_29 = torch.ops.aten.mul.Tensor(convert_element_type_78, rsqrt_6);  convert_element_type_78 = rsqrt_6 = None
        convert_element_type_79 = torch.ops.prims.convert_element_type.default(mul_29, torch.bfloat16);  mul_29 = None
        convert_element_type_80 = torch.ops.prims.convert_element_type.default(primals_33, torch.bfloat16)
        permute_14 = torch.ops.aten.permute.default(convert_element_type_80, [1, 0]);  convert_element_type_80 = None
        view_42 = torch.ops.aten.view.default(convert_element_type_79, [115328, 128])
        mm_14 = torch.ops.aten.mm.default(view_42, permute_14);  view_42 = None
        view_43 = torch.ops.aten.view.default(mm_14, [128, 901, 1024]);  mm_14 = None
        split_3 = torch.ops.aten.split.Tensor(view_43, 512, -1);  view_43 = None
        getitem_22 = split_3[0]
        getitem_23 = split_3[1];  split_3 = None
        convert_element_type_83 = torch.ops.prims.convert_element_type.default(getitem_22, torch.float32);  getitem_22 = None
        sigmoid_3 = torch.ops.aten.sigmoid.default(convert_element_type_83)
        mul_30 = torch.ops.aten.mul.Tensor(convert_element_type_83, sigmoid_3);  convert_element_type_83 = sigmoid_3 = None
        convert_element_type_84 = torch.ops.prims.convert_element_type.default(mul_30, torch.bfloat16);  mul_30 = None
        mul_31 = torch.ops.aten.mul.Tensor(convert_element_type_84, getitem_23);  convert_element_type_84 = getitem_23 = None
        convert_element_type_85 = torch.ops.prims.convert_element_type.default(primals_34, torch.bfloat16)
        permute_15 = torch.ops.aten.permute.default(convert_element_type_85, [1, 0]);  convert_element_type_85 = None
        view_44 = torch.ops.aten.view.default(mul_31, [115328, 512]);  mul_31 = None
        mm_15 = torch.ops.aten.mm.default(view_44, permute_15);  view_44 = None
        view_45 = torch.ops.aten.view.default(mm_15, [128, 901, 128]);  mm_15 = None
        add_24 = torch.ops.aten.add.Tensor(convert_element_type_79, view_45);  convert_element_type_79 = view_45 = None
        convert_element_type_88 = torch.ops.prims.convert_element_type.default(add_24, torch.float32);  add_24 = None
        pow_8 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_88, 2)
        mean_7 = torch.ops.aten.mean.dim(pow_8, [-1], True);  pow_8 = None
        add_25 = torch.ops.aten.add.Tensor(mean_7, 1e-05);  mean_7 = None
        rsqrt_7 = torch.ops.aten.rsqrt.default(add_25);  add_25 = None
        mul_32 = torch.ops.aten.mul.Tensor(convert_element_type_88, rsqrt_7);  convert_element_type_88 = rsqrt_7 = None
        convert_element_type_89 = torch.ops.prims.convert_element_type.default(mul_32, torch.bfloat16);  mul_32 = None
        add_27 = torch.ops.aten.add.Tensor(convert_element_type_89, add);  convert_element_type_89 = add = None
        view_46 = torch.ops.aten.view.default(add_27, [115328, 128])
        mm_16 = torch.ops.aten.mm.default(view_46, permute);  view_46 = None
        view_47 = torch.ops.aten.view.default(mm_16, [128, 901, 384]);  mm_16 = None
        view_48 = torch.ops.aten.view.default(view_47, [128, 901, 6, 64]);  view_47 = None
        slice_55 = torch.ops.aten.slice.Tensor(view_48, 2, 0, 2)
        slice_58 = torch.ops.aten.slice.Tensor(view_48, 2, 2, 4)
        slice_61 = torch.ops.aten.slice.Tensor(view_48, 2, 4, 9223372036854775807);  view_48 = None
        convert_element_type_93 = torch.ops.prims.convert_element_type.default(slice_55, torch.float32);  slice_55 = None
        convert_element_type_94 = torch.ops.prims.convert_element_type.default(slice_58, torch.float32);  slice_58 = None
        mul_33 = torch.ops.aten.mul.Tensor(convert_element_type_93, unsqueeze)
        slice_62 = torch.ops.aten.slice.Tensor(convert_element_type_93, 3, 0, 32)
        slice_63 = torch.ops.aten.slice.Tensor(convert_element_type_93, 3, 32, 9223372036854775807);  convert_element_type_93 = None
        neg_8 = torch.ops.aten.neg.default(slice_63);  slice_63 = None
        cat_9 = torch.ops.aten.cat.default([neg_8, slice_62], -1);  neg_8 = slice_62 = None
        mul_34 = torch.ops.aten.mul.Tensor(cat_9, unsqueeze_1);  cat_9 = None
        add_28 = torch.ops.aten.add.Tensor(mul_33, mul_34);  mul_33 = mul_34 = None
        mul_35 = torch.ops.aten.mul.Tensor(convert_element_type_94, unsqueeze)
        slice_64 = torch.ops.aten.slice.Tensor(convert_element_type_94, 3, 0, 32)
        slice_65 = torch.ops.aten.slice.Tensor(convert_element_type_94, 3, 32, 9223372036854775807);  convert_element_type_94 = None
        neg_9 = torch.ops.aten.neg.default(slice_65);  slice_65 = None
        cat_10 = torch.ops.aten.cat.default([neg_9, slice_64], -1);  neg_9 = slice_64 = None
        mul_36 = torch.ops.aten.mul.Tensor(cat_10, unsqueeze_1);  cat_10 = None
        add_29 = torch.ops.aten.add.Tensor(mul_35, mul_36);  mul_35 = mul_36 = None
        convert_element_type_95 = torch.ops.prims.convert_element_type.default(add_28, torch.bfloat16);  add_28 = None
        convert_element_type_96 = torch.ops.prims.convert_element_type.default(add_29, torch.bfloat16);  add_29 = None
        _flash_attn_forward_4 = torch.ops.flash_attn._flash_attn_forward.default(convert_element_type_95, convert_element_type_96, slice_61, 0.0, 0.125, False, -1, -1, 0.0, None, False);  convert_element_type_95 = convert_element_type_96 = slice_61 = None
        getitem_24 = _flash_attn_forward_4[0];  _flash_attn_forward_4 = None
        view_49 = torch.ops.aten.view.default(getitem_24, [128, 901, 128]);  getitem_24 = None
        view_50 = torch.ops.aten.view.default(view_49, [115328, 128]);  view_49 = None
        mm_17 = torch.ops.aten.mm.default(view_50, permute_1);  view_50 = None
        view_51 = torch.ops.aten.view.default(mm_17, [128, 901, 128]);  mm_17 = None
        add_30 = torch.ops.aten.add.Tensor(add_27, view_51);  add_27 = view_51 = None
        convert_element_type_100 = torch.ops.prims.convert_element_type.default(add_30, torch.float32);  add_30 = None
        pow_9 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_100, 2)
        mean_8 = torch.ops.aten.mean.dim(pow_9, [-1], True);  pow_9 = None
        add_31 = torch.ops.aten.add.Tensor(mean_8, 1e-05);  mean_8 = None
        rsqrt_8 = torch.ops.aten.rsqrt.default(add_31);  add_31 = None
        mul_37 = torch.ops.aten.mul.Tensor(convert_element_type_100, rsqrt_8);  convert_element_type_100 = rsqrt_8 = None
        convert_element_type_101 = torch.ops.prims.convert_element_type.default(mul_37, torch.bfloat16);  mul_37 = None
        view_52 = torch.ops.aten.view.default(convert_element_type_101, [115328, 128])
        mm_18 = torch.ops.aten.mm.default(view_52, permute_2);  view_52 = None
        view_53 = torch.ops.aten.view.default(mm_18, [128, 901, 1024]);  mm_18 = None
        split_4 = torch.ops.aten.split.Tensor(view_53, 512, -1);  view_53 = None
        getitem_28 = split_4[0]
        getitem_29 = split_4[1];  split_4 = None
        convert_element_type_105 = torch.ops.prims.convert_element_type.default(getitem_28, torch.float32);  getitem_28 = None
        sigmoid_4 = torch.ops.aten.sigmoid.default(convert_element_type_105)
        mul_38 = torch.ops.aten.mul.Tensor(convert_element_type_105, sigmoid_4);  convert_element_type_105 = sigmoid_4 = None
        convert_element_type_106 = torch.ops.prims.convert_element_type.default(mul_38, torch.bfloat16);  mul_38 = None
        mul_39 = torch.ops.aten.mul.Tensor(convert_element_type_106, getitem_29);  convert_element_type_106 = getitem_29 = None
        view_54 = torch.ops.aten.view.default(mul_39, [115328, 512]);  mul_39 = None
        mm_19 = torch.ops.aten.mm.default(view_54, permute_3);  view_54 = None
        view_55 = torch.ops.aten.view.default(mm_19, [128, 901, 128]);  mm_19 = None
        add_32 = torch.ops.aten.add.Tensor(convert_element_type_101, view_55);  convert_element_type_101 = view_55 = None
        convert_element_type_110 = torch.ops.prims.convert_element_type.default(add_32, torch.float32);  add_32 = None
        pow_10 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_110, 2)
        mean_9 = torch.ops.aten.mean.dim(pow_10, [-1], True);  pow_10 = None
        add_33 = torch.ops.aten.add.Tensor(mean_9, 1e-05);  mean_9 = None
        rsqrt_9 = torch.ops.aten.rsqrt.default(add_33);  add_33 = None
        mul_40 = torch.ops.aten.mul.Tensor(convert_element_type_110, rsqrt_9);  convert_element_type_110 = rsqrt_9 = None
        convert_element_type_111 = torch.ops.prims.convert_element_type.default(mul_40, torch.bfloat16);  mul_40 = None
        view_56 = torch.ops.aten.view.default(convert_element_type_111, [115328, 128])
        mm_20 = torch.ops.aten.mm.default(view_56, permute_4);  view_56 = None
        view_57 = torch.ops.aten.view.default(mm_20, [128, 901, 384]);  mm_20 = None
        view_58 = torch.ops.aten.view.default(view_57, [128, 901, 6, 64]);  view_57 = None
        slice_68 = torch.ops.aten.slice.Tensor(view_58, 2, 0, 2)
        slice_71 = torch.ops.aten.slice.Tensor(view_58, 2, 2, 4)
        slice_74 = torch.ops.aten.slice.Tensor(view_58, 2, 4, 9223372036854775807);  view_58 = None
        convert_element_type_115 = torch.ops.prims.convert_element_type.default(slice_68, torch.float32);  slice_68 = None
        convert_element_type_116 = torch.ops.prims.convert_element_type.default(slice_71, torch.float32);  slice_71 = None
        mul_41 = torch.ops.aten.mul.Tensor(convert_element_type_115, unsqueeze)
        slice_75 = torch.ops.aten.slice.Tensor(convert_element_type_115, 3, 0, 32)
        slice_76 = torch.ops.aten.slice.Tensor(convert_element_type_115, 3, 32, 9223372036854775807);  convert_element_type_115 = None
        neg_10 = torch.ops.aten.neg.default(slice_76);  slice_76 = None
        cat_11 = torch.ops.aten.cat.default([neg_10, slice_75], -1);  neg_10 = slice_75 = None
        mul_42 = torch.ops.aten.mul.Tensor(cat_11, unsqueeze_1);  cat_11 = None
        add_34 = torch.ops.aten.add.Tensor(mul_41, mul_42);  mul_41 = mul_42 = None
        mul_43 = torch.ops.aten.mul.Tensor(convert_element_type_116, unsqueeze)
        slice_77 = torch.ops.aten.slice.Tensor(convert_element_type_116, 3, 0, 32)
        slice_78 = torch.ops.aten.slice.Tensor(convert_element_type_116, 3, 32, 9223372036854775807);  convert_element_type_116 = None
        neg_11 = torch.ops.aten.neg.default(slice_78);  slice_78 = None
        cat_12 = torch.ops.aten.cat.default([neg_11, slice_77], -1);  neg_11 = slice_77 = None
        mul_44 = torch.ops.aten.mul.Tensor(cat_12, unsqueeze_1);  cat_12 = None
        add_35 = torch.ops.aten.add.Tensor(mul_43, mul_44);  mul_43 = mul_44 = None
        convert_element_type_117 = torch.ops.prims.convert_element_type.default(add_34, torch.bfloat16);  add_34 = None
        convert_element_type_118 = torch.ops.prims.convert_element_type.default(add_35, torch.bfloat16);  add_35 = None
        _flash_attn_forward_5 = torch.ops.flash_attn._flash_attn_forward.default(convert_element_type_117, convert_element_type_118, slice_74, 0.0, 0.125, False, -1, -1, 0.0, None, False);  convert_element_type_117 = convert_element_type_118 = slice_74 = None
        getitem_30 = _flash_attn_forward_5[0];  _flash_attn_forward_5 = None
        view_59 = torch.ops.aten.view.default(getitem_30, [128, 901, 128]);  getitem_30 = None
        view_60 = torch.ops.aten.view.default(view_59, [115328, 128]);  view_59 = None
        mm_21 = torch.ops.aten.mm.default(view_60, permute_5);  view_60 = None
        view_61 = torch.ops.aten.view.default(mm_21, [128, 901, 128]);  mm_21 = None
        add_36 = torch.ops.aten.add.Tensor(convert_element_type_111, view_61);  convert_element_type_111 = view_61 = None
        convert_element_type_122 = torch.ops.prims.convert_element_type.default(add_36, torch.float32);  add_36 = None
        pow_11 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_122, 2)
        mean_10 = torch.ops.aten.mean.dim(pow_11, [-1], True);  pow_11 = None
        add_37 = torch.ops.aten.add.Tensor(mean_10, 1e-05);  mean_10 = None
        rsqrt_10 = torch.ops.aten.rsqrt.default(add_37);  add_37 = None
        mul_45 = torch.ops.aten.mul.Tensor(convert_element_type_122, rsqrt_10);  convert_element_type_122 = rsqrt_10 = None
        convert_element_type_123 = torch.ops.prims.convert_element_type.default(mul_45, torch.bfloat16);  mul_45 = None
        view_62 = torch.ops.aten.view.default(convert_element_type_123, [115328, 128])
        mm_22 = torch.ops.aten.mm.default(view_62, permute_6);  view_62 = None
        view_63 = torch.ops.aten.view.default(mm_22, [128, 901, 1024]);  mm_22 = None
        split_5 = torch.ops.aten.split.Tensor(view_63, 512, -1);  view_63 = None
        getitem_34 = split_5[0]
        getitem_35 = split_5[1];  split_5 = None
        convert_element_type_127 = torch.ops.prims.convert_element_type.default(getitem_34, torch.float32);  getitem_34 = None
        sigmoid_5 = torch.ops.aten.sigmoid.default(convert_element_type_127)
        mul_46 = torch.ops.aten.mul.Tensor(convert_element_type_127, sigmoid_5);  convert_element_type_127 = sigmoid_5 = None
        convert_element_type_128 = torch.ops.prims.convert_element_type.default(mul_46, torch.bfloat16);  mul_46 = None
        mul_47 = torch.ops.aten.mul.Tensor(convert_element_type_128, getitem_35);  convert_element_type_128 = getitem_35 = None
        view_64 = torch.ops.aten.view.default(mul_47, [115328, 512]);  mul_47 = None
        mm_23 = torch.ops.aten.mm.default(view_64, permute_7);  view_64 = None
        view_65 = torch.ops.aten.view.default(mm_23, [128, 901, 128]);  mm_23 = None
        add_38 = torch.ops.aten.add.Tensor(convert_element_type_123, view_65);  convert_element_type_123 = view_65 = None
        convert_element_type_132 = torch.ops.prims.convert_element_type.default(add_38, torch.float32);  add_38 = None
        pow_12 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_132, 2)
        mean_11 = torch.ops.aten.mean.dim(pow_12, [-1], True);  pow_12 = None
        add_39 = torch.ops.aten.add.Tensor(mean_11, 1e-05);  mean_11 = None
        rsqrt_11 = torch.ops.aten.rsqrt.default(add_39);  add_39 = None
        mul_48 = torch.ops.aten.mul.Tensor(convert_element_type_132, rsqrt_11);  convert_element_type_132 = rsqrt_11 = None
        convert_element_type_133 = torch.ops.prims.convert_element_type.default(mul_48, torch.bfloat16);  mul_48 = None
        view_66 = torch.ops.aten.view.default(convert_element_type_133, [115328, 128])
        mm_24 = torch.ops.aten.mm.default(view_66, permute_8);  view_66 = None
        view_67 = torch.ops.aten.view.default(mm_24, [128, 901, 384]);  mm_24 = None
        view_68 = torch.ops.aten.view.default(view_67, [128, 901, 6, 64]);  view_67 = None
        slice_81 = torch.ops.aten.slice.Tensor(view_68, 2, 0, 2)
        slice_84 = torch.ops.aten.slice.Tensor(view_68, 2, 2, 4)
        slice_87 = torch.ops.aten.slice.Tensor(view_68, 2, 4, 9223372036854775807);  view_68 = None
        convert_element_type_137 = torch.ops.prims.convert_element_type.default(slice_81, torch.float32);  slice_81 = None
        convert_element_type_138 = torch.ops.prims.convert_element_type.default(slice_84, torch.float32);  slice_84 = None
        mul_49 = torch.ops.aten.mul.Tensor(convert_element_type_137, unsqueeze)
        slice_88 = torch.ops.aten.slice.Tensor(convert_element_type_137, 3, 0, 32)
        slice_89 = torch.ops.aten.slice.Tensor(convert_element_type_137, 3, 32, 9223372036854775807);  convert_element_type_137 = None
        neg_12 = torch.ops.aten.neg.default(slice_89);  slice_89 = None
        cat_13 = torch.ops.aten.cat.default([neg_12, slice_88], -1);  neg_12 = slice_88 = None
        mul_50 = torch.ops.aten.mul.Tensor(cat_13, unsqueeze_1);  cat_13 = None
        add_40 = torch.ops.aten.add.Tensor(mul_49, mul_50);  mul_49 = mul_50 = None
        mul_51 = torch.ops.aten.mul.Tensor(convert_element_type_138, unsqueeze)
        slice_90 = torch.ops.aten.slice.Tensor(convert_element_type_138, 3, 0, 32)
        slice_91 = torch.ops.aten.slice.Tensor(convert_element_type_138, 3, 32, 9223372036854775807);  convert_element_type_138 = None
        neg_13 = torch.ops.aten.neg.default(slice_91);  slice_91 = None
        cat_14 = torch.ops.aten.cat.default([neg_13, slice_90], -1);  neg_13 = slice_90 = None
        mul_52 = torch.ops.aten.mul.Tensor(cat_14, unsqueeze_1);  cat_14 = None
        add_41 = torch.ops.aten.add.Tensor(mul_51, mul_52);  mul_51 = mul_52 = None
        convert_element_type_139 = torch.ops.prims.convert_element_type.default(add_40, torch.bfloat16);  add_40 = None
        convert_element_type_140 = torch.ops.prims.convert_element_type.default(add_41, torch.bfloat16);  add_41 = None
        _flash_attn_forward_6 = torch.ops.flash_attn._flash_attn_forward.default(convert_element_type_139, convert_element_type_140, slice_87, 0.0, 0.125, False, -1, -1, 0.0, None, False);  convert_element_type_139 = convert_element_type_140 = slice_87 = None
        getitem_36 = _flash_attn_forward_6[0];  _flash_attn_forward_6 = None
        view_69 = torch.ops.aten.view.default(getitem_36, [128, 901, 128]);  getitem_36 = None
        view_70 = torch.ops.aten.view.default(view_69, [115328, 128]);  view_69 = None
        mm_25 = torch.ops.aten.mm.default(view_70, permute_9);  view_70 = None
        view_71 = torch.ops.aten.view.default(mm_25, [128, 901, 128]);  mm_25 = None
        add_42 = torch.ops.aten.add.Tensor(convert_element_type_133, view_71);  convert_element_type_133 = view_71 = None
        convert_element_type_144 = torch.ops.prims.convert_element_type.default(add_42, torch.float32);  add_42 = None
        pow_13 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_144, 2)
        mean_12 = torch.ops.aten.mean.dim(pow_13, [-1], True);  pow_13 = None
        add_43 = torch.ops.aten.add.Tensor(mean_12, 1e-05);  mean_12 = None
        rsqrt_12 = torch.ops.aten.rsqrt.default(add_43);  add_43 = None
        mul_53 = torch.ops.aten.mul.Tensor(convert_element_type_144, rsqrt_12);  convert_element_type_144 = rsqrt_12 = None
        convert_element_type_145 = torch.ops.prims.convert_element_type.default(mul_53, torch.bfloat16);  mul_53 = None
        view_72 = torch.ops.aten.view.default(convert_element_type_145, [115328, 128])
        mm_26 = torch.ops.aten.mm.default(view_72, permute_10);  view_72 = None
        view_73 = torch.ops.aten.view.default(mm_26, [128, 901, 1024]);  mm_26 = None
        split_6 = torch.ops.aten.split.Tensor(view_73, 512, -1);  view_73 = None
        getitem_40 = split_6[0]
        getitem_41 = split_6[1];  split_6 = None
        convert_element_type_149 = torch.ops.prims.convert_element_type.default(getitem_40, torch.float32);  getitem_40 = None
        sigmoid_6 = torch.ops.aten.sigmoid.default(convert_element_type_149)
        mul_54 = torch.ops.aten.mul.Tensor(convert_element_type_149, sigmoid_6);  convert_element_type_149 = sigmoid_6 = None
        convert_element_type_150 = torch.ops.prims.convert_element_type.default(mul_54, torch.bfloat16);  mul_54 = None
        mul_55 = torch.ops.aten.mul.Tensor(convert_element_type_150, getitem_41);  convert_element_type_150 = getitem_41 = None
        view_74 = torch.ops.aten.view.default(mul_55, [115328, 512]);  mul_55 = None
        mm_27 = torch.ops.aten.mm.default(view_74, permute_11);  view_74 = None
        view_75 = torch.ops.aten.view.default(mm_27, [128, 901, 128]);  mm_27 = None
        add_44 = torch.ops.aten.add.Tensor(convert_element_type_145, view_75);  convert_element_type_145 = view_75 = None
        convert_element_type_154 = torch.ops.prims.convert_element_type.default(add_44, torch.float32);  add_44 = None
        pow_14 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_154, 2)
        mean_13 = torch.ops.aten.mean.dim(pow_14, [-1], True);  pow_14 = None
        add_45 = torch.ops.aten.add.Tensor(mean_13, 1e-05);  mean_13 = None
        rsqrt_13 = torch.ops.aten.rsqrt.default(add_45);  add_45 = None
        mul_56 = torch.ops.aten.mul.Tensor(convert_element_type_154, rsqrt_13);  convert_element_type_154 = rsqrt_13 = None
        convert_element_type_155 = torch.ops.prims.convert_element_type.default(mul_56, torch.bfloat16);  mul_56 = None
        view_76 = torch.ops.aten.view.default(convert_element_type_155, [115328, 128])
        mm_28 = torch.ops.aten.mm.default(view_76, permute_12);  view_76 = None
        view_77 = torch.ops.aten.view.default(mm_28, [128, 901, 384]);  mm_28 = None
        view_78 = torch.ops.aten.view.default(view_77, [128, 901, 6, 64]);  view_77 = None
        slice_94 = torch.ops.aten.slice.Tensor(view_78, 2, 0, 2)
        slice_97 = torch.ops.aten.slice.Tensor(view_78, 2, 2, 4)
        slice_100 = torch.ops.aten.slice.Tensor(view_78, 2, 4, 9223372036854775807);  view_78 = None
        convert_element_type_159 = torch.ops.prims.convert_element_type.default(slice_94, torch.float32);  slice_94 = None
        convert_element_type_160 = torch.ops.prims.convert_element_type.default(slice_97, torch.float32);  slice_97 = None
        mul_57 = torch.ops.aten.mul.Tensor(convert_element_type_159, unsqueeze)
        slice_101 = torch.ops.aten.slice.Tensor(convert_element_type_159, 3, 0, 32)
        slice_102 = torch.ops.aten.slice.Tensor(convert_element_type_159, 3, 32, 9223372036854775807);  convert_element_type_159 = None
        neg_14 = torch.ops.aten.neg.default(slice_102);  slice_102 = None
        cat_15 = torch.ops.aten.cat.default([neg_14, slice_101], -1);  neg_14 = slice_101 = None
        mul_58 = torch.ops.aten.mul.Tensor(cat_15, unsqueeze_1);  cat_15 = None
        add_46 = torch.ops.aten.add.Tensor(mul_57, mul_58);  mul_57 = mul_58 = None
        mul_59 = torch.ops.aten.mul.Tensor(convert_element_type_160, unsqueeze)
        slice_103 = torch.ops.aten.slice.Tensor(convert_element_type_160, 3, 0, 32)
        slice_104 = torch.ops.aten.slice.Tensor(convert_element_type_160, 3, 32, 9223372036854775807);  convert_element_type_160 = None
        neg_15 = torch.ops.aten.neg.default(slice_104);  slice_104 = None
        cat_16 = torch.ops.aten.cat.default([neg_15, slice_103], -1);  neg_15 = slice_103 = None
        mul_60 = torch.ops.aten.mul.Tensor(cat_16, unsqueeze_1);  cat_16 = None
        add_47 = torch.ops.aten.add.Tensor(mul_59, mul_60);  mul_59 = mul_60 = None
        convert_element_type_161 = torch.ops.prims.convert_element_type.default(add_46, torch.bfloat16);  add_46 = None
        convert_element_type_162 = torch.ops.prims.convert_element_type.default(add_47, torch.bfloat16);  add_47 = None
        _flash_attn_forward_7 = torch.ops.flash_attn._flash_attn_forward.default(convert_element_type_161, convert_element_type_162, slice_100, 0.0, 0.125, False, -1, -1, 0.0, None, False);  convert_element_type_161 = convert_element_type_162 = slice_100 = None
        getitem_42 = _flash_attn_forward_7[0];  _flash_attn_forward_7 = None
        view_79 = torch.ops.aten.view.default(getitem_42, [128, 901, 128]);  getitem_42 = None
        view_80 = torch.ops.aten.view.default(view_79, [115328, 128]);  view_79 = None
        mm_29 = torch.ops.aten.mm.default(view_80, permute_13);  view_80 = None
        view_81 = torch.ops.aten.view.default(mm_29, [128, 901, 128]);  mm_29 = None
        add_48 = torch.ops.aten.add.Tensor(convert_element_type_155, view_81);  convert_element_type_155 = view_81 = None
        convert_element_type_166 = torch.ops.prims.convert_element_type.default(add_48, torch.float32);  add_48 = None
        pow_15 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_166, 2)
        mean_14 = torch.ops.aten.mean.dim(pow_15, [-1], True);  pow_15 = None
        add_49 = torch.ops.aten.add.Tensor(mean_14, 1e-05);  mean_14 = None
        rsqrt_14 = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
        mul_61 = torch.ops.aten.mul.Tensor(convert_element_type_166, rsqrt_14);  convert_element_type_166 = rsqrt_14 = None
        convert_element_type_167 = torch.ops.prims.convert_element_type.default(mul_61, torch.bfloat16);  mul_61 = None
        view_82 = torch.ops.aten.view.default(convert_element_type_167, [115328, 128])
        mm_30 = torch.ops.aten.mm.default(view_82, permute_14);  view_82 = None
        view_83 = torch.ops.aten.view.default(mm_30, [128, 901, 1024]);  mm_30 = None
        split_7 = torch.ops.aten.split.Tensor(view_83, 512, -1);  view_83 = None
        getitem_46 = split_7[0]
        getitem_47 = split_7[1];  split_7 = None
        convert_element_type_171 = torch.ops.prims.convert_element_type.default(getitem_46, torch.float32);  getitem_46 = None
        sigmoid_7 = torch.ops.aten.sigmoid.default(convert_element_type_171)
        mul_62 = torch.ops.aten.mul.Tensor(convert_element_type_171, sigmoid_7);  convert_element_type_171 = sigmoid_7 = None
        convert_element_type_172 = torch.ops.prims.convert_element_type.default(mul_62, torch.bfloat16);  mul_62 = None
        mul_63 = torch.ops.aten.mul.Tensor(convert_element_type_172, getitem_47);  convert_element_type_172 = getitem_47 = None
        view_84 = torch.ops.aten.view.default(mul_63, [115328, 512]);  mul_63 = None
        mm_31 = torch.ops.aten.mm.default(view_84, permute_15);  view_84 = None
        view_85 = torch.ops.aten.view.default(mm_31, [128, 901, 128]);  mm_31 = None
        add_50 = torch.ops.aten.add.Tensor(convert_element_type_167, view_85);  convert_element_type_167 = view_85 = None
        convert_element_type_176 = torch.ops.prims.convert_element_type.default(add_50, torch.float32);  add_50 = None
        pow_16 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_176, 2)
        mean_15 = torch.ops.aten.mean.dim(pow_16, [-1], True);  pow_16 = None
        add_51 = torch.ops.aten.add.Tensor(mean_15, 1e-05);  mean_15 = None
        rsqrt_15 = torch.ops.aten.rsqrt.default(add_51);  add_51 = None
        mul_64 = torch.ops.aten.mul.Tensor(convert_element_type_176, rsqrt_15);  convert_element_type_176 = rsqrt_15 = None
        convert_element_type_177 = torch.ops.prims.convert_element_type.default(mul_64, torch.bfloat16);  mul_64 = None
        add_52 = torch.ops.aten.add.Tensor(where, convert_element_type_177);  where = None
        convert_element_type_178 = torch.ops.prims.convert_element_type.default(primals_35, torch.bfloat16)
        permute_32 = torch.ops.aten.permute.default(convert_element_type_178, [1, 0]);  convert_element_type_178 = None
        view_86 = torch.ops.aten.view.default(add_52, [115328, 128])
        mm_32 = torch.ops.aten.mm.default(view_86, permute_32);  view_86 = None
        view_87 = torch.ops.aten.view.default(mm_32, [128, 901, 384]);  mm_32 = None
        view_88 = torch.ops.aten.view.default(view_87, [128, 901, 6, 64]);  view_87 = None
        slice_107 = torch.ops.aten.slice.Tensor(view_88, 2, 0, 2)
        slice_110 = torch.ops.aten.slice.Tensor(view_88, 2, 2, 4)
        slice_113 = torch.ops.aten.slice.Tensor(view_88, 2, 4, 9223372036854775807);  view_88 = None
        convert_element_type_181 = torch.ops.prims.convert_element_type.default(slice_107, torch.float32);  slice_107 = None
        convert_element_type_182 = torch.ops.prims.convert_element_type.default(slice_110, torch.float32);  slice_110 = None
        mul_65 = torch.ops.aten.mul.Tensor(convert_element_type_181, unsqueeze)
        slice_114 = torch.ops.aten.slice.Tensor(convert_element_type_181, 3, 0, 32)
        slice_115 = torch.ops.aten.slice.Tensor(convert_element_type_181, 3, 32, 9223372036854775807);  convert_element_type_181 = None
        neg_16 = torch.ops.aten.neg.default(slice_115);  slice_115 = None
        cat_17 = torch.ops.aten.cat.default([neg_16, slice_114], -1);  neg_16 = slice_114 = None
        mul_66 = torch.ops.aten.mul.Tensor(cat_17, unsqueeze_1);  cat_17 = None
        add_53 = torch.ops.aten.add.Tensor(mul_65, mul_66);  mul_65 = mul_66 = None
        mul_67 = torch.ops.aten.mul.Tensor(convert_element_type_182, unsqueeze)
        slice_116 = torch.ops.aten.slice.Tensor(convert_element_type_182, 3, 0, 32)
        slice_117 = torch.ops.aten.slice.Tensor(convert_element_type_182, 3, 32, 9223372036854775807);  convert_element_type_182 = None
        neg_17 = torch.ops.aten.neg.default(slice_117);  slice_117 = None
        cat_18 = torch.ops.aten.cat.default([neg_17, slice_116], -1);  neg_17 = slice_116 = None
        mul_68 = torch.ops.aten.mul.Tensor(cat_18, unsqueeze_1);  cat_18 = None
        add_54 = torch.ops.aten.add.Tensor(mul_67, mul_68);  mul_67 = mul_68 = None
        convert_element_type_183 = torch.ops.prims.convert_element_type.default(add_53, torch.bfloat16);  add_53 = None
        convert_element_type_184 = torch.ops.prims.convert_element_type.default(add_54, torch.bfloat16);  add_54 = None
        _flash_attn_forward_8 = torch.ops.flash_attn._flash_attn_forward.default(convert_element_type_183, convert_element_type_184, slice_113, 0.0, 0.125, False, -1, -1, 0.0, None, False);  convert_element_type_183 = convert_element_type_184 = slice_113 = None
        getitem_48 = _flash_attn_forward_8[0];  _flash_attn_forward_8 = None
        view_89 = torch.ops.aten.view.default(getitem_48, [128, 901, 128]);  getitem_48 = None
        convert_element_type_185 = torch.ops.prims.convert_element_type.default(primals_36, torch.bfloat16)
        permute_33 = torch.ops.aten.permute.default(convert_element_type_185, [1, 0]);  convert_element_type_185 = None
        view_90 = torch.ops.aten.view.default(view_89, [115328, 128]);  view_89 = None
        mm_33 = torch.ops.aten.mm.default(view_90, permute_33);  view_90 = None
        view_91 = torch.ops.aten.view.default(mm_33, [128, 901, 128]);  mm_33 = None
        add_55 = torch.ops.aten.add.Tensor(add_52, view_91);  add_52 = view_91 = None
        convert_element_type_188 = torch.ops.prims.convert_element_type.default(add_55, torch.float32);  add_55 = None
        pow_17 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_188, 2)
        mean_16 = torch.ops.aten.mean.dim(pow_17, [-1], True);  pow_17 = None
        add_56 = torch.ops.aten.add.Tensor(mean_16, 1e-05);  mean_16 = None
        rsqrt_16 = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
        mul_69 = torch.ops.aten.mul.Tensor(convert_element_type_188, rsqrt_16);  convert_element_type_188 = rsqrt_16 = None
        convert_element_type_189 = torch.ops.prims.convert_element_type.default(mul_69, torch.bfloat16);  mul_69 = None
        convert_element_type_190 = torch.ops.prims.convert_element_type.default(primals_37, torch.bfloat16)
        permute_34 = torch.ops.aten.permute.default(convert_element_type_190, [1, 0]);  convert_element_type_190 = None
        view_92 = torch.ops.aten.view.default(convert_element_type_189, [115328, 128])
        mm_34 = torch.ops.aten.mm.default(view_92, permute_34);  view_92 = None
        view_93 = torch.ops.aten.view.default(mm_34, [128, 901, 1024]);  mm_34 = None
        split_8 = torch.ops.aten.split.Tensor(view_93, 512, -1);  view_93 = None
        getitem_52 = split_8[0]
        getitem_53 = split_8[1];  split_8 = None
        convert_element_type_193 = torch.ops.prims.convert_element_type.default(getitem_52, torch.float32);  getitem_52 = None
        sigmoid_8 = torch.ops.aten.sigmoid.default(convert_element_type_193)
        mul_70 = torch.ops.aten.mul.Tensor(convert_element_type_193, sigmoid_8);  convert_element_type_193 = sigmoid_8 = None
        convert_element_type_194 = torch.ops.prims.convert_element_type.default(mul_70, torch.bfloat16);  mul_70 = None
        mul_71 = torch.ops.aten.mul.Tensor(convert_element_type_194, getitem_53);  convert_element_type_194 = getitem_53 = None
        convert_element_type_195 = torch.ops.prims.convert_element_type.default(primals_38, torch.bfloat16)
        permute_35 = torch.ops.aten.permute.default(convert_element_type_195, [1, 0]);  convert_element_type_195 = None
        view_94 = torch.ops.aten.view.default(mul_71, [115328, 512]);  mul_71 = None
        mm_35 = torch.ops.aten.mm.default(view_94, permute_35);  view_94 = None
        view_95 = torch.ops.aten.view.default(mm_35, [128, 901, 128]);  mm_35 = None
        add_57 = torch.ops.aten.add.Tensor(convert_element_type_189, view_95);  convert_element_type_189 = view_95 = None
        convert_element_type_198 = torch.ops.prims.convert_element_type.default(add_57, torch.float32);  add_57 = None
        pow_18 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_198, 2)
        mean_17 = torch.ops.aten.mean.dim(pow_18, [-1], True);  pow_18 = None
        add_58 = torch.ops.aten.add.Tensor(mean_17, 1e-05);  mean_17 = None
        rsqrt_17 = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
        mul_72 = torch.ops.aten.mul.Tensor(convert_element_type_198, rsqrt_17);  convert_element_type_198 = rsqrt_17 = None
        convert_element_type_199 = torch.ops.prims.convert_element_type.default(mul_72, torch.bfloat16);  mul_72 = None
        convert_element_type_200 = torch.ops.prims.convert_element_type.default(primals_39, torch.bfloat16)
        permute_36 = torch.ops.aten.permute.default(convert_element_type_200, [1, 0]);  convert_element_type_200 = None
        view_96 = torch.ops.aten.view.default(convert_element_type_199, [115328, 128])
        mm_36 = torch.ops.aten.mm.default(view_96, permute_36);  view_96 = None
        view_97 = torch.ops.aten.view.default(mm_36, [128, 901, 384]);  mm_36 = None
        view_98 = torch.ops.aten.view.default(view_97, [128, 901, 6, 64]);  view_97 = None
        slice_120 = torch.ops.aten.slice.Tensor(view_98, 2, 0, 2)
        slice_123 = torch.ops.aten.slice.Tensor(view_98, 2, 2, 4)
        slice_126 = torch.ops.aten.slice.Tensor(view_98, 2, 4, 9223372036854775807);  view_98 = None
        convert_element_type_203 = torch.ops.prims.convert_element_type.default(slice_120, torch.float32);  slice_120 = None
        convert_element_type_204 = torch.ops.prims.convert_element_type.default(slice_123, torch.float32);  slice_123 = None
        mul_73 = torch.ops.aten.mul.Tensor(convert_element_type_203, unsqueeze)
        slice_127 = torch.ops.aten.slice.Tensor(convert_element_type_203, 3, 0, 32)
        slice_128 = torch.ops.aten.slice.Tensor(convert_element_type_203, 3, 32, 9223372036854775807);  convert_element_type_203 = None
        neg_18 = torch.ops.aten.neg.default(slice_128);  slice_128 = None
        cat_19 = torch.ops.aten.cat.default([neg_18, slice_127], -1);  neg_18 = slice_127 = None
        mul_74 = torch.ops.aten.mul.Tensor(cat_19, unsqueeze_1);  cat_19 = None
        add_59 = torch.ops.aten.add.Tensor(mul_73, mul_74);  mul_73 = mul_74 = None
        mul_75 = torch.ops.aten.mul.Tensor(convert_element_type_204, unsqueeze)
        slice_129 = torch.ops.aten.slice.Tensor(convert_element_type_204, 3, 0, 32)
        slice_130 = torch.ops.aten.slice.Tensor(convert_element_type_204, 3, 32, 9223372036854775807);  convert_element_type_204 = None
        neg_19 = torch.ops.aten.neg.default(slice_130);  slice_130 = None
        cat_20 = torch.ops.aten.cat.default([neg_19, slice_129], -1);  neg_19 = slice_129 = None
        mul_76 = torch.ops.aten.mul.Tensor(cat_20, unsqueeze_1);  cat_20 = None
        add_60 = torch.ops.aten.add.Tensor(mul_75, mul_76);  mul_75 = mul_76 = None
        convert_element_type_205 = torch.ops.prims.convert_element_type.default(add_59, torch.bfloat16);  add_59 = None
        convert_element_type_206 = torch.ops.prims.convert_element_type.default(add_60, torch.bfloat16);  add_60 = None
        _flash_attn_forward_9 = torch.ops.flash_attn._flash_attn_forward.default(convert_element_type_205, convert_element_type_206, slice_126, 0.0, 0.125, False, -1, -1, 0.0, None, False);  convert_element_type_205 = convert_element_type_206 = slice_126 = None
        getitem_54 = _flash_attn_forward_9[0];  _flash_attn_forward_9 = None
        view_99 = torch.ops.aten.view.default(getitem_54, [128, 901, 128]);  getitem_54 = None
        convert_element_type_207 = torch.ops.prims.convert_element_type.default(primals_40, torch.bfloat16)
        permute_37 = torch.ops.aten.permute.default(convert_element_type_207, [1, 0]);  convert_element_type_207 = None
        view_100 = torch.ops.aten.view.default(view_99, [115328, 128]);  view_99 = None
        mm_37 = torch.ops.aten.mm.default(view_100, permute_37);  view_100 = None
        view_101 = torch.ops.aten.view.default(mm_37, [128, 901, 128]);  mm_37 = None
        add_61 = torch.ops.aten.add.Tensor(convert_element_type_199, view_101);  convert_element_type_199 = view_101 = None
        convert_element_type_210 = torch.ops.prims.convert_element_type.default(add_61, torch.float32);  add_61 = None
        pow_19 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_210, 2)
        mean_18 = torch.ops.aten.mean.dim(pow_19, [-1], True);  pow_19 = None
        add_62 = torch.ops.aten.add.Tensor(mean_18, 1e-05);  mean_18 = None
        rsqrt_18 = torch.ops.aten.rsqrt.default(add_62);  add_62 = None
        mul_77 = torch.ops.aten.mul.Tensor(convert_element_type_210, rsqrt_18);  convert_element_type_210 = rsqrt_18 = None
        convert_element_type_211 = torch.ops.prims.convert_element_type.default(mul_77, torch.bfloat16);  mul_77 = None
        convert_element_type_212 = torch.ops.prims.convert_element_type.default(primals_41, torch.bfloat16)
        permute_38 = torch.ops.aten.permute.default(convert_element_type_212, [1, 0]);  convert_element_type_212 = None
        view_102 = torch.ops.aten.view.default(convert_element_type_211, [115328, 128])
        mm_38 = torch.ops.aten.mm.default(view_102, permute_38);  view_102 = None
        view_103 = torch.ops.aten.view.default(mm_38, [128, 901, 1024]);  mm_38 = None
        split_9 = torch.ops.aten.split.Tensor(view_103, 512, -1);  view_103 = None
        getitem_58 = split_9[0]
        getitem_59 = split_9[1];  split_9 = None
        convert_element_type_215 = torch.ops.prims.convert_element_type.default(getitem_58, torch.float32);  getitem_58 = None
        sigmoid_9 = torch.ops.aten.sigmoid.default(convert_element_type_215)
        mul_78 = torch.ops.aten.mul.Tensor(convert_element_type_215, sigmoid_9);  convert_element_type_215 = sigmoid_9 = None
        convert_element_type_216 = torch.ops.prims.convert_element_type.default(mul_78, torch.bfloat16);  mul_78 = None
        mul_79 = torch.ops.aten.mul.Tensor(convert_element_type_216, getitem_59);  convert_element_type_216 = getitem_59 = None
        convert_element_type_217 = torch.ops.prims.convert_element_type.default(primals_42, torch.bfloat16)
        permute_39 = torch.ops.aten.permute.default(convert_element_type_217, [1, 0]);  convert_element_type_217 = None
        view_104 = torch.ops.aten.view.default(mul_79, [115328, 512]);  mul_79 = None
        mm_39 = torch.ops.aten.mm.default(view_104, permute_39);  view_104 = None
        view_105 = torch.ops.aten.view.default(mm_39, [128, 901, 128]);  mm_39 = None
        add_63 = torch.ops.aten.add.Tensor(convert_element_type_211, view_105);  convert_element_type_211 = view_105 = None
        convert_element_type_220 = torch.ops.prims.convert_element_type.default(add_63, torch.float32);  add_63 = None
        pow_20 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_220, 2)
        mean_19 = torch.ops.aten.mean.dim(pow_20, [-1], True);  pow_20 = None
        add_64 = torch.ops.aten.add.Tensor(mean_19, 1e-05);  mean_19 = None
        rsqrt_19 = torch.ops.aten.rsqrt.default(add_64);  add_64 = None
        mul_80 = torch.ops.aten.mul.Tensor(convert_element_type_220, rsqrt_19);  convert_element_type_220 = rsqrt_19 = None
        convert_element_type_221 = torch.ops.prims.convert_element_type.default(mul_80, torch.bfloat16);  mul_80 = None
        convert_element_type_222 = torch.ops.prims.convert_element_type.default(primals_43, torch.bfloat16)
        permute_40 = torch.ops.aten.permute.default(convert_element_type_222, [1, 0]);  convert_element_type_222 = None
        view_106 = torch.ops.aten.view.default(convert_element_type_221, [115328, 128])
        mm_40 = torch.ops.aten.mm.default(view_106, permute_40);  view_106 = None
        view_107 = torch.ops.aten.view.default(mm_40, [128, 901, 384]);  mm_40 = None
        view_108 = torch.ops.aten.view.default(view_107, [128, 901, 6, 64]);  view_107 = None
        slice_133 = torch.ops.aten.slice.Tensor(view_108, 2, 0, 2)
        slice_136 = torch.ops.aten.slice.Tensor(view_108, 2, 2, 4)
        slice_139 = torch.ops.aten.slice.Tensor(view_108, 2, 4, 9223372036854775807);  view_108 = None
        convert_element_type_225 = torch.ops.prims.convert_element_type.default(slice_133, torch.float32);  slice_133 = None
        convert_element_type_226 = torch.ops.prims.convert_element_type.default(slice_136, torch.float32);  slice_136 = None
        mul_81 = torch.ops.aten.mul.Tensor(convert_element_type_225, unsqueeze)
        slice_140 = torch.ops.aten.slice.Tensor(convert_element_type_225, 3, 0, 32)
        slice_141 = torch.ops.aten.slice.Tensor(convert_element_type_225, 3, 32, 9223372036854775807);  convert_element_type_225 = None
        neg_20 = torch.ops.aten.neg.default(slice_141);  slice_141 = None
        cat_21 = torch.ops.aten.cat.default([neg_20, slice_140], -1);  neg_20 = slice_140 = None
        mul_82 = torch.ops.aten.mul.Tensor(cat_21, unsqueeze_1);  cat_21 = None
        add_65 = torch.ops.aten.add.Tensor(mul_81, mul_82);  mul_81 = mul_82 = None
        mul_83 = torch.ops.aten.mul.Tensor(convert_element_type_226, unsqueeze)
        slice_142 = torch.ops.aten.slice.Tensor(convert_element_type_226, 3, 0, 32)
        slice_143 = torch.ops.aten.slice.Tensor(convert_element_type_226, 3, 32, 9223372036854775807);  convert_element_type_226 = None
        neg_21 = torch.ops.aten.neg.default(slice_143);  slice_143 = None
        cat_22 = torch.ops.aten.cat.default([neg_21, slice_142], -1);  neg_21 = slice_142 = None
        mul_84 = torch.ops.aten.mul.Tensor(cat_22, unsqueeze_1);  cat_22 = None
        add_66 = torch.ops.aten.add.Tensor(mul_83, mul_84);  mul_83 = mul_84 = None
        convert_element_type_227 = torch.ops.prims.convert_element_type.default(add_65, torch.bfloat16);  add_65 = None
        convert_element_type_228 = torch.ops.prims.convert_element_type.default(add_66, torch.bfloat16);  add_66 = None
        _flash_attn_forward_10 = torch.ops.flash_attn._flash_attn_forward.default(convert_element_type_227, convert_element_type_228, slice_139, 0.0, 0.125, False, -1, -1, 0.0, None, False);  convert_element_type_227 = convert_element_type_228 = slice_139 = None
        getitem_60 = _flash_attn_forward_10[0];  _flash_attn_forward_10 = None
        view_109 = torch.ops.aten.view.default(getitem_60, [128, 901, 128]);  getitem_60 = None
        convert_element_type_229 = torch.ops.prims.convert_element_type.default(primals_44, torch.bfloat16)
        permute_41 = torch.ops.aten.permute.default(convert_element_type_229, [1, 0]);  convert_element_type_229 = None
        view_110 = torch.ops.aten.view.default(view_109, [115328, 128]);  view_109 = None
        mm_41 = torch.ops.aten.mm.default(view_110, permute_41);  view_110 = None
        view_111 = torch.ops.aten.view.default(mm_41, [128, 901, 128]);  mm_41 = None
        add_67 = torch.ops.aten.add.Tensor(convert_element_type_221, view_111);  convert_element_type_221 = view_111 = None
        convert_element_type_232 = torch.ops.prims.convert_element_type.default(add_67, torch.float32);  add_67 = None
        pow_21 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_232, 2)
        mean_20 = torch.ops.aten.mean.dim(pow_21, [-1], True);  pow_21 = None
        add_68 = torch.ops.aten.add.Tensor(mean_20, 1e-05);  mean_20 = None
        rsqrt_20 = torch.ops.aten.rsqrt.default(add_68);  add_68 = None
        mul_85 = torch.ops.aten.mul.Tensor(convert_element_type_232, rsqrt_20);  convert_element_type_232 = rsqrt_20 = None
        convert_element_type_233 = torch.ops.prims.convert_element_type.default(mul_85, torch.bfloat16);  mul_85 = None
        convert_element_type_234 = torch.ops.prims.convert_element_type.default(primals_45, torch.bfloat16)
        permute_42 = torch.ops.aten.permute.default(convert_element_type_234, [1, 0]);  convert_element_type_234 = None
        view_112 = torch.ops.aten.view.default(convert_element_type_233, [115328, 128])
        mm_42 = torch.ops.aten.mm.default(view_112, permute_42);  view_112 = None
        view_113 = torch.ops.aten.view.default(mm_42, [128, 901, 1024]);  mm_42 = None
        split_10 = torch.ops.aten.split.Tensor(view_113, 512, -1);  view_113 = None
        getitem_64 = split_10[0]
        getitem_65 = split_10[1];  split_10 = None
        convert_element_type_237 = torch.ops.prims.convert_element_type.default(getitem_64, torch.float32);  getitem_64 = None
        sigmoid_10 = torch.ops.aten.sigmoid.default(convert_element_type_237)
        mul_86 = torch.ops.aten.mul.Tensor(convert_element_type_237, sigmoid_10);  convert_element_type_237 = sigmoid_10 = None
        convert_element_type_238 = torch.ops.prims.convert_element_type.default(mul_86, torch.bfloat16);  mul_86 = None
        mul_87 = torch.ops.aten.mul.Tensor(convert_element_type_238, getitem_65);  convert_element_type_238 = getitem_65 = None
        convert_element_type_239 = torch.ops.prims.convert_element_type.default(primals_46, torch.bfloat16)
        permute_43 = torch.ops.aten.permute.default(convert_element_type_239, [1, 0]);  convert_element_type_239 = None
        view_114 = torch.ops.aten.view.default(mul_87, [115328, 512]);  mul_87 = None
        mm_43 = torch.ops.aten.mm.default(view_114, permute_43);  view_114 = None
        view_115 = torch.ops.aten.view.default(mm_43, [128, 901, 128]);  mm_43 = None
        add_69 = torch.ops.aten.add.Tensor(convert_element_type_233, view_115);  convert_element_type_233 = view_115 = None
        convert_element_type_242 = torch.ops.prims.convert_element_type.default(add_69, torch.float32);  add_69 = None
        pow_22 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_242, 2)
        mean_21 = torch.ops.aten.mean.dim(pow_22, [-1], True);  pow_22 = None
        add_70 = torch.ops.aten.add.Tensor(mean_21, 1e-05);  mean_21 = None
        rsqrt_21 = torch.ops.aten.rsqrt.default(add_70);  add_70 = None
        mul_88 = torch.ops.aten.mul.Tensor(convert_element_type_242, rsqrt_21);  convert_element_type_242 = rsqrt_21 = None
        convert_element_type_243 = torch.ops.prims.convert_element_type.default(mul_88, torch.bfloat16);  mul_88 = None
        convert_element_type_244 = torch.ops.prims.convert_element_type.default(primals_47, torch.bfloat16)
        permute_44 = torch.ops.aten.permute.default(convert_element_type_244, [1, 0]);  convert_element_type_244 = None
        view_116 = torch.ops.aten.view.default(convert_element_type_243, [115328, 128])
        mm_44 = torch.ops.aten.mm.default(view_116, permute_44);  view_116 = None
        view_117 = torch.ops.aten.view.default(mm_44, [128, 901, 384]);  mm_44 = None
        view_118 = torch.ops.aten.view.default(view_117, [128, 901, 6, 64]);  view_117 = None
        slice_146 = torch.ops.aten.slice.Tensor(view_118, 2, 0, 2)
        slice_149 = torch.ops.aten.slice.Tensor(view_118, 2, 2, 4)
        slice_152 = torch.ops.aten.slice.Tensor(view_118, 2, 4, 9223372036854775807);  view_118 = None
        convert_element_type_247 = torch.ops.prims.convert_element_type.default(slice_146, torch.float32);  slice_146 = None
        convert_element_type_248 = torch.ops.prims.convert_element_type.default(slice_149, torch.float32);  slice_149 = None
        mul_89 = torch.ops.aten.mul.Tensor(convert_element_type_247, unsqueeze)
        slice_153 = torch.ops.aten.slice.Tensor(convert_element_type_247, 3, 0, 32)
        slice_154 = torch.ops.aten.slice.Tensor(convert_element_type_247, 3, 32, 9223372036854775807);  convert_element_type_247 = None
        neg_22 = torch.ops.aten.neg.default(slice_154);  slice_154 = None
        cat_23 = torch.ops.aten.cat.default([neg_22, slice_153], -1);  neg_22 = slice_153 = None
        mul_90 = torch.ops.aten.mul.Tensor(cat_23, unsqueeze_1);  cat_23 = None
        add_71 = torch.ops.aten.add.Tensor(mul_89, mul_90);  mul_89 = mul_90 = None
        mul_91 = torch.ops.aten.mul.Tensor(convert_element_type_248, unsqueeze)
        slice_155 = torch.ops.aten.slice.Tensor(convert_element_type_248, 3, 0, 32)
        slice_156 = torch.ops.aten.slice.Tensor(convert_element_type_248, 3, 32, 9223372036854775807);  convert_element_type_248 = None
        neg_23 = torch.ops.aten.neg.default(slice_156);  slice_156 = None
        cat_24 = torch.ops.aten.cat.default([neg_23, slice_155], -1);  neg_23 = slice_155 = None
        mul_92 = torch.ops.aten.mul.Tensor(cat_24, unsqueeze_1);  cat_24 = None
        add_72 = torch.ops.aten.add.Tensor(mul_91, mul_92);  mul_91 = mul_92 = None
        convert_element_type_249 = torch.ops.prims.convert_element_type.default(add_71, torch.bfloat16);  add_71 = None
        convert_element_type_250 = torch.ops.prims.convert_element_type.default(add_72, torch.bfloat16);  add_72 = None
        _flash_attn_forward_11 = torch.ops.flash_attn._flash_attn_forward.default(convert_element_type_249, convert_element_type_250, slice_152, 0.0, 0.125, False, -1, -1, 0.0, None, False);  convert_element_type_249 = convert_element_type_250 = slice_152 = None
        getitem_66 = _flash_attn_forward_11[0];  _flash_attn_forward_11 = None
        view_119 = torch.ops.aten.view.default(getitem_66, [128, 901, 128]);  getitem_66 = None
        convert_element_type_251 = torch.ops.prims.convert_element_type.default(primals_48, torch.bfloat16)
        permute_45 = torch.ops.aten.permute.default(convert_element_type_251, [1, 0]);  convert_element_type_251 = None
        view_120 = torch.ops.aten.view.default(view_119, [115328, 128]);  view_119 = None
        mm_45 = torch.ops.aten.mm.default(view_120, permute_45);  view_120 = None
        view_121 = torch.ops.aten.view.default(mm_45, [128, 901, 128]);  mm_45 = None
        add_73 = torch.ops.aten.add.Tensor(convert_element_type_243, view_121);  convert_element_type_243 = view_121 = None
        convert_element_type_254 = torch.ops.prims.convert_element_type.default(add_73, torch.float32);  add_73 = None
        pow_23 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_254, 2)
        mean_22 = torch.ops.aten.mean.dim(pow_23, [-1], True);  pow_23 = None
        add_74 = torch.ops.aten.add.Tensor(mean_22, 1e-05);  mean_22 = None
        rsqrt_22 = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
        mul_93 = torch.ops.aten.mul.Tensor(convert_element_type_254, rsqrt_22);  convert_element_type_254 = rsqrt_22 = None
        convert_element_type_255 = torch.ops.prims.convert_element_type.default(mul_93, torch.bfloat16);  mul_93 = None
        convert_element_type_256 = torch.ops.prims.convert_element_type.default(primals_49, torch.bfloat16)
        permute_46 = torch.ops.aten.permute.default(convert_element_type_256, [1, 0]);  convert_element_type_256 = None
        view_122 = torch.ops.aten.view.default(convert_element_type_255, [115328, 128])
        mm_46 = torch.ops.aten.mm.default(view_122, permute_46);  view_122 = None
        view_123 = torch.ops.aten.view.default(mm_46, [128, 901, 1024]);  mm_46 = None
        split_11 = torch.ops.aten.split.Tensor(view_123, 512, -1);  view_123 = None
        getitem_70 = split_11[0]
        getitem_71 = split_11[1];  split_11 = None
        convert_element_type_259 = torch.ops.prims.convert_element_type.default(getitem_70, torch.float32);  getitem_70 = None
        sigmoid_11 = torch.ops.aten.sigmoid.default(convert_element_type_259)
        mul_94 = torch.ops.aten.mul.Tensor(convert_element_type_259, sigmoid_11);  convert_element_type_259 = sigmoid_11 = None
        convert_element_type_260 = torch.ops.prims.convert_element_type.default(mul_94, torch.bfloat16);  mul_94 = None
        mul_95 = torch.ops.aten.mul.Tensor(convert_element_type_260, getitem_71);  convert_element_type_260 = getitem_71 = None
        convert_element_type_261 = torch.ops.prims.convert_element_type.default(primals_50, torch.bfloat16)
        permute_47 = torch.ops.aten.permute.default(convert_element_type_261, [1, 0]);  convert_element_type_261 = None
        view_124 = torch.ops.aten.view.default(mul_95, [115328, 512]);  mul_95 = None
        mm_47 = torch.ops.aten.mm.default(view_124, permute_47);  view_124 = None
        view_125 = torch.ops.aten.view.default(mm_47, [128, 901, 128]);  mm_47 = None
        add_75 = torch.ops.aten.add.Tensor(convert_element_type_255, view_125);  convert_element_type_255 = view_125 = None
        convert_element_type_264 = torch.ops.prims.convert_element_type.default(add_75, torch.float32);  add_75 = None
        pow_24 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_264, 2)
        mean_23 = torch.ops.aten.mean.dim(pow_24, [-1], True);  pow_24 = None
        add_76 = torch.ops.aten.add.Tensor(mean_23, 1e-05);  mean_23 = None
        rsqrt_23 = torch.ops.aten.rsqrt.default(add_76);  add_76 = None
        mul_96 = torch.ops.aten.mul.Tensor(convert_element_type_264, rsqrt_23);  convert_element_type_264 = rsqrt_23 = None
        convert_element_type_265 = torch.ops.prims.convert_element_type.default(mul_96, torch.bfloat16);  mul_96 = None
        add_77 = torch.ops.aten.add.Tensor(convert_element_type_265, mul);  mul = None
        add_78 = torch.ops.aten.add.Tensor(convert_element_type_177, add_77);  convert_element_type_177 = None
        view_126 = torch.ops.aten.view.default(add_78, [115328, 128])
        mm_48 = torch.ops.aten.mm.default(view_126, permute);  view_126 = None
        view_127 = torch.ops.aten.view.default(mm_48, [128, 901, 384]);  mm_48 = None
        view_128 = torch.ops.aten.view.default(view_127, [128, 901, 6, 64]);  view_127 = None
        slice_159 = torch.ops.aten.slice.Tensor(view_128, 2, 0, 2)
        slice_162 = torch.ops.aten.slice.Tensor(view_128, 2, 2, 4)
        slice_165 = torch.ops.aten.slice.Tensor(view_128, 2, 4, 9223372036854775807);  view_128 = None
        convert_element_type_269 = torch.ops.prims.convert_element_type.default(slice_159, torch.float32);  slice_159 = None
        convert_element_type_270 = torch.ops.prims.convert_element_type.default(slice_162, torch.float32);  slice_162 = None
        mul_97 = torch.ops.aten.mul.Tensor(convert_element_type_269, unsqueeze)
        slice_166 = torch.ops.aten.slice.Tensor(convert_element_type_269, 3, 0, 32)
        slice_167 = torch.ops.aten.slice.Tensor(convert_element_type_269, 3, 32, 9223372036854775807);  convert_element_type_269 = None
        neg_24 = torch.ops.aten.neg.default(slice_167);  slice_167 = None
        cat_25 = torch.ops.aten.cat.default([neg_24, slice_166], -1);  neg_24 = slice_166 = None
        mul_98 = torch.ops.aten.mul.Tensor(cat_25, unsqueeze_1);  cat_25 = None
        add_79 = torch.ops.aten.add.Tensor(mul_97, mul_98);  mul_97 = mul_98 = None
        mul_99 = torch.ops.aten.mul.Tensor(convert_element_type_270, unsqueeze)
        slice_168 = torch.ops.aten.slice.Tensor(convert_element_type_270, 3, 0, 32)
        slice_169 = torch.ops.aten.slice.Tensor(convert_element_type_270, 3, 32, 9223372036854775807);  convert_element_type_270 = None
        neg_25 = torch.ops.aten.neg.default(slice_169);  slice_169 = None
        cat_26 = torch.ops.aten.cat.default([neg_25, slice_168], -1);  neg_25 = slice_168 = None
        mul_100 = torch.ops.aten.mul.Tensor(cat_26, unsqueeze_1);  cat_26 = None
        add_80 = torch.ops.aten.add.Tensor(mul_99, mul_100);  mul_99 = mul_100 = None
        convert_element_type_271 = torch.ops.prims.convert_element_type.default(add_79, torch.bfloat16);  add_79 = None
        convert_element_type_272 = torch.ops.prims.convert_element_type.default(add_80, torch.bfloat16);  add_80 = None
        _flash_attn_forward_12 = torch.ops.flash_attn._flash_attn_forward.default(convert_element_type_271, convert_element_type_272, slice_165, 0.0, 0.125, False, -1, -1, 0.0, None, False);  convert_element_type_271 = convert_element_type_272 = slice_165 = None
        getitem_72 = _flash_attn_forward_12[0];  _flash_attn_forward_12 = None
        view_129 = torch.ops.aten.view.default(getitem_72, [128, 901, 128]);  getitem_72 = None
        view_130 = torch.ops.aten.view.default(view_129, [115328, 128]);  view_129 = None
        mm_49 = torch.ops.aten.mm.default(view_130, permute_1);  view_130 = None
        view_131 = torch.ops.aten.view.default(mm_49, [128, 901, 128]);  mm_49 = None
        add_81 = torch.ops.aten.add.Tensor(add_78, view_131);  add_78 = view_131 = None
        convert_element_type_276 = torch.ops.prims.convert_element_type.default(add_81, torch.float32);  add_81 = None
        pow_25 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_276, 2)
        mean_24 = torch.ops.aten.mean.dim(pow_25, [-1], True);  pow_25 = None
        add_82 = torch.ops.aten.add.Tensor(mean_24, 1e-05);  mean_24 = None
        rsqrt_24 = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
        mul_101 = torch.ops.aten.mul.Tensor(convert_element_type_276, rsqrt_24);  convert_element_type_276 = rsqrt_24 = None
        convert_element_type_277 = torch.ops.prims.convert_element_type.default(mul_101, torch.bfloat16);  mul_101 = None
        view_132 = torch.ops.aten.view.default(convert_element_type_277, [115328, 128])
        mm_50 = torch.ops.aten.mm.default(view_132, permute_2);  view_132 = None
        view_133 = torch.ops.aten.view.default(mm_50, [128, 901, 1024]);  mm_50 = None
        split_12 = torch.ops.aten.split.Tensor(view_133, 512, -1);  view_133 = None
        getitem_76 = split_12[0]
        getitem_77 = split_12[1];  split_12 = None
        convert_element_type_281 = torch.ops.prims.convert_element_type.default(getitem_76, torch.float32);  getitem_76 = None
        sigmoid_12 = torch.ops.aten.sigmoid.default(convert_element_type_281)
        mul_102 = torch.ops.aten.mul.Tensor(convert_element_type_281, sigmoid_12);  convert_element_type_281 = sigmoid_12 = None
        convert_element_type_282 = torch.ops.prims.convert_element_type.default(mul_102, torch.bfloat16);  mul_102 = None
        mul_103 = torch.ops.aten.mul.Tensor(convert_element_type_282, getitem_77);  convert_element_type_282 = getitem_77 = None
        view_134 = torch.ops.aten.view.default(mul_103, [115328, 512]);  mul_103 = None
        mm_51 = torch.ops.aten.mm.default(view_134, permute_3);  view_134 = None
        view_135 = torch.ops.aten.view.default(mm_51, [128, 901, 128]);  mm_51 = None
        add_83 = torch.ops.aten.add.Tensor(convert_element_type_277, view_135);  convert_element_type_277 = view_135 = None
        convert_element_type_286 = torch.ops.prims.convert_element_type.default(add_83, torch.float32);  add_83 = None
        pow_26 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_286, 2)
        mean_25 = torch.ops.aten.mean.dim(pow_26, [-1], True);  pow_26 = None
        add_84 = torch.ops.aten.add.Tensor(mean_25, 1e-05);  mean_25 = None
        rsqrt_25 = torch.ops.aten.rsqrt.default(add_84);  add_84 = None
        mul_104 = torch.ops.aten.mul.Tensor(convert_element_type_286, rsqrt_25);  convert_element_type_286 = rsqrt_25 = None
        convert_element_type_287 = torch.ops.prims.convert_element_type.default(mul_104, torch.bfloat16);  mul_104 = None
        view_136 = torch.ops.aten.view.default(convert_element_type_287, [115328, 128])
        mm_52 = torch.ops.aten.mm.default(view_136, permute_4);  view_136 = None
        view_137 = torch.ops.aten.view.default(mm_52, [128, 901, 384]);  mm_52 = None
        view_138 = torch.ops.aten.view.default(view_137, [128, 901, 6, 64]);  view_137 = None
        slice_172 = torch.ops.aten.slice.Tensor(view_138, 2, 0, 2)
        slice_175 = torch.ops.aten.slice.Tensor(view_138, 2, 2, 4)
        slice_178 = torch.ops.aten.slice.Tensor(view_138, 2, 4, 9223372036854775807);  view_138 = None
        convert_element_type_291 = torch.ops.prims.convert_element_type.default(slice_172, torch.float32);  slice_172 = None
        convert_element_type_292 = torch.ops.prims.convert_element_type.default(slice_175, torch.float32);  slice_175 = None
        mul_105 = torch.ops.aten.mul.Tensor(convert_element_type_291, unsqueeze)
        slice_179 = torch.ops.aten.slice.Tensor(convert_element_type_291, 3, 0, 32)
        slice_180 = torch.ops.aten.slice.Tensor(convert_element_type_291, 3, 32, 9223372036854775807);  convert_element_type_291 = None
        neg_26 = torch.ops.aten.neg.default(slice_180);  slice_180 = None
        cat_27 = torch.ops.aten.cat.default([neg_26, slice_179], -1);  neg_26 = slice_179 = None
        mul_106 = torch.ops.aten.mul.Tensor(cat_27, unsqueeze_1);  cat_27 = None
        add_85 = torch.ops.aten.add.Tensor(mul_105, mul_106);  mul_105 = mul_106 = None
        mul_107 = torch.ops.aten.mul.Tensor(convert_element_type_292, unsqueeze)
        slice_181 = torch.ops.aten.slice.Tensor(convert_element_type_292, 3, 0, 32)
        slice_182 = torch.ops.aten.slice.Tensor(convert_element_type_292, 3, 32, 9223372036854775807);  convert_element_type_292 = None
        neg_27 = torch.ops.aten.neg.default(slice_182);  slice_182 = None
        cat_28 = torch.ops.aten.cat.default([neg_27, slice_181], -1);  neg_27 = slice_181 = None
        mul_108 = torch.ops.aten.mul.Tensor(cat_28, unsqueeze_1);  cat_28 = None
        add_86 = torch.ops.aten.add.Tensor(mul_107, mul_108);  mul_107 = mul_108 = None
        convert_element_type_293 = torch.ops.prims.convert_element_type.default(add_85, torch.bfloat16);  add_85 = None
        convert_element_type_294 = torch.ops.prims.convert_element_type.default(add_86, torch.bfloat16);  add_86 = None
        _flash_attn_forward_13 = torch.ops.flash_attn._flash_attn_forward.default(convert_element_type_293, convert_element_type_294, slice_178, 0.0, 0.125, False, -1, -1, 0.0, None, False);  convert_element_type_293 = convert_element_type_294 = slice_178 = None
        getitem_78 = _flash_attn_forward_13[0];  _flash_attn_forward_13 = None
        view_139 = torch.ops.aten.view.default(getitem_78, [128, 901, 128]);  getitem_78 = None
        view_140 = torch.ops.aten.view.default(view_139, [115328, 128]);  view_139 = None
        mm_53 = torch.ops.aten.mm.default(view_140, permute_5);  view_140 = None
        view_141 = torch.ops.aten.view.default(mm_53, [128, 901, 128]);  mm_53 = None
        add_87 = torch.ops.aten.add.Tensor(convert_element_type_287, view_141);  convert_element_type_287 = view_141 = None
        convert_element_type_298 = torch.ops.prims.convert_element_type.default(add_87, torch.float32);  add_87 = None
        pow_27 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_298, 2)
        mean_26 = torch.ops.aten.mean.dim(pow_27, [-1], True);  pow_27 = None
        add_88 = torch.ops.aten.add.Tensor(mean_26, 1e-05);  mean_26 = None
        rsqrt_26 = torch.ops.aten.rsqrt.default(add_88);  add_88 = None
        mul_109 = torch.ops.aten.mul.Tensor(convert_element_type_298, rsqrt_26);  convert_element_type_298 = rsqrt_26 = None
        convert_element_type_299 = torch.ops.prims.convert_element_type.default(mul_109, torch.bfloat16);  mul_109 = None
        view_142 = torch.ops.aten.view.default(convert_element_type_299, [115328, 128])
        mm_54 = torch.ops.aten.mm.default(view_142, permute_6);  view_142 = None
        view_143 = torch.ops.aten.view.default(mm_54, [128, 901, 1024]);  mm_54 = None
        split_13 = torch.ops.aten.split.Tensor(view_143, 512, -1);  view_143 = None
        getitem_82 = split_13[0]
        getitem_83 = split_13[1];  split_13 = None
        convert_element_type_303 = torch.ops.prims.convert_element_type.default(getitem_82, torch.float32);  getitem_82 = None
        sigmoid_13 = torch.ops.aten.sigmoid.default(convert_element_type_303)
        mul_110 = torch.ops.aten.mul.Tensor(convert_element_type_303, sigmoid_13);  convert_element_type_303 = sigmoid_13 = None
        convert_element_type_304 = torch.ops.prims.convert_element_type.default(mul_110, torch.bfloat16);  mul_110 = None
        mul_111 = torch.ops.aten.mul.Tensor(convert_element_type_304, getitem_83);  convert_element_type_304 = getitem_83 = None
        view_144 = torch.ops.aten.view.default(mul_111, [115328, 512]);  mul_111 = None
        mm_55 = torch.ops.aten.mm.default(view_144, permute_7);  view_144 = None
        view_145 = torch.ops.aten.view.default(mm_55, [128, 901, 128]);  mm_55 = None
        add_89 = torch.ops.aten.add.Tensor(convert_element_type_299, view_145);  convert_element_type_299 = view_145 = None
        convert_element_type_308 = torch.ops.prims.convert_element_type.default(add_89, torch.float32);  add_89 = None
        pow_28 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_308, 2)
        mean_27 = torch.ops.aten.mean.dim(pow_28, [-1], True);  pow_28 = None
        add_90 = torch.ops.aten.add.Tensor(mean_27, 1e-05);  mean_27 = None
        rsqrt_27 = torch.ops.aten.rsqrt.default(add_90);  add_90 = None
        mul_112 = torch.ops.aten.mul.Tensor(convert_element_type_308, rsqrt_27);  convert_element_type_308 = rsqrt_27 = None
        convert_element_type_309 = torch.ops.prims.convert_element_type.default(mul_112, torch.bfloat16);  mul_112 = None
        view_146 = torch.ops.aten.view.default(convert_element_type_309, [115328, 128])
        mm_56 = torch.ops.aten.mm.default(view_146, permute_8);  view_146 = None
        view_147 = torch.ops.aten.view.default(mm_56, [128, 901, 384]);  mm_56 = None
        view_148 = torch.ops.aten.view.default(view_147, [128, 901, 6, 64]);  view_147 = None
        slice_185 = torch.ops.aten.slice.Tensor(view_148, 2, 0, 2)
        slice_188 = torch.ops.aten.slice.Tensor(view_148, 2, 2, 4)
        slice_191 = torch.ops.aten.slice.Tensor(view_148, 2, 4, 9223372036854775807);  view_148 = None
        convert_element_type_313 = torch.ops.prims.convert_element_type.default(slice_185, torch.float32);  slice_185 = None
        convert_element_type_314 = torch.ops.prims.convert_element_type.default(slice_188, torch.float32);  slice_188 = None
        mul_113 = torch.ops.aten.mul.Tensor(convert_element_type_313, unsqueeze)
        slice_192 = torch.ops.aten.slice.Tensor(convert_element_type_313, 3, 0, 32)
        slice_193 = torch.ops.aten.slice.Tensor(convert_element_type_313, 3, 32, 9223372036854775807);  convert_element_type_313 = None
        neg_28 = torch.ops.aten.neg.default(slice_193);  slice_193 = None
        cat_29 = torch.ops.aten.cat.default([neg_28, slice_192], -1);  neg_28 = slice_192 = None
        mul_114 = torch.ops.aten.mul.Tensor(cat_29, unsqueeze_1);  cat_29 = None
        add_91 = torch.ops.aten.add.Tensor(mul_113, mul_114);  mul_113 = mul_114 = None
        mul_115 = torch.ops.aten.mul.Tensor(convert_element_type_314, unsqueeze)
        slice_194 = torch.ops.aten.slice.Tensor(convert_element_type_314, 3, 0, 32)
        slice_195 = torch.ops.aten.slice.Tensor(convert_element_type_314, 3, 32, 9223372036854775807);  convert_element_type_314 = None
        neg_29 = torch.ops.aten.neg.default(slice_195);  slice_195 = None
        cat_30 = torch.ops.aten.cat.default([neg_29, slice_194], -1);  neg_29 = slice_194 = None
        mul_116 = torch.ops.aten.mul.Tensor(cat_30, unsqueeze_1);  cat_30 = None
        add_92 = torch.ops.aten.add.Tensor(mul_115, mul_116);  mul_115 = mul_116 = None
        convert_element_type_315 = torch.ops.prims.convert_element_type.default(add_91, torch.bfloat16);  add_91 = None
        convert_element_type_316 = torch.ops.prims.convert_element_type.default(add_92, torch.bfloat16);  add_92 = None
        _flash_attn_forward_14 = torch.ops.flash_attn._flash_attn_forward.default(convert_element_type_315, convert_element_type_316, slice_191, 0.0, 0.125, False, -1, -1, 0.0, None, False);  convert_element_type_315 = convert_element_type_316 = slice_191 = None
        getitem_84 = _flash_attn_forward_14[0];  _flash_attn_forward_14 = None
        view_149 = torch.ops.aten.view.default(getitem_84, [128, 901, 128]);  getitem_84 = None
        view_150 = torch.ops.aten.view.default(view_149, [115328, 128]);  view_149 = None
        mm_57 = torch.ops.aten.mm.default(view_150, permute_9);  view_150 = None
        view_151 = torch.ops.aten.view.default(mm_57, [128, 901, 128]);  mm_57 = None
        add_93 = torch.ops.aten.add.Tensor(convert_element_type_309, view_151);  convert_element_type_309 = view_151 = None
        convert_element_type_320 = torch.ops.prims.convert_element_type.default(add_93, torch.float32);  add_93 = None
        pow_29 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_320, 2)
        mean_28 = torch.ops.aten.mean.dim(pow_29, [-1], True);  pow_29 = None
        add_94 = torch.ops.aten.add.Tensor(mean_28, 1e-05);  mean_28 = None
        rsqrt_28 = torch.ops.aten.rsqrt.default(add_94);  add_94 = None
        mul_117 = torch.ops.aten.mul.Tensor(convert_element_type_320, rsqrt_28);  convert_element_type_320 = rsqrt_28 = None
        convert_element_type_321 = torch.ops.prims.convert_element_type.default(mul_117, torch.bfloat16);  mul_117 = None
        view_152 = torch.ops.aten.view.default(convert_element_type_321, [115328, 128])
        mm_58 = torch.ops.aten.mm.default(view_152, permute_10);  view_152 = None
        view_153 = torch.ops.aten.view.default(mm_58, [128, 901, 1024]);  mm_58 = None
        split_14 = torch.ops.aten.split.Tensor(view_153, 512, -1);  view_153 = None
        getitem_88 = split_14[0]
        getitem_89 = split_14[1];  split_14 = None
        convert_element_type_325 = torch.ops.prims.convert_element_type.default(getitem_88, torch.float32);  getitem_88 = None
        sigmoid_14 = torch.ops.aten.sigmoid.default(convert_element_type_325)
        mul_118 = torch.ops.aten.mul.Tensor(convert_element_type_325, sigmoid_14);  convert_element_type_325 = sigmoid_14 = None
        convert_element_type_326 = torch.ops.prims.convert_element_type.default(mul_118, torch.bfloat16);  mul_118 = None
        mul_119 = torch.ops.aten.mul.Tensor(convert_element_type_326, getitem_89);  convert_element_type_326 = getitem_89 = None
        view_154 = torch.ops.aten.view.default(mul_119, [115328, 512]);  mul_119 = None
        mm_59 = torch.ops.aten.mm.default(view_154, permute_11);  view_154 = None
        view_155 = torch.ops.aten.view.default(mm_59, [128, 901, 128]);  mm_59 = None
        add_95 = torch.ops.aten.add.Tensor(convert_element_type_321, view_155);  convert_element_type_321 = view_155 = None
        convert_element_type_330 = torch.ops.prims.convert_element_type.default(add_95, torch.float32);  add_95 = None
        pow_30 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_330, 2)
        mean_29 = torch.ops.aten.mean.dim(pow_30, [-1], True);  pow_30 = None
        add_96 = torch.ops.aten.add.Tensor(mean_29, 1e-05);  mean_29 = None
        rsqrt_29 = torch.ops.aten.rsqrt.default(add_96);  add_96 = None
        mul_120 = torch.ops.aten.mul.Tensor(convert_element_type_330, rsqrt_29);  convert_element_type_330 = rsqrt_29 = None
        convert_element_type_331 = torch.ops.prims.convert_element_type.default(mul_120, torch.bfloat16);  mul_120 = None
        view_156 = torch.ops.aten.view.default(convert_element_type_331, [115328, 128])
        mm_60 = torch.ops.aten.mm.default(view_156, permute_12);  view_156 = None
        view_157 = torch.ops.aten.view.default(mm_60, [128, 901, 384]);  mm_60 = None
        view_158 = torch.ops.aten.view.default(view_157, [128, 901, 6, 64]);  view_157 = None
        slice_198 = torch.ops.aten.slice.Tensor(view_158, 2, 0, 2)
        slice_201 = torch.ops.aten.slice.Tensor(view_158, 2, 2, 4)
        slice_204 = torch.ops.aten.slice.Tensor(view_158, 2, 4, 9223372036854775807);  view_158 = None
        convert_element_type_335 = torch.ops.prims.convert_element_type.default(slice_198, torch.float32);  slice_198 = None
        convert_element_type_336 = torch.ops.prims.convert_element_type.default(slice_201, torch.float32);  slice_201 = None
        mul_121 = torch.ops.aten.mul.Tensor(convert_element_type_335, unsqueeze)
        slice_205 = torch.ops.aten.slice.Tensor(convert_element_type_335, 3, 0, 32)
        slice_206 = torch.ops.aten.slice.Tensor(convert_element_type_335, 3, 32, 9223372036854775807);  convert_element_type_335 = None
        neg_30 = torch.ops.aten.neg.default(slice_206);  slice_206 = None
        cat_31 = torch.ops.aten.cat.default([neg_30, slice_205], -1);  neg_30 = slice_205 = None
        mul_122 = torch.ops.aten.mul.Tensor(cat_31, unsqueeze_1);  cat_31 = None
        add_97 = torch.ops.aten.add.Tensor(mul_121, mul_122);  mul_121 = mul_122 = None
        mul_123 = torch.ops.aten.mul.Tensor(convert_element_type_336, unsqueeze)
        slice_207 = torch.ops.aten.slice.Tensor(convert_element_type_336, 3, 0, 32)
        slice_208 = torch.ops.aten.slice.Tensor(convert_element_type_336, 3, 32, 9223372036854775807);  convert_element_type_336 = None
        neg_31 = torch.ops.aten.neg.default(slice_208);  slice_208 = None
        cat_32 = torch.ops.aten.cat.default([neg_31, slice_207], -1);  neg_31 = slice_207 = None
        mul_124 = torch.ops.aten.mul.Tensor(cat_32, unsqueeze_1);  cat_32 = None
        add_98 = torch.ops.aten.add.Tensor(mul_123, mul_124);  mul_123 = mul_124 = None
        convert_element_type_337 = torch.ops.prims.convert_element_type.default(add_97, torch.bfloat16);  add_97 = None
        convert_element_type_338 = torch.ops.prims.convert_element_type.default(add_98, torch.bfloat16);  add_98 = None
        _flash_attn_forward_15 = torch.ops.flash_attn._flash_attn_forward.default(convert_element_type_337, convert_element_type_338, slice_204, 0.0, 0.125, False, -1, -1, 0.0, None, False);  convert_element_type_337 = convert_element_type_338 = slice_204 = None
        getitem_90 = _flash_attn_forward_15[0];  _flash_attn_forward_15 = None
        view_159 = torch.ops.aten.view.default(getitem_90, [128, 901, 128]);  getitem_90 = None
        view_160 = torch.ops.aten.view.default(view_159, [115328, 128]);  view_159 = None
        mm_61 = torch.ops.aten.mm.default(view_160, permute_13);  view_160 = None
        view_161 = torch.ops.aten.view.default(mm_61, [128, 901, 128]);  mm_61 = None
        add_99 = torch.ops.aten.add.Tensor(convert_element_type_331, view_161);  convert_element_type_331 = view_161 = None
        convert_element_type_342 = torch.ops.prims.convert_element_type.default(add_99, torch.float32);  add_99 = None
        pow_31 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_342, 2)
        mean_30 = torch.ops.aten.mean.dim(pow_31, [-1], True);  pow_31 = None
        add_100 = torch.ops.aten.add.Tensor(mean_30, 1e-05);  mean_30 = None
        rsqrt_30 = torch.ops.aten.rsqrt.default(add_100);  add_100 = None
        mul_125 = torch.ops.aten.mul.Tensor(convert_element_type_342, rsqrt_30);  convert_element_type_342 = rsqrt_30 = None
        convert_element_type_343 = torch.ops.prims.convert_element_type.default(mul_125, torch.bfloat16);  mul_125 = None
        view_162 = torch.ops.aten.view.default(convert_element_type_343, [115328, 128])
        mm_62 = torch.ops.aten.mm.default(view_162, permute_14);  view_162 = None
        view_163 = torch.ops.aten.view.default(mm_62, [128, 901, 1024]);  mm_62 = None
        split_15 = torch.ops.aten.split.Tensor(view_163, 512, -1);  view_163 = None
        getitem_94 = split_15[0]
        getitem_95 = split_15[1];  split_15 = None
        convert_element_type_347 = torch.ops.prims.convert_element_type.default(getitem_94, torch.float32);  getitem_94 = None
        sigmoid_15 = torch.ops.aten.sigmoid.default(convert_element_type_347)
        mul_126 = torch.ops.aten.mul.Tensor(convert_element_type_347, sigmoid_15);  convert_element_type_347 = sigmoid_15 = None
        convert_element_type_348 = torch.ops.prims.convert_element_type.default(mul_126, torch.bfloat16);  mul_126 = None
        mul_127 = torch.ops.aten.mul.Tensor(convert_element_type_348, getitem_95);  convert_element_type_348 = getitem_95 = None
        view_164 = torch.ops.aten.view.default(mul_127, [115328, 512]);  mul_127 = None
        mm_63 = torch.ops.aten.mm.default(view_164, permute_15);  view_164 = None
        view_165 = torch.ops.aten.view.default(mm_63, [128, 901, 128]);  mm_63 = None
        add_101 = torch.ops.aten.add.Tensor(convert_element_type_343, view_165);  convert_element_type_343 = view_165 = None
        convert_element_type_352 = torch.ops.prims.convert_element_type.default(add_101, torch.float32);  add_101 = None
        pow_32 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_352, 2)
        mean_31 = torch.ops.aten.mean.dim(pow_32, [-1], True);  pow_32 = None
        add_102 = torch.ops.aten.add.Tensor(mean_31, 1e-05);  mean_31 = None
        rsqrt_31 = torch.ops.aten.rsqrt.default(add_102);  add_102 = None
        mul_128 = torch.ops.aten.mul.Tensor(convert_element_type_352, rsqrt_31);  convert_element_type_352 = rsqrt_31 = None
        convert_element_type_353 = torch.ops.prims.convert_element_type.default(mul_128, torch.bfloat16);  mul_128 = None
        add_104 = torch.ops.aten.add.Tensor(convert_element_type_353, add_77);  convert_element_type_353 = add_77 = None
        view_166 = torch.ops.aten.view.default(add_104, [115328, 128])
        mm_64 = torch.ops.aten.mm.default(view_166, permute)
        view_167 = torch.ops.aten.view.default(mm_64, [128, 901, 384]);  mm_64 = None
        view_168 = torch.ops.aten.view.default(view_167, [128, 901, 6, 64]);  view_167 = None
        slice_211 = torch.ops.aten.slice.Tensor(view_168, 2, 0, 2)
        slice_214 = torch.ops.aten.slice.Tensor(view_168, 2, 2, 4)
        slice_217 = torch.ops.aten.slice.Tensor(view_168, 2, 4, 9223372036854775807);  view_168 = None
        convert_element_type_357 = torch.ops.prims.convert_element_type.default(slice_211, torch.float32);  slice_211 = None
        convert_element_type_358 = torch.ops.prims.convert_element_type.default(slice_214, torch.float32);  slice_214 = None
        mul_129 = torch.ops.aten.mul.Tensor(convert_element_type_357, unsqueeze)
        slice_218 = torch.ops.aten.slice.Tensor(convert_element_type_357, 3, 0, 32)
        slice_219 = torch.ops.aten.slice.Tensor(convert_element_type_357, 3, 32, 9223372036854775807);  convert_element_type_357 = None
        neg_32 = torch.ops.aten.neg.default(slice_219);  slice_219 = None
        cat_33 = torch.ops.aten.cat.default([neg_32, slice_218], -1);  neg_32 = slice_218 = None
        mul_130 = torch.ops.aten.mul.Tensor(cat_33, unsqueeze_1);  cat_33 = None
        add_105 = torch.ops.aten.add.Tensor(mul_129, mul_130);  mul_129 = mul_130 = None
        mul_131 = torch.ops.aten.mul.Tensor(convert_element_type_358, unsqueeze)
        slice_220 = torch.ops.aten.slice.Tensor(convert_element_type_358, 3, 0, 32)
        slice_221 = torch.ops.aten.slice.Tensor(convert_element_type_358, 3, 32, 9223372036854775807);  convert_element_type_358 = None
        neg_33 = torch.ops.aten.neg.default(slice_221);  slice_221 = None
        cat_34 = torch.ops.aten.cat.default([neg_33, slice_220], -1);  neg_33 = slice_220 = None
        mul_132 = torch.ops.aten.mul.Tensor(cat_34, unsqueeze_1);  cat_34 = None
        add_106 = torch.ops.aten.add.Tensor(mul_131, mul_132);  mul_131 = mul_132 = None
        convert_element_type_359 = torch.ops.prims.convert_element_type.default(add_105, torch.bfloat16);  add_105 = None
        convert_element_type_360 = torch.ops.prims.convert_element_type.default(add_106, torch.bfloat16);  add_106 = None
        _flash_attn_forward_16 = torch.ops.flash_attn._flash_attn_forward.default(convert_element_type_359, convert_element_type_360, slice_217, 0.0, 0.125, False, -1, -1, 0.0, None, False)
        getitem_96 = _flash_attn_forward_16[0]
        getitem_97 = _flash_attn_forward_16[1]
        getitem_99 = _flash_attn_forward_16[3];  _flash_attn_forward_16 = None
        view_169 = torch.ops.aten.view.default(getitem_96, [128, 901, 128])
        view_170 = torch.ops.aten.view.default(view_169, [115328, 128]);  view_169 = None
        mm_65 = torch.ops.aten.mm.default(view_170, permute_1);  view_170 = None
        view_171 = torch.ops.aten.view.default(mm_65, [128, 901, 128]);  mm_65 = None
        add_107 = torch.ops.aten.add.Tensor(add_104, view_171);  add_104 = view_171 = None
        convert_element_type_364 = torch.ops.prims.convert_element_type.default(add_107, torch.float32)
        pow_33 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_364, 2)
        mean_32 = torch.ops.aten.mean.dim(pow_33, [-1], True);  pow_33 = None
        add_108 = torch.ops.aten.add.Tensor(mean_32, 1e-05);  mean_32 = None
        rsqrt_32 = torch.ops.aten.rsqrt.default(add_108);  add_108 = None
        mul_133 = torch.ops.aten.mul.Tensor(convert_element_type_364, rsqrt_32);  convert_element_type_364 = None
        convert_element_type_365 = torch.ops.prims.convert_element_type.default(mul_133, torch.bfloat16);  mul_133 = None
        view_172 = torch.ops.aten.view.default(convert_element_type_365, [115328, 128])
        mm_66 = torch.ops.aten.mm.default(view_172, permute_2)
        view_173 = torch.ops.aten.view.default(mm_66, [128, 901, 1024]);  mm_66 = None
        split_16 = torch.ops.aten.split.Tensor(view_173, 512, -1);  view_173 = None
        getitem_100 = split_16[0]
        getitem_101 = split_16[1];  split_16 = None
        convert_element_type_369 = torch.ops.prims.convert_element_type.default(getitem_100, torch.float32)
        sigmoid_16 = torch.ops.aten.sigmoid.default(convert_element_type_369)
        mul_134 = torch.ops.aten.mul.Tensor(convert_element_type_369, sigmoid_16);  convert_element_type_369 = sigmoid_16 = None
        convert_element_type_370 = torch.ops.prims.convert_element_type.default(mul_134, torch.bfloat16);  mul_134 = None
        mul_135 = torch.ops.aten.mul.Tensor(convert_element_type_370, getitem_101);  convert_element_type_370 = None
        view_174 = torch.ops.aten.view.default(mul_135, [115328, 512]);  mul_135 = None
        mm_67 = torch.ops.aten.mm.default(view_174, permute_3)
        view_175 = torch.ops.aten.view.default(mm_67, [128, 901, 128]);  mm_67 = None
        add_109 = torch.ops.aten.add.Tensor(convert_element_type_365, view_175);  convert_element_type_365 = view_175 = None
        convert_element_type_374 = torch.ops.prims.convert_element_type.default(add_109, torch.float32)
        pow_34 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_374, 2)
        mean_33 = torch.ops.aten.mean.dim(pow_34, [-1], True);  pow_34 = None
        add_110 = torch.ops.aten.add.Tensor(mean_33, 1e-05);  mean_33 = None
        rsqrt_33 = torch.ops.aten.rsqrt.default(add_110);  add_110 = None
        mul_136 = torch.ops.aten.mul.Tensor(convert_element_type_374, rsqrt_33);  convert_element_type_374 = None
        convert_element_type_375 = torch.ops.prims.convert_element_type.default(mul_136, torch.bfloat16);  mul_136 = None
        view_176 = torch.ops.aten.view.default(convert_element_type_375, [115328, 128])
        mm_68 = torch.ops.aten.mm.default(view_176, permute_4)
        view_177 = torch.ops.aten.view.default(mm_68, [128, 901, 384]);  mm_68 = None
        view_178 = torch.ops.aten.view.default(view_177, [128, 901, 6, 64]);  view_177 = None
        slice_224 = torch.ops.aten.slice.Tensor(view_178, 2, 0, 2)
        slice_227 = torch.ops.aten.slice.Tensor(view_178, 2, 2, 4)
        slice_230 = torch.ops.aten.slice.Tensor(view_178, 2, 4, 9223372036854775807);  view_178 = None
        convert_element_type_379 = torch.ops.prims.convert_element_type.default(slice_224, torch.float32);  slice_224 = None
        convert_element_type_380 = torch.ops.prims.convert_element_type.default(slice_227, torch.float32);  slice_227 = None
        mul_137 = torch.ops.aten.mul.Tensor(convert_element_type_379, unsqueeze)
        slice_231 = torch.ops.aten.slice.Tensor(convert_element_type_379, 3, 0, 32)
        slice_232 = torch.ops.aten.slice.Tensor(convert_element_type_379, 3, 32, 9223372036854775807);  convert_element_type_379 = None
        neg_34 = torch.ops.aten.neg.default(slice_232);  slice_232 = None
        cat_35 = torch.ops.aten.cat.default([neg_34, slice_231], -1);  neg_34 = slice_231 = None
        mul_138 = torch.ops.aten.mul.Tensor(cat_35, unsqueeze_1);  cat_35 = None
        add_111 = torch.ops.aten.add.Tensor(mul_137, mul_138);  mul_137 = mul_138 = None
        mul_139 = torch.ops.aten.mul.Tensor(convert_element_type_380, unsqueeze)
        slice_233 = torch.ops.aten.slice.Tensor(convert_element_type_380, 3, 0, 32)
        slice_234 = torch.ops.aten.slice.Tensor(convert_element_type_380, 3, 32, 9223372036854775807);  convert_element_type_380 = None
        neg_35 = torch.ops.aten.neg.default(slice_234);  slice_234 = None
        cat_36 = torch.ops.aten.cat.default([neg_35, slice_233], -1);  neg_35 = slice_233 = None
        mul_140 = torch.ops.aten.mul.Tensor(cat_36, unsqueeze_1);  cat_36 = None
        add_112 = torch.ops.aten.add.Tensor(mul_139, mul_140);  mul_139 = mul_140 = None
        convert_element_type_381 = torch.ops.prims.convert_element_type.default(add_111, torch.bfloat16);  add_111 = None
        convert_element_type_382 = torch.ops.prims.convert_element_type.default(add_112, torch.bfloat16);  add_112 = None
        _flash_attn_forward_17 = torch.ops.flash_attn._flash_attn_forward.default(convert_element_type_381, convert_element_type_382, slice_230, 0.0, 0.125, False, -1, -1, 0.0, None, False)
        getitem_102 = _flash_attn_forward_17[0]
        getitem_103 = _flash_attn_forward_17[1]
        getitem_105 = _flash_attn_forward_17[3];  _flash_attn_forward_17 = None
        view_179 = torch.ops.aten.view.default(getitem_102, [128, 901, 128])
        view_180 = torch.ops.aten.view.default(view_179, [115328, 128]);  view_179 = None
        mm_69 = torch.ops.aten.mm.default(view_180, permute_5);  view_180 = None
        view_181 = torch.ops.aten.view.default(mm_69, [128, 901, 128]);  mm_69 = None
        add_113 = torch.ops.aten.add.Tensor(convert_element_type_375, view_181);  convert_element_type_375 = view_181 = None
        convert_element_type_386 = torch.ops.prims.convert_element_type.default(add_113, torch.float32)
        pow_35 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_386, 2)
        mean_34 = torch.ops.aten.mean.dim(pow_35, [-1], True);  pow_35 = None
        add_114 = torch.ops.aten.add.Tensor(mean_34, 1e-05);  mean_34 = None
        rsqrt_34 = torch.ops.aten.rsqrt.default(add_114);  add_114 = None
        mul_141 = torch.ops.aten.mul.Tensor(convert_element_type_386, rsqrt_34);  convert_element_type_386 = None
        convert_element_type_387 = torch.ops.prims.convert_element_type.default(mul_141, torch.bfloat16);  mul_141 = None
        view_182 = torch.ops.aten.view.default(convert_element_type_387, [115328, 128])
        mm_70 = torch.ops.aten.mm.default(view_182, permute_6)
        view_183 = torch.ops.aten.view.default(mm_70, [128, 901, 1024]);  mm_70 = None
        split_17 = torch.ops.aten.split.Tensor(view_183, 512, -1);  view_183 = None
        getitem_106 = split_17[0]
        getitem_107 = split_17[1];  split_17 = None
        convert_element_type_391 = torch.ops.prims.convert_element_type.default(getitem_106, torch.float32)
        sigmoid_17 = torch.ops.aten.sigmoid.default(convert_element_type_391)
        mul_142 = torch.ops.aten.mul.Tensor(convert_element_type_391, sigmoid_17);  convert_element_type_391 = sigmoid_17 = None
        convert_element_type_392 = torch.ops.prims.convert_element_type.default(mul_142, torch.bfloat16);  mul_142 = None
        mul_143 = torch.ops.aten.mul.Tensor(convert_element_type_392, getitem_107);  convert_element_type_392 = None
        view_184 = torch.ops.aten.view.default(mul_143, [115328, 512]);  mul_143 = None
        mm_71 = torch.ops.aten.mm.default(view_184, permute_7)
        view_185 = torch.ops.aten.view.default(mm_71, [128, 901, 128]);  mm_71 = None
        add_115 = torch.ops.aten.add.Tensor(convert_element_type_387, view_185);  convert_element_type_387 = view_185 = None
        convert_element_type_396 = torch.ops.prims.convert_element_type.default(add_115, torch.float32)
        pow_36 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_396, 2)
        mean_35 = torch.ops.aten.mean.dim(pow_36, [-1], True);  pow_36 = None
        add_116 = torch.ops.aten.add.Tensor(mean_35, 1e-05);  mean_35 = None
        rsqrt_35 = torch.ops.aten.rsqrt.default(add_116);  add_116 = None
        mul_144 = torch.ops.aten.mul.Tensor(convert_element_type_396, rsqrt_35);  convert_element_type_396 = None
        convert_element_type_397 = torch.ops.prims.convert_element_type.default(mul_144, torch.bfloat16);  mul_144 = None
        view_186 = torch.ops.aten.view.default(convert_element_type_397, [115328, 128])
        mm_72 = torch.ops.aten.mm.default(view_186, permute_8)
        view_187 = torch.ops.aten.view.default(mm_72, [128, 901, 384]);  mm_72 = None
        view_188 = torch.ops.aten.view.default(view_187, [128, 901, 6, 64]);  view_187 = None
        slice_237 = torch.ops.aten.slice.Tensor(view_188, 2, 0, 2)
        slice_240 = torch.ops.aten.slice.Tensor(view_188, 2, 2, 4)
        slice_243 = torch.ops.aten.slice.Tensor(view_188, 2, 4, 9223372036854775807);  view_188 = None
        convert_element_type_401 = torch.ops.prims.convert_element_type.default(slice_237, torch.float32);  slice_237 = None
        convert_element_type_402 = torch.ops.prims.convert_element_type.default(slice_240, torch.float32);  slice_240 = None
        mul_145 = torch.ops.aten.mul.Tensor(convert_element_type_401, unsqueeze)
        slice_244 = torch.ops.aten.slice.Tensor(convert_element_type_401, 3, 0, 32)
        slice_245 = torch.ops.aten.slice.Tensor(convert_element_type_401, 3, 32, 9223372036854775807);  convert_element_type_401 = None
        neg_36 = torch.ops.aten.neg.default(slice_245);  slice_245 = None
        cat_37 = torch.ops.aten.cat.default([neg_36, slice_244], -1);  neg_36 = slice_244 = None
        mul_146 = torch.ops.aten.mul.Tensor(cat_37, unsqueeze_1);  cat_37 = None
        add_117 = torch.ops.aten.add.Tensor(mul_145, mul_146);  mul_145 = mul_146 = None
        mul_147 = torch.ops.aten.mul.Tensor(convert_element_type_402, unsqueeze)
        slice_246 = torch.ops.aten.slice.Tensor(convert_element_type_402, 3, 0, 32)
        slice_247 = torch.ops.aten.slice.Tensor(convert_element_type_402, 3, 32, 9223372036854775807);  convert_element_type_402 = None
        neg_37 = torch.ops.aten.neg.default(slice_247);  slice_247 = None
        cat_38 = torch.ops.aten.cat.default([neg_37, slice_246], -1);  neg_37 = slice_246 = None
        mul_148 = torch.ops.aten.mul.Tensor(cat_38, unsqueeze_1);  cat_38 = None
        add_118 = torch.ops.aten.add.Tensor(mul_147, mul_148);  mul_147 = mul_148 = None
        convert_element_type_403 = torch.ops.prims.convert_element_type.default(add_117, torch.bfloat16);  add_117 = None
        convert_element_type_404 = torch.ops.prims.convert_element_type.default(add_118, torch.bfloat16);  add_118 = None
        _flash_attn_forward_18 = torch.ops.flash_attn._flash_attn_forward.default(convert_element_type_403, convert_element_type_404, slice_243, 0.0, 0.125, False, -1, -1, 0.0, None, False)
        getitem_108 = _flash_attn_forward_18[0]
        getitem_109 = _flash_attn_forward_18[1]
        getitem_111 = _flash_attn_forward_18[3];  _flash_attn_forward_18 = None
        view_189 = torch.ops.aten.view.default(getitem_108, [128, 901, 128])
        view_190 = torch.ops.aten.view.default(view_189, [115328, 128]);  view_189 = None
        mm_73 = torch.ops.aten.mm.default(view_190, permute_9);  view_190 = None
        view_191 = torch.ops.aten.view.default(mm_73, [128, 901, 128]);  mm_73 = None
        add_119 = torch.ops.aten.add.Tensor(convert_element_type_397, view_191);  convert_element_type_397 = view_191 = None
        convert_element_type_408 = torch.ops.prims.convert_element_type.default(add_119, torch.float32)
        pow_37 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_408, 2)
        mean_36 = torch.ops.aten.mean.dim(pow_37, [-1], True);  pow_37 = None
        add_120 = torch.ops.aten.add.Tensor(mean_36, 1e-05);  mean_36 = None
        rsqrt_36 = torch.ops.aten.rsqrt.default(add_120);  add_120 = None
        mul_149 = torch.ops.aten.mul.Tensor(convert_element_type_408, rsqrt_36);  convert_element_type_408 = None
        convert_element_type_409 = torch.ops.prims.convert_element_type.default(mul_149, torch.bfloat16);  mul_149 = None
        view_192 = torch.ops.aten.view.default(convert_element_type_409, [115328, 128])
        mm_74 = torch.ops.aten.mm.default(view_192, permute_10)
        view_193 = torch.ops.aten.view.default(mm_74, [128, 901, 1024]);  mm_74 = None
        split_18 = torch.ops.aten.split.Tensor(view_193, 512, -1);  view_193 = None
        getitem_112 = split_18[0]
        getitem_113 = split_18[1];  split_18 = None
        convert_element_type_413 = torch.ops.prims.convert_element_type.default(getitem_112, torch.float32)
        sigmoid_18 = torch.ops.aten.sigmoid.default(convert_element_type_413)
        mul_150 = torch.ops.aten.mul.Tensor(convert_element_type_413, sigmoid_18);  convert_element_type_413 = sigmoid_18 = None
        convert_element_type_414 = torch.ops.prims.convert_element_type.default(mul_150, torch.bfloat16);  mul_150 = None
        mul_151 = torch.ops.aten.mul.Tensor(convert_element_type_414, getitem_113);  convert_element_type_414 = None
        view_194 = torch.ops.aten.view.default(mul_151, [115328, 512]);  mul_151 = None
        mm_75 = torch.ops.aten.mm.default(view_194, permute_11)
        view_195 = torch.ops.aten.view.default(mm_75, [128, 901, 128]);  mm_75 = None
        add_121 = torch.ops.aten.add.Tensor(convert_element_type_409, view_195);  convert_element_type_409 = view_195 = None
        convert_element_type_418 = torch.ops.prims.convert_element_type.default(add_121, torch.float32)
        pow_38 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_418, 2)
        mean_37 = torch.ops.aten.mean.dim(pow_38, [-1], True);  pow_38 = None
        add_122 = torch.ops.aten.add.Tensor(mean_37, 1e-05);  mean_37 = None
        rsqrt_37 = torch.ops.aten.rsqrt.default(add_122);  add_122 = None
        mul_152 = torch.ops.aten.mul.Tensor(convert_element_type_418, rsqrt_37);  convert_element_type_418 = None
        convert_element_type_419 = torch.ops.prims.convert_element_type.default(mul_152, torch.bfloat16);  mul_152 = None
        view_196 = torch.ops.aten.view.default(convert_element_type_419, [115328, 128])
        mm_76 = torch.ops.aten.mm.default(view_196, permute_12)
        view_197 = torch.ops.aten.view.default(mm_76, [128, 901, 384]);  mm_76 = None
        view_198 = torch.ops.aten.view.default(view_197, [128, 901, 6, 64]);  view_197 = None
        slice_250 = torch.ops.aten.slice.Tensor(view_198, 2, 0, 2)
        slice_253 = torch.ops.aten.slice.Tensor(view_198, 2, 2, 4)
        slice_256 = torch.ops.aten.slice.Tensor(view_198, 2, 4, 9223372036854775807);  view_198 = None
        convert_element_type_423 = torch.ops.prims.convert_element_type.default(slice_250, torch.float32);  slice_250 = None
        convert_element_type_424 = torch.ops.prims.convert_element_type.default(slice_253, torch.float32);  slice_253 = None
        mul_153 = torch.ops.aten.mul.Tensor(convert_element_type_423, unsqueeze)
        slice_257 = torch.ops.aten.slice.Tensor(convert_element_type_423, 3, 0, 32)
        slice_258 = torch.ops.aten.slice.Tensor(convert_element_type_423, 3, 32, 9223372036854775807);  convert_element_type_423 = None
        neg_38 = torch.ops.aten.neg.default(slice_258);  slice_258 = None
        cat_39 = torch.ops.aten.cat.default([neg_38, slice_257], -1);  neg_38 = slice_257 = None
        mul_154 = torch.ops.aten.mul.Tensor(cat_39, unsqueeze_1);  cat_39 = None
        add_123 = torch.ops.aten.add.Tensor(mul_153, mul_154);  mul_153 = mul_154 = None
        mul_155 = torch.ops.aten.mul.Tensor(convert_element_type_424, unsqueeze)
        slice_259 = torch.ops.aten.slice.Tensor(convert_element_type_424, 3, 0, 32)
        slice_260 = torch.ops.aten.slice.Tensor(convert_element_type_424, 3, 32, 9223372036854775807);  convert_element_type_424 = None
        neg_39 = torch.ops.aten.neg.default(slice_260);  slice_260 = None
        cat_40 = torch.ops.aten.cat.default([neg_39, slice_259], -1);  neg_39 = slice_259 = None
        mul_156 = torch.ops.aten.mul.Tensor(cat_40, unsqueeze_1);  cat_40 = None
        add_124 = torch.ops.aten.add.Tensor(mul_155, mul_156);  mul_155 = mul_156 = None
        convert_element_type_425 = torch.ops.prims.convert_element_type.default(add_123, torch.bfloat16);  add_123 = None
        convert_element_type_426 = torch.ops.prims.convert_element_type.default(add_124, torch.bfloat16);  add_124 = None
        _flash_attn_forward_19 = torch.ops.flash_attn._flash_attn_forward.default(convert_element_type_425, convert_element_type_426, slice_256, 0.0, 0.125, False, -1, -1, 0.0, None, False)
        getitem_114 = _flash_attn_forward_19[0]
        getitem_115 = _flash_attn_forward_19[1]
        getitem_117 = _flash_attn_forward_19[3];  _flash_attn_forward_19 = None
        view_199 = torch.ops.aten.view.default(getitem_114, [128, 901, 128])
        view_200 = torch.ops.aten.view.default(view_199, [115328, 128]);  view_199 = None
        mm_77 = torch.ops.aten.mm.default(view_200, permute_13);  view_200 = None
        view_201 = torch.ops.aten.view.default(mm_77, [128, 901, 128]);  mm_77 = None
        add_125 = torch.ops.aten.add.Tensor(convert_element_type_419, view_201);  convert_element_type_419 = view_201 = None
        convert_element_type_430 = torch.ops.prims.convert_element_type.default(add_125, torch.float32)
        pow_39 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_430, 2)
        mean_38 = torch.ops.aten.mean.dim(pow_39, [-1], True);  pow_39 = None
        add_126 = torch.ops.aten.add.Tensor(mean_38, 1e-05);  mean_38 = None
        rsqrt_38 = torch.ops.aten.rsqrt.default(add_126);  add_126 = None
        mul_157 = torch.ops.aten.mul.Tensor(convert_element_type_430, rsqrt_38);  convert_element_type_430 = None
        convert_element_type_431 = torch.ops.prims.convert_element_type.default(mul_157, torch.bfloat16);  mul_157 = None
        view_202 = torch.ops.aten.view.default(convert_element_type_431, [115328, 128])
        mm_78 = torch.ops.aten.mm.default(view_202, permute_14)
        view_203 = torch.ops.aten.view.default(mm_78, [128, 901, 1024]);  mm_78 = None
        split_19 = torch.ops.aten.split.Tensor(view_203, 512, -1);  view_203 = None
        getitem_118 = split_19[0]
        getitem_119 = split_19[1];  split_19 = None
        convert_element_type_435 = torch.ops.prims.convert_element_type.default(getitem_118, torch.float32)
        sigmoid_19 = torch.ops.aten.sigmoid.default(convert_element_type_435)
        mul_158 = torch.ops.aten.mul.Tensor(convert_element_type_435, sigmoid_19);  convert_element_type_435 = sigmoid_19 = None
        convert_element_type_436 = torch.ops.prims.convert_element_type.default(mul_158, torch.bfloat16);  mul_158 = None
        mul_159 = torch.ops.aten.mul.Tensor(convert_element_type_436, getitem_119);  convert_element_type_436 = None
        view_204 = torch.ops.aten.view.default(mul_159, [115328, 512]);  mul_159 = None
        mm_79 = torch.ops.aten.mm.default(view_204, permute_15)
        view_205 = torch.ops.aten.view.default(mm_79, [128, 901, 128]);  mm_79 = None
        add_127 = torch.ops.aten.add.Tensor(convert_element_type_431, view_205);  convert_element_type_431 = view_205 = None
        convert_element_type_440 = torch.ops.prims.convert_element_type.default(add_127, torch.float32)
        pow_40 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_440, 2)
        mean_39 = torch.ops.aten.mean.dim(pow_40, [-1], True);  pow_40 = None
        add_128 = torch.ops.aten.add.Tensor(mean_39, 1e-05);  mean_39 = None
        rsqrt_39 = torch.ops.aten.rsqrt.default(add_128);  add_128 = None
        mul_160 = torch.ops.aten.mul.Tensor(convert_element_type_440, rsqrt_39);  convert_element_type_440 = None
        convert_element_type_441 = torch.ops.prims.convert_element_type.default(mul_160, torch.bfloat16);  mul_160 = None
        add_129 = torch.ops.aten.add.Tensor(convert_element_type_265, convert_element_type_441);  convert_element_type_265 = None
        view_206 = torch.ops.aten.view.default(add_129, [115328, 128])
        mm_80 = torch.ops.aten.mm.default(view_206, permute_32)
        view_207 = torch.ops.aten.view.default(mm_80, [128, 901, 384]);  mm_80 = None
        view_208 = torch.ops.aten.view.default(view_207, [128, 901, 6, 64]);  view_207 = None
        slice_263 = torch.ops.aten.slice.Tensor(view_208, 2, 0, 2)
        slice_266 = torch.ops.aten.slice.Tensor(view_208, 2, 2, 4)
        slice_269 = torch.ops.aten.slice.Tensor(view_208, 2, 4, 9223372036854775807);  view_208 = None
        convert_element_type_445 = torch.ops.prims.convert_element_type.default(slice_263, torch.float32);  slice_263 = None
        convert_element_type_446 = torch.ops.prims.convert_element_type.default(slice_266, torch.float32);  slice_266 = None
        mul_161 = torch.ops.aten.mul.Tensor(convert_element_type_445, unsqueeze)
        slice_270 = torch.ops.aten.slice.Tensor(convert_element_type_445, 3, 0, 32)
        slice_271 = torch.ops.aten.slice.Tensor(convert_element_type_445, 3, 32, 9223372036854775807);  convert_element_type_445 = None
        neg_40 = torch.ops.aten.neg.default(slice_271);  slice_271 = None
        cat_41 = torch.ops.aten.cat.default([neg_40, slice_270], -1);  neg_40 = slice_270 = None
        mul_162 = torch.ops.aten.mul.Tensor(cat_41, unsqueeze_1);  cat_41 = None
        add_130 = torch.ops.aten.add.Tensor(mul_161, mul_162);  mul_161 = mul_162 = None
        mul_163 = torch.ops.aten.mul.Tensor(convert_element_type_446, unsqueeze)
        slice_272 = torch.ops.aten.slice.Tensor(convert_element_type_446, 3, 0, 32)
        slice_273 = torch.ops.aten.slice.Tensor(convert_element_type_446, 3, 32, 9223372036854775807);  convert_element_type_446 = None
        neg_41 = torch.ops.aten.neg.default(slice_273);  slice_273 = None
        cat_42 = torch.ops.aten.cat.default([neg_41, slice_272], -1);  neg_41 = slice_272 = None
        mul_164 = torch.ops.aten.mul.Tensor(cat_42, unsqueeze_1);  cat_42 = None
        add_131 = torch.ops.aten.add.Tensor(mul_163, mul_164);  mul_163 = mul_164 = None
        convert_element_type_447 = torch.ops.prims.convert_element_type.default(add_130, torch.bfloat16);  add_130 = None
        convert_element_type_448 = torch.ops.prims.convert_element_type.default(add_131, torch.bfloat16);  add_131 = None
        _flash_attn_forward_20 = torch.ops.flash_attn._flash_attn_forward.default(convert_element_type_447, convert_element_type_448, slice_269, 0.0, 0.125, False, -1, -1, 0.0, None, False)
        getitem_120 = _flash_attn_forward_20[0]
        getitem_121 = _flash_attn_forward_20[1]
        getitem_123 = _flash_attn_forward_20[3];  _flash_attn_forward_20 = None
        view_209 = torch.ops.aten.view.default(getitem_120, [128, 901, 128])
        view_210 = torch.ops.aten.view.default(view_209, [115328, 128]);  view_209 = None
        mm_81 = torch.ops.aten.mm.default(view_210, permute_33);  view_210 = None
        view_211 = torch.ops.aten.view.default(mm_81, [128, 901, 128]);  mm_81 = None
        add_132 = torch.ops.aten.add.Tensor(add_129, view_211);  add_129 = view_211 = None
        convert_element_type_452 = torch.ops.prims.convert_element_type.default(add_132, torch.float32)
        pow_41 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_452, 2)
        mean_40 = torch.ops.aten.mean.dim(pow_41, [-1], True);  pow_41 = None
        add_133 = torch.ops.aten.add.Tensor(mean_40, 1e-05);  mean_40 = None
        rsqrt_40 = torch.ops.aten.rsqrt.default(add_133);  add_133 = None
        mul_165 = torch.ops.aten.mul.Tensor(convert_element_type_452, rsqrt_40);  convert_element_type_452 = None
        convert_element_type_453 = torch.ops.prims.convert_element_type.default(mul_165, torch.bfloat16);  mul_165 = None
        view_212 = torch.ops.aten.view.default(convert_element_type_453, [115328, 128])
        mm_82 = torch.ops.aten.mm.default(view_212, permute_34)
        view_213 = torch.ops.aten.view.default(mm_82, [128, 901, 1024]);  mm_82 = None
        split_20 = torch.ops.aten.split.Tensor(view_213, 512, -1);  view_213 = None
        getitem_124 = split_20[0]
        getitem_125 = split_20[1];  split_20 = None
        convert_element_type_457 = torch.ops.prims.convert_element_type.default(getitem_124, torch.float32)
        sigmoid_20 = torch.ops.aten.sigmoid.default(convert_element_type_457)
        mul_166 = torch.ops.aten.mul.Tensor(convert_element_type_457, sigmoid_20);  convert_element_type_457 = sigmoid_20 = None
        convert_element_type_458 = torch.ops.prims.convert_element_type.default(mul_166, torch.bfloat16);  mul_166 = None
        mul_167 = torch.ops.aten.mul.Tensor(convert_element_type_458, getitem_125);  convert_element_type_458 = None
        view_214 = torch.ops.aten.view.default(mul_167, [115328, 512]);  mul_167 = None
        mm_83 = torch.ops.aten.mm.default(view_214, permute_35)
        view_215 = torch.ops.aten.view.default(mm_83, [128, 901, 128]);  mm_83 = None
        add_134 = torch.ops.aten.add.Tensor(convert_element_type_453, view_215);  convert_element_type_453 = view_215 = None
        convert_element_type_462 = torch.ops.prims.convert_element_type.default(add_134, torch.float32)
        pow_42 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_462, 2)
        mean_41 = torch.ops.aten.mean.dim(pow_42, [-1], True);  pow_42 = None
        add_135 = torch.ops.aten.add.Tensor(mean_41, 1e-05);  mean_41 = None
        rsqrt_41 = torch.ops.aten.rsqrt.default(add_135);  add_135 = None
        mul_168 = torch.ops.aten.mul.Tensor(convert_element_type_462, rsqrt_41);  convert_element_type_462 = None
        convert_element_type_463 = torch.ops.prims.convert_element_type.default(mul_168, torch.bfloat16);  mul_168 = None
        view_216 = torch.ops.aten.view.default(convert_element_type_463, [115328, 128])
        mm_84 = torch.ops.aten.mm.default(view_216, permute_36)
        view_217 = torch.ops.aten.view.default(mm_84, [128, 901, 384]);  mm_84 = None
        view_218 = torch.ops.aten.view.default(view_217, [128, 901, 6, 64]);  view_217 = None
        slice_276 = torch.ops.aten.slice.Tensor(view_218, 2, 0, 2)
        slice_279 = torch.ops.aten.slice.Tensor(view_218, 2, 2, 4)
        slice_282 = torch.ops.aten.slice.Tensor(view_218, 2, 4, 9223372036854775807);  view_218 = None
        convert_element_type_467 = torch.ops.prims.convert_element_type.default(slice_276, torch.float32);  slice_276 = None
        convert_element_type_468 = torch.ops.prims.convert_element_type.default(slice_279, torch.float32);  slice_279 = None
        mul_169 = torch.ops.aten.mul.Tensor(convert_element_type_467, unsqueeze)
        slice_283 = torch.ops.aten.slice.Tensor(convert_element_type_467, 3, 0, 32)
        slice_284 = torch.ops.aten.slice.Tensor(convert_element_type_467, 3, 32, 9223372036854775807);  convert_element_type_467 = None
        neg_42 = torch.ops.aten.neg.default(slice_284);  slice_284 = None
        cat_43 = torch.ops.aten.cat.default([neg_42, slice_283], -1);  neg_42 = slice_283 = None
        mul_170 = torch.ops.aten.mul.Tensor(cat_43, unsqueeze_1);  cat_43 = None
        add_136 = torch.ops.aten.add.Tensor(mul_169, mul_170);  mul_169 = mul_170 = None
        mul_171 = torch.ops.aten.mul.Tensor(convert_element_type_468, unsqueeze)
        slice_285 = torch.ops.aten.slice.Tensor(convert_element_type_468, 3, 0, 32)
        slice_286 = torch.ops.aten.slice.Tensor(convert_element_type_468, 3, 32, 9223372036854775807);  convert_element_type_468 = None
        neg_43 = torch.ops.aten.neg.default(slice_286);  slice_286 = None
        cat_44 = torch.ops.aten.cat.default([neg_43, slice_285], -1);  neg_43 = slice_285 = None
        mul_172 = torch.ops.aten.mul.Tensor(cat_44, unsqueeze_1);  cat_44 = None
        add_137 = torch.ops.aten.add.Tensor(mul_171, mul_172);  mul_171 = mul_172 = None
        convert_element_type_469 = torch.ops.prims.convert_element_type.default(add_136, torch.bfloat16);  add_136 = None
        convert_element_type_470 = torch.ops.prims.convert_element_type.default(add_137, torch.bfloat16);  add_137 = None
        _flash_attn_forward_21 = torch.ops.flash_attn._flash_attn_forward.default(convert_element_type_469, convert_element_type_470, slice_282, 0.0, 0.125, False, -1, -1, 0.0, None, False)
        getitem_126 = _flash_attn_forward_21[0]
        getitem_127 = _flash_attn_forward_21[1]
        getitem_129 = _flash_attn_forward_21[3];  _flash_attn_forward_21 = None
        view_219 = torch.ops.aten.view.default(getitem_126, [128, 901, 128])
        view_220 = torch.ops.aten.view.default(view_219, [115328, 128]);  view_219 = None
        mm_85 = torch.ops.aten.mm.default(view_220, permute_37);  view_220 = None
        view_221 = torch.ops.aten.view.default(mm_85, [128, 901, 128]);  mm_85 = None
        add_138 = torch.ops.aten.add.Tensor(convert_element_type_463, view_221);  convert_element_type_463 = view_221 = None
        convert_element_type_474 = torch.ops.prims.convert_element_type.default(add_138, torch.float32)
        pow_43 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_474, 2)
        mean_42 = torch.ops.aten.mean.dim(pow_43, [-1], True);  pow_43 = None
        add_139 = torch.ops.aten.add.Tensor(mean_42, 1e-05);  mean_42 = None
        rsqrt_42 = torch.ops.aten.rsqrt.default(add_139);  add_139 = None
        mul_173 = torch.ops.aten.mul.Tensor(convert_element_type_474, rsqrt_42);  convert_element_type_474 = None
        convert_element_type_475 = torch.ops.prims.convert_element_type.default(mul_173, torch.bfloat16);  mul_173 = None
        view_222 = torch.ops.aten.view.default(convert_element_type_475, [115328, 128])
        mm_86 = torch.ops.aten.mm.default(view_222, permute_38)
        view_223 = torch.ops.aten.view.default(mm_86, [128, 901, 1024]);  mm_86 = None
        split_21 = torch.ops.aten.split.Tensor(view_223, 512, -1);  view_223 = None
        getitem_130 = split_21[0]
        getitem_131 = split_21[1];  split_21 = None
        convert_element_type_479 = torch.ops.prims.convert_element_type.default(getitem_130, torch.float32)
        sigmoid_21 = torch.ops.aten.sigmoid.default(convert_element_type_479)
        mul_174 = torch.ops.aten.mul.Tensor(convert_element_type_479, sigmoid_21);  convert_element_type_479 = sigmoid_21 = None
        convert_element_type_480 = torch.ops.prims.convert_element_type.default(mul_174, torch.bfloat16);  mul_174 = None
        mul_175 = torch.ops.aten.mul.Tensor(convert_element_type_480, getitem_131);  convert_element_type_480 = None
        view_224 = torch.ops.aten.view.default(mul_175, [115328, 512]);  mul_175 = None
        mm_87 = torch.ops.aten.mm.default(view_224, permute_39)
        view_225 = torch.ops.aten.view.default(mm_87, [128, 901, 128]);  mm_87 = None
        add_140 = torch.ops.aten.add.Tensor(convert_element_type_475, view_225);  convert_element_type_475 = view_225 = None
        convert_element_type_484 = torch.ops.prims.convert_element_type.default(add_140, torch.float32)
        pow_44 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_484, 2)
        mean_43 = torch.ops.aten.mean.dim(pow_44, [-1], True);  pow_44 = None
        add_141 = torch.ops.aten.add.Tensor(mean_43, 1e-05);  mean_43 = None
        rsqrt_43 = torch.ops.aten.rsqrt.default(add_141);  add_141 = None
        mul_176 = torch.ops.aten.mul.Tensor(convert_element_type_484, rsqrt_43);  convert_element_type_484 = None
        convert_element_type_485 = torch.ops.prims.convert_element_type.default(mul_176, torch.bfloat16);  mul_176 = None
        view_226 = torch.ops.aten.view.default(convert_element_type_485, [115328, 128])
        mm_88 = torch.ops.aten.mm.default(view_226, permute_40)
        view_227 = torch.ops.aten.view.default(mm_88, [128, 901, 384]);  mm_88 = None
        view_228 = torch.ops.aten.view.default(view_227, [128, 901, 6, 64]);  view_227 = None
        slice_289 = torch.ops.aten.slice.Tensor(view_228, 2, 0, 2)
        slice_292 = torch.ops.aten.slice.Tensor(view_228, 2, 2, 4)
        slice_295 = torch.ops.aten.slice.Tensor(view_228, 2, 4, 9223372036854775807);  view_228 = None
        convert_element_type_489 = torch.ops.prims.convert_element_type.default(slice_289, torch.float32);  slice_289 = None
        convert_element_type_490 = torch.ops.prims.convert_element_type.default(slice_292, torch.float32);  slice_292 = None
        mul_177 = torch.ops.aten.mul.Tensor(convert_element_type_489, unsqueeze)
        slice_296 = torch.ops.aten.slice.Tensor(convert_element_type_489, 3, 0, 32)
        slice_297 = torch.ops.aten.slice.Tensor(convert_element_type_489, 3, 32, 9223372036854775807);  convert_element_type_489 = None
        neg_44 = torch.ops.aten.neg.default(slice_297);  slice_297 = None
        cat_45 = torch.ops.aten.cat.default([neg_44, slice_296], -1);  neg_44 = slice_296 = None
        mul_178 = torch.ops.aten.mul.Tensor(cat_45, unsqueeze_1);  cat_45 = None
        add_142 = torch.ops.aten.add.Tensor(mul_177, mul_178);  mul_177 = mul_178 = None
        mul_179 = torch.ops.aten.mul.Tensor(convert_element_type_490, unsqueeze)
        slice_298 = torch.ops.aten.slice.Tensor(convert_element_type_490, 3, 0, 32)
        slice_299 = torch.ops.aten.slice.Tensor(convert_element_type_490, 3, 32, 9223372036854775807);  convert_element_type_490 = None
        neg_45 = torch.ops.aten.neg.default(slice_299);  slice_299 = None
        cat_46 = torch.ops.aten.cat.default([neg_45, slice_298], -1);  neg_45 = slice_298 = None
        mul_180 = torch.ops.aten.mul.Tensor(cat_46, unsqueeze_1);  cat_46 = None
        add_143 = torch.ops.aten.add.Tensor(mul_179, mul_180);  mul_179 = mul_180 = None
        convert_element_type_491 = torch.ops.prims.convert_element_type.default(add_142, torch.bfloat16);  add_142 = None
        convert_element_type_492 = torch.ops.prims.convert_element_type.default(add_143, torch.bfloat16);  add_143 = None
        _flash_attn_forward_22 = torch.ops.flash_attn._flash_attn_forward.default(convert_element_type_491, convert_element_type_492, slice_295, 0.0, 0.125, False, -1, -1, 0.0, None, False)
        getitem_132 = _flash_attn_forward_22[0]
        getitem_133 = _flash_attn_forward_22[1]
        getitem_135 = _flash_attn_forward_22[3];  _flash_attn_forward_22 = None
        view_229 = torch.ops.aten.view.default(getitem_132, [128, 901, 128])
        view_230 = torch.ops.aten.view.default(view_229, [115328, 128]);  view_229 = None
        mm_89 = torch.ops.aten.mm.default(view_230, permute_41);  view_230 = None
        view_231 = torch.ops.aten.view.default(mm_89, [128, 901, 128]);  mm_89 = None
        add_144 = torch.ops.aten.add.Tensor(convert_element_type_485, view_231);  convert_element_type_485 = view_231 = None
        convert_element_type_496 = torch.ops.prims.convert_element_type.default(add_144, torch.float32)
        pow_45 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_496, 2)
        mean_44 = torch.ops.aten.mean.dim(pow_45, [-1], True);  pow_45 = None
        add_145 = torch.ops.aten.add.Tensor(mean_44, 1e-05);  mean_44 = None
        rsqrt_44 = torch.ops.aten.rsqrt.default(add_145);  add_145 = None
        mul_181 = torch.ops.aten.mul.Tensor(convert_element_type_496, rsqrt_44);  convert_element_type_496 = None
        convert_element_type_497 = torch.ops.prims.convert_element_type.default(mul_181, torch.bfloat16);  mul_181 = None
        view_232 = torch.ops.aten.view.default(convert_element_type_497, [115328, 128])
        mm_90 = torch.ops.aten.mm.default(view_232, permute_42)
        view_233 = torch.ops.aten.view.default(mm_90, [128, 901, 1024]);  mm_90 = None
        split_22 = torch.ops.aten.split.Tensor(view_233, 512, -1);  view_233 = None
        getitem_136 = split_22[0]
        getitem_137 = split_22[1];  split_22 = None
        convert_element_type_501 = torch.ops.prims.convert_element_type.default(getitem_136, torch.float32)
        sigmoid_22 = torch.ops.aten.sigmoid.default(convert_element_type_501)
        mul_182 = torch.ops.aten.mul.Tensor(convert_element_type_501, sigmoid_22);  convert_element_type_501 = sigmoid_22 = None
        convert_element_type_502 = torch.ops.prims.convert_element_type.default(mul_182, torch.bfloat16);  mul_182 = None
        mul_183 = torch.ops.aten.mul.Tensor(convert_element_type_502, getitem_137);  convert_element_type_502 = None
        view_234 = torch.ops.aten.view.default(mul_183, [115328, 512]);  mul_183 = None
        mm_91 = torch.ops.aten.mm.default(view_234, permute_43)
        view_235 = torch.ops.aten.view.default(mm_91, [128, 901, 128]);  mm_91 = None
        add_146 = torch.ops.aten.add.Tensor(convert_element_type_497, view_235);  convert_element_type_497 = view_235 = None
        convert_element_type_506 = torch.ops.prims.convert_element_type.default(add_146, torch.float32)
        pow_46 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_506, 2)
        mean_45 = torch.ops.aten.mean.dim(pow_46, [-1], True);  pow_46 = None
        add_147 = torch.ops.aten.add.Tensor(mean_45, 1e-05);  mean_45 = None
        rsqrt_45 = torch.ops.aten.rsqrt.default(add_147);  add_147 = None
        mul_184 = torch.ops.aten.mul.Tensor(convert_element_type_506, rsqrt_45);  convert_element_type_506 = None
        convert_element_type_507 = torch.ops.prims.convert_element_type.default(mul_184, torch.bfloat16);  mul_184 = None
        view_236 = torch.ops.aten.view.default(convert_element_type_507, [115328, 128])
        mm_92 = torch.ops.aten.mm.default(view_236, permute_44)
        view_237 = torch.ops.aten.view.default(mm_92, [128, 901, 384]);  mm_92 = None
        view_238 = torch.ops.aten.view.default(view_237, [128, 901, 6, 64]);  view_237 = None
        slice_302 = torch.ops.aten.slice.Tensor(view_238, 2, 0, 2)
        slice_305 = torch.ops.aten.slice.Tensor(view_238, 2, 2, 4)
        slice_308 = torch.ops.aten.slice.Tensor(view_238, 2, 4, 9223372036854775807);  view_238 = None
        convert_element_type_511 = torch.ops.prims.convert_element_type.default(slice_302, torch.float32);  slice_302 = None
        convert_element_type_512 = torch.ops.prims.convert_element_type.default(slice_305, torch.float32);  slice_305 = None
        mul_185 = torch.ops.aten.mul.Tensor(convert_element_type_511, unsqueeze)
        slice_309 = torch.ops.aten.slice.Tensor(convert_element_type_511, 3, 0, 32)
        slice_310 = torch.ops.aten.slice.Tensor(convert_element_type_511, 3, 32, 9223372036854775807);  convert_element_type_511 = None
        neg_46 = torch.ops.aten.neg.default(slice_310);  slice_310 = None
        cat_47 = torch.ops.aten.cat.default([neg_46, slice_309], -1);  neg_46 = slice_309 = None
        mul_186 = torch.ops.aten.mul.Tensor(cat_47, unsqueeze_1);  cat_47 = None
        add_148 = torch.ops.aten.add.Tensor(mul_185, mul_186);  mul_185 = mul_186 = None
        mul_187 = torch.ops.aten.mul.Tensor(convert_element_type_512, unsqueeze);  unsqueeze = None
        slice_311 = torch.ops.aten.slice.Tensor(convert_element_type_512, 3, 0, 32)
        slice_312 = torch.ops.aten.slice.Tensor(convert_element_type_512, 3, 32, 9223372036854775807);  convert_element_type_512 = None
        neg_47 = torch.ops.aten.neg.default(slice_312);  slice_312 = None
        cat_48 = torch.ops.aten.cat.default([neg_47, slice_311], -1);  neg_47 = slice_311 = None
        mul_188 = torch.ops.aten.mul.Tensor(cat_48, unsqueeze_1);  cat_48 = unsqueeze_1 = None
        add_149 = torch.ops.aten.add.Tensor(mul_187, mul_188);  mul_187 = mul_188 = None
        convert_element_type_513 = torch.ops.prims.convert_element_type.default(add_148, torch.bfloat16);  add_148 = None
        convert_element_type_514 = torch.ops.prims.convert_element_type.default(add_149, torch.bfloat16);  add_149 = None
        _flash_attn_forward_23 = torch.ops.flash_attn._flash_attn_forward.default(convert_element_type_513, convert_element_type_514, slice_308, 0.0, 0.125, False, -1, -1, 0.0, None, False)
        getitem_138 = _flash_attn_forward_23[0]
        getitem_139 = _flash_attn_forward_23[1]
        getitem_141 = _flash_attn_forward_23[3];  _flash_attn_forward_23 = None
        view_239 = torch.ops.aten.view.default(getitem_138, [128, 901, 128])
        view_240 = torch.ops.aten.view.default(view_239, [115328, 128]);  view_239 = None
        mm_93 = torch.ops.aten.mm.default(view_240, permute_45);  view_240 = None
        view_241 = torch.ops.aten.view.default(mm_93, [128, 901, 128]);  mm_93 = None
        add_150 = torch.ops.aten.add.Tensor(convert_element_type_507, view_241);  convert_element_type_507 = view_241 = None
        convert_element_type_518 = torch.ops.prims.convert_element_type.default(add_150, torch.float32)
        pow_47 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_518, 2)
        mean_46 = torch.ops.aten.mean.dim(pow_47, [-1], True);  pow_47 = None
        add_151 = torch.ops.aten.add.Tensor(mean_46, 1e-05);  mean_46 = None
        rsqrt_46 = torch.ops.aten.rsqrt.default(add_151);  add_151 = None
        mul_189 = torch.ops.aten.mul.Tensor(convert_element_type_518, rsqrt_46);  convert_element_type_518 = None
        convert_element_type_519 = torch.ops.prims.convert_element_type.default(mul_189, torch.bfloat16);  mul_189 = None
        view_242 = torch.ops.aten.view.default(convert_element_type_519, [115328, 128])
        mm_94 = torch.ops.aten.mm.default(view_242, permute_46)
        view_243 = torch.ops.aten.view.default(mm_94, [128, 901, 1024]);  mm_94 = None
        split_23 = torch.ops.aten.split.Tensor(view_243, 512, -1);  view_243 = None
        getitem_142 = split_23[0]
        getitem_143 = split_23[1];  split_23 = None
        convert_element_type_523 = torch.ops.prims.convert_element_type.default(getitem_142, torch.float32)
        sigmoid_23 = torch.ops.aten.sigmoid.default(convert_element_type_523)
        mul_190 = torch.ops.aten.mul.Tensor(convert_element_type_523, sigmoid_23);  convert_element_type_523 = sigmoid_23 = None
        convert_element_type_524 = torch.ops.prims.convert_element_type.default(mul_190, torch.bfloat16);  mul_190 = None
        mul_191 = torch.ops.aten.mul.Tensor(convert_element_type_524, getitem_143);  convert_element_type_524 = None
        view_244 = torch.ops.aten.view.default(mul_191, [115328, 512]);  mul_191 = None
        mm_95 = torch.ops.aten.mm.default(view_244, permute_47)
        view_245 = torch.ops.aten.view.default(mm_95, [128, 901, 128]);  mm_95 = None
        add_152 = torch.ops.aten.add.Tensor(convert_element_type_519, view_245);  convert_element_type_519 = view_245 = None
        convert_element_type_528 = torch.ops.prims.convert_element_type.default(add_152, torch.float32)
        pow_48 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_528, 2)
        mean_47 = torch.ops.aten.mean.dim(pow_48, [-1], True);  pow_48 = None
        add_153 = torch.ops.aten.add.Tensor(mean_47, 1e-05);  mean_47 = None
        rsqrt_47 = torch.ops.aten.rsqrt.default(add_153);  add_153 = None
        mul_192 = torch.ops.aten.mul.Tensor(convert_element_type_528, rsqrt_47);  convert_element_type_528 = None
        convert_element_type_529 = torch.ops.prims.convert_element_type.default(mul_192, torch.bfloat16);  mul_192 = None
        convert_element_type_530 = torch.ops.prims.convert_element_type.default(primals_51, torch.bfloat16);  primals_51 = None
        permute_96 = torch.ops.aten.permute.default(convert_element_type_530, [1, 0]);  convert_element_type_530 = None
        view_246 = torch.ops.aten.view.default(convert_element_type_529, [115328, 128])
        mm_96 = torch.ops.aten.mm.default(view_246, permute_96)
        view_247 = torch.ops.aten.view.default(mm_96, [128, 901, 10]);  mm_96 = None
        slice_314 = torch.ops.aten.slice.Tensor(view_247, 1, 1, 9223372036854775807);  view_247 = None
        select = torch.ops.aten.select.int(convert_element_type_529, 1, 0)
        convert_element_type_533 = torch.ops.prims.convert_element_type.default(primals_52, torch.bfloat16)
        convert_element_type_534 = torch.ops.prims.convert_element_type.default(primals_53, torch.bfloat16)
        permute_97 = torch.ops.aten.permute.default(convert_element_type_533, [1, 0]);  convert_element_type_533 = None
        addmm = torch.ops.aten.addmm.default(convert_element_type_534, select, permute_97);  convert_element_type_534 = None
        convert_element_type_538 = torch.ops.prims.convert_element_type.default(addmm, torch.float32);  addmm = None
        select_1 = torch.ops.aten.select.int(convert_element_type_538, 1, 0)
        select_2 = torch.ops.aten.select.int(convert_element_type_538, 1, 1)
        add_154 = torch.ops.aten.add.Tensor(where_2, 1);  where_2 = None
        ge = torch.ops.aten.ge.Scalar(add_154, 16)
        gt = torch.ops.aten.gt.Tensor(select_1, select_2)
        bitwise_or = torch.ops.aten.bitwise_or.Tensor(ge, gt);  gt = None
        inductor_seeds_default = torch.ops.prims.inductor_seeds.default(2, device(type='cuda', index=0))
        inductor_lookup_seed_default = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0)
        inductor_random_default = torch.ops.prims.inductor_random.default([128], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        lt = torch.ops.aten.lt.Scalar(inductor_random_default, 0.1);  inductor_random_default = None
        inductor_lookup_seed_default_1 = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 1);  inductor_seeds_default = None
        inductor_randint_default = torch.ops.prims.inductor_randint.default(2, 17, [128], inductor_lookup_seed_default_1);  inductor_lookup_seed_default_1 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(inductor_randint_default, torch.int32);  inductor_randint_default = None
        mul_193 = torch.ops.aten.mul.Tensor(lt, convert_element_type_default_1);  lt = convert_element_type_default_1 = None
        ge_1 = torch.ops.aten.ge.Tensor(add_154, mul_193);  mul_193 = None
        bitwise_and = torch.ops.aten.bitwise_and.Tensor(bitwise_or, ge_1);  bitwise_or = ge_1 = None
        convert_element_type_539 = torch.ops.prims.convert_element_type.default(primals_15, torch.bfloat16);  primals_15 = None
        embedding_1 = torch.ops.aten.embedding.default(convert_element_type_539, where_3);  convert_element_type_539 = None
        index_1 = torch.ops.aten.index.Tensor(primals_17, [where_5]);  primals_17 = None
        convert_element_type_540 = torch.ops.prims.convert_element_type.default(index_1, torch.bfloat16)
        view_248 = torch.ops.aten.view.default(convert_element_type_540, [-1, 1, 128]);  convert_element_type_540 = None
        cat_49 = torch.ops.aten.cat.default([view_248, embedding_1], -2);  view_248 = embedding_1 = None
        mul_194 = torch.ops.aten.mul.Tensor(cat_49, 11.313708498984761);  cat_49 = None
        add_155 = torch.ops.aten.add.Tensor(convert_element_type_529, mul_194)
        add_156 = torch.ops.aten.add.Tensor(convert_element_type_441, add_155)
        convert_element_type_541 = torch.ops.prims.convert_element_type.default(primals_19, torch.bfloat16);  primals_19 = None
        permute_98 = torch.ops.aten.permute.default(convert_element_type_541, [1, 0]);  convert_element_type_541 = None
        view_249 = torch.ops.aten.view.default(add_156, [115328, 128])
        mm_97 = torch.ops.aten.mm.default(view_249, permute_98);  view_249 = None
        view_250 = torch.ops.aten.view.default(mm_97, [128, 901, 384]);  mm_97 = None
        view_251 = torch.ops.aten.view.default(view_250, [128, 901, 6, 64]);  view_250 = None
        slice_318 = torch.ops.aten.slice.Tensor(view_251, 2, 0, 2)
        slice_321 = torch.ops.aten.slice.Tensor(view_251, 2, 2, 4)
        slice_324 = torch.ops.aten.slice.Tensor(view_251, 2, 4, 9223372036854775807);  view_251 = None
        convert_element_type_544 = torch.ops.prims.convert_element_type.default(slice_318, torch.float32);  slice_318 = None
        convert_element_type_545 = torch.ops.prims.convert_element_type.default(slice_321, torch.float32);  slice_321 = None
        unsqueeze_96 = torch.ops.aten.unsqueeze.default(primals_13, -2)
        mul_195 = torch.ops.aten.mul.Tensor(convert_element_type_544, unsqueeze_96)
        slice_325 = torch.ops.aten.slice.Tensor(convert_element_type_544, 3, 0, 32)
        slice_326 = torch.ops.aten.slice.Tensor(convert_element_type_544, 3, 32, 9223372036854775807);  convert_element_type_544 = None
        neg_48 = torch.ops.aten.neg.default(slice_326);  slice_326 = None
        cat_50 = torch.ops.aten.cat.default([neg_48, slice_325], -1);  neg_48 = slice_325 = None
        unsqueeze_97 = torch.ops.aten.unsqueeze.default(primals_14, -2)
        mul_196 = torch.ops.aten.mul.Tensor(cat_50, unsqueeze_97);  cat_50 = None
        add_157 = torch.ops.aten.add.Tensor(mul_195, mul_196);  mul_195 = mul_196 = None
        mul_197 = torch.ops.aten.mul.Tensor(convert_element_type_545, unsqueeze_96)
        slice_327 = torch.ops.aten.slice.Tensor(convert_element_type_545, 3, 0, 32)
        slice_328 = torch.ops.aten.slice.Tensor(convert_element_type_545, 3, 32, 9223372036854775807);  convert_element_type_545 = None
        neg_49 = torch.ops.aten.neg.default(slice_328);  slice_328 = None
        cat_51 = torch.ops.aten.cat.default([neg_49, slice_327], -1);  neg_49 = slice_327 = None
        mul_198 = torch.ops.aten.mul.Tensor(cat_51, unsqueeze_97);  cat_51 = None
        add_158 = torch.ops.aten.add.Tensor(mul_197, mul_198);  mul_197 = mul_198 = None
        convert_element_type_546 = torch.ops.prims.convert_element_type.default(add_157, torch.bfloat16);  add_157 = None
        convert_element_type_547 = torch.ops.prims.convert_element_type.default(add_158, torch.bfloat16);  add_158 = None
        _flash_attn_forward_24 = torch.ops.flash_attn._flash_attn_forward.default(convert_element_type_546, convert_element_type_547, slice_324, 0.0, 0.125, False, -1, -1, 0.0, None, False);  convert_element_type_546 = convert_element_type_547 = slice_324 = None
        getitem_144 = _flash_attn_forward_24[0];  _flash_attn_forward_24 = None
        view_252 = torch.ops.aten.view.default(getitem_144, [128, 901, 128]);  getitem_144 = None
        convert_element_type_548 = torch.ops.prims.convert_element_type.default(primals_20, torch.bfloat16);  primals_20 = None
        permute_99 = torch.ops.aten.permute.default(convert_element_type_548, [1, 0]);  convert_element_type_548 = None
        view_253 = torch.ops.aten.view.default(view_252, [115328, 128]);  view_252 = None
        mm_98 = torch.ops.aten.mm.default(view_253, permute_99);  view_253 = None
        view_254 = torch.ops.aten.view.default(mm_98, [128, 901, 128]);  mm_98 = None
        add_159 = torch.ops.aten.add.Tensor(add_156, view_254);  add_156 = view_254 = None
        convert_element_type_551 = torch.ops.prims.convert_element_type.default(add_159, torch.float32);  add_159 = None
        pow_49 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_551, 2)
        mean_48 = torch.ops.aten.mean.dim(pow_49, [-1], True);  pow_49 = None
        add_160 = torch.ops.aten.add.Tensor(mean_48, 1e-05);  mean_48 = None
        rsqrt_48 = torch.ops.aten.rsqrt.default(add_160);  add_160 = None
        mul_199 = torch.ops.aten.mul.Tensor(convert_element_type_551, rsqrt_48);  convert_element_type_551 = rsqrt_48 = None
        convert_element_type_552 = torch.ops.prims.convert_element_type.default(mul_199, torch.bfloat16);  mul_199 = None
        convert_element_type_553 = torch.ops.prims.convert_element_type.default(primals_21, torch.bfloat16);  primals_21 = None
        permute_100 = torch.ops.aten.permute.default(convert_element_type_553, [1, 0]);  convert_element_type_553 = None
        view_255 = torch.ops.aten.view.default(convert_element_type_552, [115328, 128])
        mm_99 = torch.ops.aten.mm.default(view_255, permute_100);  view_255 = None
        view_256 = torch.ops.aten.view.default(mm_99, [128, 901, 1024]);  mm_99 = None
        split_24 = torch.ops.aten.split.Tensor(view_256, 512, -1);  view_256 = None
        getitem_148 = split_24[0]
        getitem_149 = split_24[1];  split_24 = None
        convert_element_type_556 = torch.ops.prims.convert_element_type.default(getitem_148, torch.float32);  getitem_148 = None
        sigmoid_24 = torch.ops.aten.sigmoid.default(convert_element_type_556)
        mul_200 = torch.ops.aten.mul.Tensor(convert_element_type_556, sigmoid_24);  convert_element_type_556 = sigmoid_24 = None
        convert_element_type_557 = torch.ops.prims.convert_element_type.default(mul_200, torch.bfloat16);  mul_200 = None
        mul_201 = torch.ops.aten.mul.Tensor(convert_element_type_557, getitem_149);  convert_element_type_557 = getitem_149 = None
        convert_element_type_558 = torch.ops.prims.convert_element_type.default(primals_22, torch.bfloat16);  primals_22 = None
        permute_101 = torch.ops.aten.permute.default(convert_element_type_558, [1, 0]);  convert_element_type_558 = None
        view_257 = torch.ops.aten.view.default(mul_201, [115328, 512]);  mul_201 = None
        mm_100 = torch.ops.aten.mm.default(view_257, permute_101);  view_257 = None
        view_258 = torch.ops.aten.view.default(mm_100, [128, 901, 128]);  mm_100 = None
        add_161 = torch.ops.aten.add.Tensor(convert_element_type_552, view_258);  convert_element_type_552 = view_258 = None
        convert_element_type_561 = torch.ops.prims.convert_element_type.default(add_161, torch.float32);  add_161 = None
        pow_50 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_561, 2)
        mean_49 = torch.ops.aten.mean.dim(pow_50, [-1], True);  pow_50 = None
        add_162 = torch.ops.aten.add.Tensor(mean_49, 1e-05);  mean_49 = None
        rsqrt_49 = torch.ops.aten.rsqrt.default(add_162);  add_162 = None
        mul_202 = torch.ops.aten.mul.Tensor(convert_element_type_561, rsqrt_49);  convert_element_type_561 = rsqrt_49 = None
        convert_element_type_562 = torch.ops.prims.convert_element_type.default(mul_202, torch.bfloat16);  mul_202 = None
        convert_element_type_563 = torch.ops.prims.convert_element_type.default(primals_23, torch.bfloat16);  primals_23 = None
        permute_102 = torch.ops.aten.permute.default(convert_element_type_563, [1, 0]);  convert_element_type_563 = None
        view_259 = torch.ops.aten.view.default(convert_element_type_562, [115328, 128])
        mm_101 = torch.ops.aten.mm.default(view_259, permute_102);  view_259 = None
        view_260 = torch.ops.aten.view.default(mm_101, [128, 901, 384]);  mm_101 = None
        view_261 = torch.ops.aten.view.default(view_260, [128, 901, 6, 64]);  view_260 = None
        slice_331 = torch.ops.aten.slice.Tensor(view_261, 2, 0, 2)
        slice_334 = torch.ops.aten.slice.Tensor(view_261, 2, 2, 4)
        slice_337 = torch.ops.aten.slice.Tensor(view_261, 2, 4, 9223372036854775807);  view_261 = None
        convert_element_type_566 = torch.ops.prims.convert_element_type.default(slice_331, torch.float32);  slice_331 = None
        convert_element_type_567 = torch.ops.prims.convert_element_type.default(slice_334, torch.float32);  slice_334 = None
        mul_203 = torch.ops.aten.mul.Tensor(convert_element_type_566, unsqueeze_96)
        slice_338 = torch.ops.aten.slice.Tensor(convert_element_type_566, 3, 0, 32)
        slice_339 = torch.ops.aten.slice.Tensor(convert_element_type_566, 3, 32, 9223372036854775807);  convert_element_type_566 = None
        neg_50 = torch.ops.aten.neg.default(slice_339);  slice_339 = None
        cat_52 = torch.ops.aten.cat.default([neg_50, slice_338], -1);  neg_50 = slice_338 = None
        mul_204 = torch.ops.aten.mul.Tensor(cat_52, unsqueeze_97);  cat_52 = None
        add_163 = torch.ops.aten.add.Tensor(mul_203, mul_204);  mul_203 = mul_204 = None
        mul_205 = torch.ops.aten.mul.Tensor(convert_element_type_567, unsqueeze_96)
        slice_340 = torch.ops.aten.slice.Tensor(convert_element_type_567, 3, 0, 32)
        slice_341 = torch.ops.aten.slice.Tensor(convert_element_type_567, 3, 32, 9223372036854775807);  convert_element_type_567 = None
        neg_51 = torch.ops.aten.neg.default(slice_341);  slice_341 = None
        cat_53 = torch.ops.aten.cat.default([neg_51, slice_340], -1);  neg_51 = slice_340 = None
        mul_206 = torch.ops.aten.mul.Tensor(cat_53, unsqueeze_97);  cat_53 = None
        add_164 = torch.ops.aten.add.Tensor(mul_205, mul_206);  mul_205 = mul_206 = None
        convert_element_type_568 = torch.ops.prims.convert_element_type.default(add_163, torch.bfloat16);  add_163 = None
        convert_element_type_569 = torch.ops.prims.convert_element_type.default(add_164, torch.bfloat16);  add_164 = None
        _flash_attn_forward_25 = torch.ops.flash_attn._flash_attn_forward.default(convert_element_type_568, convert_element_type_569, slice_337, 0.0, 0.125, False, -1, -1, 0.0, None, False);  convert_element_type_568 = convert_element_type_569 = slice_337 = None
        getitem_150 = _flash_attn_forward_25[0];  _flash_attn_forward_25 = None
        view_262 = torch.ops.aten.view.default(getitem_150, [128, 901, 128]);  getitem_150 = None
        convert_element_type_570 = torch.ops.prims.convert_element_type.default(primals_24, torch.bfloat16);  primals_24 = None
        permute_103 = torch.ops.aten.permute.default(convert_element_type_570, [1, 0]);  convert_element_type_570 = None
        view_263 = torch.ops.aten.view.default(view_262, [115328, 128]);  view_262 = None
        mm_102 = torch.ops.aten.mm.default(view_263, permute_103);  view_263 = None
        view_264 = torch.ops.aten.view.default(mm_102, [128, 901, 128]);  mm_102 = None
        add_165 = torch.ops.aten.add.Tensor(convert_element_type_562, view_264);  convert_element_type_562 = view_264 = None
        convert_element_type_573 = torch.ops.prims.convert_element_type.default(add_165, torch.float32);  add_165 = None
        pow_51 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_573, 2)
        mean_50 = torch.ops.aten.mean.dim(pow_51, [-1], True);  pow_51 = None
        add_166 = torch.ops.aten.add.Tensor(mean_50, 1e-05);  mean_50 = None
        rsqrt_50 = torch.ops.aten.rsqrt.default(add_166);  add_166 = None
        mul_207 = torch.ops.aten.mul.Tensor(convert_element_type_573, rsqrt_50);  convert_element_type_573 = rsqrt_50 = None
        convert_element_type_574 = torch.ops.prims.convert_element_type.default(mul_207, torch.bfloat16);  mul_207 = None
        convert_element_type_575 = torch.ops.prims.convert_element_type.default(primals_25, torch.bfloat16);  primals_25 = None
        permute_104 = torch.ops.aten.permute.default(convert_element_type_575, [1, 0]);  convert_element_type_575 = None
        view_265 = torch.ops.aten.view.default(convert_element_type_574, [115328, 128])
        mm_103 = torch.ops.aten.mm.default(view_265, permute_104);  view_265 = None
        view_266 = torch.ops.aten.view.default(mm_103, [128, 901, 1024]);  mm_103 = None
        split_25 = torch.ops.aten.split.Tensor(view_266, 512, -1);  view_266 = None
        getitem_154 = split_25[0]
        getitem_155 = split_25[1];  split_25 = None
        convert_element_type_578 = torch.ops.prims.convert_element_type.default(getitem_154, torch.float32);  getitem_154 = None
        sigmoid_25 = torch.ops.aten.sigmoid.default(convert_element_type_578)
        mul_208 = torch.ops.aten.mul.Tensor(convert_element_type_578, sigmoid_25);  convert_element_type_578 = sigmoid_25 = None
        convert_element_type_579 = torch.ops.prims.convert_element_type.default(mul_208, torch.bfloat16);  mul_208 = None
        mul_209 = torch.ops.aten.mul.Tensor(convert_element_type_579, getitem_155);  convert_element_type_579 = getitem_155 = None
        convert_element_type_580 = torch.ops.prims.convert_element_type.default(primals_26, torch.bfloat16);  primals_26 = None
        permute_105 = torch.ops.aten.permute.default(convert_element_type_580, [1, 0]);  convert_element_type_580 = None
        view_267 = torch.ops.aten.view.default(mul_209, [115328, 512]);  mul_209 = None
        mm_104 = torch.ops.aten.mm.default(view_267, permute_105);  view_267 = None
        view_268 = torch.ops.aten.view.default(mm_104, [128, 901, 128]);  mm_104 = None
        add_167 = torch.ops.aten.add.Tensor(convert_element_type_574, view_268);  convert_element_type_574 = view_268 = None
        convert_element_type_583 = torch.ops.prims.convert_element_type.default(add_167, torch.float32);  add_167 = None
        pow_52 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_583, 2)
        mean_51 = torch.ops.aten.mean.dim(pow_52, [-1], True);  pow_52 = None
        add_168 = torch.ops.aten.add.Tensor(mean_51, 1e-05);  mean_51 = None
        rsqrt_51 = torch.ops.aten.rsqrt.default(add_168);  add_168 = None
        mul_210 = torch.ops.aten.mul.Tensor(convert_element_type_583, rsqrt_51);  convert_element_type_583 = rsqrt_51 = None
        convert_element_type_584 = torch.ops.prims.convert_element_type.default(mul_210, torch.bfloat16);  mul_210 = None
        convert_element_type_585 = torch.ops.prims.convert_element_type.default(primals_27, torch.bfloat16);  primals_27 = None
        permute_106 = torch.ops.aten.permute.default(convert_element_type_585, [1, 0]);  convert_element_type_585 = None
        view_269 = torch.ops.aten.view.default(convert_element_type_584, [115328, 128])
        mm_105 = torch.ops.aten.mm.default(view_269, permute_106);  view_269 = None
        view_270 = torch.ops.aten.view.default(mm_105, [128, 901, 384]);  mm_105 = None
        view_271 = torch.ops.aten.view.default(view_270, [128, 901, 6, 64]);  view_270 = None
        slice_344 = torch.ops.aten.slice.Tensor(view_271, 2, 0, 2)
        slice_347 = torch.ops.aten.slice.Tensor(view_271, 2, 2, 4)
        slice_350 = torch.ops.aten.slice.Tensor(view_271, 2, 4, 9223372036854775807);  view_271 = None
        convert_element_type_588 = torch.ops.prims.convert_element_type.default(slice_344, torch.float32);  slice_344 = None
        convert_element_type_589 = torch.ops.prims.convert_element_type.default(slice_347, torch.float32);  slice_347 = None
        mul_211 = torch.ops.aten.mul.Tensor(convert_element_type_588, unsqueeze_96)
        slice_351 = torch.ops.aten.slice.Tensor(convert_element_type_588, 3, 0, 32)
        slice_352 = torch.ops.aten.slice.Tensor(convert_element_type_588, 3, 32, 9223372036854775807);  convert_element_type_588 = None
        neg_52 = torch.ops.aten.neg.default(slice_352);  slice_352 = None
        cat_54 = torch.ops.aten.cat.default([neg_52, slice_351], -1);  neg_52 = slice_351 = None
        mul_212 = torch.ops.aten.mul.Tensor(cat_54, unsqueeze_97);  cat_54 = None
        add_169 = torch.ops.aten.add.Tensor(mul_211, mul_212);  mul_211 = mul_212 = None
        mul_213 = torch.ops.aten.mul.Tensor(convert_element_type_589, unsqueeze_96)
        slice_353 = torch.ops.aten.slice.Tensor(convert_element_type_589, 3, 0, 32)
        slice_354 = torch.ops.aten.slice.Tensor(convert_element_type_589, 3, 32, 9223372036854775807);  convert_element_type_589 = None
        neg_53 = torch.ops.aten.neg.default(slice_354);  slice_354 = None
        cat_55 = torch.ops.aten.cat.default([neg_53, slice_353], -1);  neg_53 = slice_353 = None
        mul_214 = torch.ops.aten.mul.Tensor(cat_55, unsqueeze_97);  cat_55 = None
        add_170 = torch.ops.aten.add.Tensor(mul_213, mul_214);  mul_213 = mul_214 = None
        convert_element_type_590 = torch.ops.prims.convert_element_type.default(add_169, torch.bfloat16);  add_169 = None
        convert_element_type_591 = torch.ops.prims.convert_element_type.default(add_170, torch.bfloat16);  add_170 = None
        _flash_attn_forward_26 = torch.ops.flash_attn._flash_attn_forward.default(convert_element_type_590, convert_element_type_591, slice_350, 0.0, 0.125, False, -1, -1, 0.0, None, False);  convert_element_type_590 = convert_element_type_591 = slice_350 = None
        getitem_156 = _flash_attn_forward_26[0];  _flash_attn_forward_26 = None
        view_272 = torch.ops.aten.view.default(getitem_156, [128, 901, 128]);  getitem_156 = None
        convert_element_type_592 = torch.ops.prims.convert_element_type.default(primals_28, torch.bfloat16);  primals_28 = None
        permute_107 = torch.ops.aten.permute.default(convert_element_type_592, [1, 0]);  convert_element_type_592 = None
        view_273 = torch.ops.aten.view.default(view_272, [115328, 128]);  view_272 = None
        mm_106 = torch.ops.aten.mm.default(view_273, permute_107);  view_273 = None
        view_274 = torch.ops.aten.view.default(mm_106, [128, 901, 128]);  mm_106 = None
        add_171 = torch.ops.aten.add.Tensor(convert_element_type_584, view_274);  convert_element_type_584 = view_274 = None
        convert_element_type_595 = torch.ops.prims.convert_element_type.default(add_171, torch.float32);  add_171 = None
        pow_53 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_595, 2)
        mean_52 = torch.ops.aten.mean.dim(pow_53, [-1], True);  pow_53 = None
        add_172 = torch.ops.aten.add.Tensor(mean_52, 1e-05);  mean_52 = None
        rsqrt_52 = torch.ops.aten.rsqrt.default(add_172);  add_172 = None
        mul_215 = torch.ops.aten.mul.Tensor(convert_element_type_595, rsqrt_52);  convert_element_type_595 = rsqrt_52 = None
        convert_element_type_596 = torch.ops.prims.convert_element_type.default(mul_215, torch.bfloat16);  mul_215 = None
        convert_element_type_597 = torch.ops.prims.convert_element_type.default(primals_29, torch.bfloat16);  primals_29 = None
        permute_108 = torch.ops.aten.permute.default(convert_element_type_597, [1, 0]);  convert_element_type_597 = None
        view_275 = torch.ops.aten.view.default(convert_element_type_596, [115328, 128])
        mm_107 = torch.ops.aten.mm.default(view_275, permute_108);  view_275 = None
        view_276 = torch.ops.aten.view.default(mm_107, [128, 901, 1024]);  mm_107 = None
        split_26 = torch.ops.aten.split.Tensor(view_276, 512, -1);  view_276 = None
        getitem_160 = split_26[0]
        getitem_161 = split_26[1];  split_26 = None
        convert_element_type_600 = torch.ops.prims.convert_element_type.default(getitem_160, torch.float32);  getitem_160 = None
        sigmoid_26 = torch.ops.aten.sigmoid.default(convert_element_type_600)
        mul_216 = torch.ops.aten.mul.Tensor(convert_element_type_600, sigmoid_26);  convert_element_type_600 = sigmoid_26 = None
        convert_element_type_601 = torch.ops.prims.convert_element_type.default(mul_216, torch.bfloat16);  mul_216 = None
        mul_217 = torch.ops.aten.mul.Tensor(convert_element_type_601, getitem_161);  convert_element_type_601 = getitem_161 = None
        convert_element_type_602 = torch.ops.prims.convert_element_type.default(primals_30, torch.bfloat16);  primals_30 = None
        permute_109 = torch.ops.aten.permute.default(convert_element_type_602, [1, 0]);  convert_element_type_602 = None
        view_277 = torch.ops.aten.view.default(mul_217, [115328, 512]);  mul_217 = None
        mm_108 = torch.ops.aten.mm.default(view_277, permute_109);  view_277 = None
        view_278 = torch.ops.aten.view.default(mm_108, [128, 901, 128]);  mm_108 = None
        add_173 = torch.ops.aten.add.Tensor(convert_element_type_596, view_278);  convert_element_type_596 = view_278 = None
        convert_element_type_605 = torch.ops.prims.convert_element_type.default(add_173, torch.float32);  add_173 = None
        pow_54 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_605, 2)
        mean_53 = torch.ops.aten.mean.dim(pow_54, [-1], True);  pow_54 = None
        add_174 = torch.ops.aten.add.Tensor(mean_53, 1e-05);  mean_53 = None
        rsqrt_53 = torch.ops.aten.rsqrt.default(add_174);  add_174 = None
        mul_218 = torch.ops.aten.mul.Tensor(convert_element_type_605, rsqrt_53);  convert_element_type_605 = rsqrt_53 = None
        convert_element_type_606 = torch.ops.prims.convert_element_type.default(mul_218, torch.bfloat16);  mul_218 = None
        convert_element_type_607 = torch.ops.prims.convert_element_type.default(primals_31, torch.bfloat16);  primals_31 = None
        permute_110 = torch.ops.aten.permute.default(convert_element_type_607, [1, 0]);  convert_element_type_607 = None
        view_279 = torch.ops.aten.view.default(convert_element_type_606, [115328, 128])
        mm_109 = torch.ops.aten.mm.default(view_279, permute_110);  view_279 = None
        view_280 = torch.ops.aten.view.default(mm_109, [128, 901, 384]);  mm_109 = None
        view_281 = torch.ops.aten.view.default(view_280, [128, 901, 6, 64]);  view_280 = None
        slice_357 = torch.ops.aten.slice.Tensor(view_281, 2, 0, 2)
        slice_360 = torch.ops.aten.slice.Tensor(view_281, 2, 2, 4)
        slice_363 = torch.ops.aten.slice.Tensor(view_281, 2, 4, 9223372036854775807);  view_281 = None
        convert_element_type_610 = torch.ops.prims.convert_element_type.default(slice_357, torch.float32);  slice_357 = None
        convert_element_type_611 = torch.ops.prims.convert_element_type.default(slice_360, torch.float32);  slice_360 = None
        mul_219 = torch.ops.aten.mul.Tensor(convert_element_type_610, unsqueeze_96)
        slice_364 = torch.ops.aten.slice.Tensor(convert_element_type_610, 3, 0, 32)
        slice_365 = torch.ops.aten.slice.Tensor(convert_element_type_610, 3, 32, 9223372036854775807);  convert_element_type_610 = None
        neg_54 = torch.ops.aten.neg.default(slice_365);  slice_365 = None
        cat_56 = torch.ops.aten.cat.default([neg_54, slice_364], -1);  neg_54 = slice_364 = None
        mul_220 = torch.ops.aten.mul.Tensor(cat_56, unsqueeze_97);  cat_56 = None
        add_175 = torch.ops.aten.add.Tensor(mul_219, mul_220);  mul_219 = mul_220 = None
        mul_221 = torch.ops.aten.mul.Tensor(convert_element_type_611, unsqueeze_96)
        slice_366 = torch.ops.aten.slice.Tensor(convert_element_type_611, 3, 0, 32)
        slice_367 = torch.ops.aten.slice.Tensor(convert_element_type_611, 3, 32, 9223372036854775807);  convert_element_type_611 = None
        neg_55 = torch.ops.aten.neg.default(slice_367);  slice_367 = None
        cat_57 = torch.ops.aten.cat.default([neg_55, slice_366], -1);  neg_55 = slice_366 = None
        mul_222 = torch.ops.aten.mul.Tensor(cat_57, unsqueeze_97);  cat_57 = None
        add_176 = torch.ops.aten.add.Tensor(mul_221, mul_222);  mul_221 = mul_222 = None
        convert_element_type_612 = torch.ops.prims.convert_element_type.default(add_175, torch.bfloat16);  add_175 = None
        convert_element_type_613 = torch.ops.prims.convert_element_type.default(add_176, torch.bfloat16);  add_176 = None
        _flash_attn_forward_27 = torch.ops.flash_attn._flash_attn_forward.default(convert_element_type_612, convert_element_type_613, slice_363, 0.0, 0.125, False, -1, -1, 0.0, None, False);  convert_element_type_612 = convert_element_type_613 = slice_363 = None
        getitem_162 = _flash_attn_forward_27[0];  _flash_attn_forward_27 = None
        view_282 = torch.ops.aten.view.default(getitem_162, [128, 901, 128]);  getitem_162 = None
        convert_element_type_614 = torch.ops.prims.convert_element_type.default(primals_32, torch.bfloat16);  primals_32 = None
        permute_111 = torch.ops.aten.permute.default(convert_element_type_614, [1, 0]);  convert_element_type_614 = None
        view_283 = torch.ops.aten.view.default(view_282, [115328, 128]);  view_282 = None
        mm_110 = torch.ops.aten.mm.default(view_283, permute_111);  view_283 = None
        view_284 = torch.ops.aten.view.default(mm_110, [128, 901, 128]);  mm_110 = None
        add_177 = torch.ops.aten.add.Tensor(convert_element_type_606, view_284);  convert_element_type_606 = view_284 = None
        convert_element_type_617 = torch.ops.prims.convert_element_type.default(add_177, torch.float32);  add_177 = None
        pow_55 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_617, 2)
        mean_54 = torch.ops.aten.mean.dim(pow_55, [-1], True);  pow_55 = None
        add_178 = torch.ops.aten.add.Tensor(mean_54, 1e-05);  mean_54 = None
        rsqrt_54 = torch.ops.aten.rsqrt.default(add_178);  add_178 = None
        mul_223 = torch.ops.aten.mul.Tensor(convert_element_type_617, rsqrt_54);  convert_element_type_617 = rsqrt_54 = None
        convert_element_type_618 = torch.ops.prims.convert_element_type.default(mul_223, torch.bfloat16);  mul_223 = None
        convert_element_type_619 = torch.ops.prims.convert_element_type.default(primals_33, torch.bfloat16);  primals_33 = None
        permute_112 = torch.ops.aten.permute.default(convert_element_type_619, [1, 0]);  convert_element_type_619 = None
        view_285 = torch.ops.aten.view.default(convert_element_type_618, [115328, 128])
        mm_111 = torch.ops.aten.mm.default(view_285, permute_112);  view_285 = None
        view_286 = torch.ops.aten.view.default(mm_111, [128, 901, 1024]);  mm_111 = None
        split_27 = torch.ops.aten.split.Tensor(view_286, 512, -1);  view_286 = None
        getitem_166 = split_27[0]
        getitem_167 = split_27[1];  split_27 = None
        convert_element_type_622 = torch.ops.prims.convert_element_type.default(getitem_166, torch.float32);  getitem_166 = None
        sigmoid_27 = torch.ops.aten.sigmoid.default(convert_element_type_622)
        mul_224 = torch.ops.aten.mul.Tensor(convert_element_type_622, sigmoid_27);  convert_element_type_622 = sigmoid_27 = None
        convert_element_type_623 = torch.ops.prims.convert_element_type.default(mul_224, torch.bfloat16);  mul_224 = None
        mul_225 = torch.ops.aten.mul.Tensor(convert_element_type_623, getitem_167);  convert_element_type_623 = getitem_167 = None
        convert_element_type_624 = torch.ops.prims.convert_element_type.default(primals_34, torch.bfloat16);  primals_34 = None
        permute_113 = torch.ops.aten.permute.default(convert_element_type_624, [1, 0]);  convert_element_type_624 = None
        view_287 = torch.ops.aten.view.default(mul_225, [115328, 512]);  mul_225 = None
        mm_112 = torch.ops.aten.mm.default(view_287, permute_113);  view_287 = None
        view_288 = torch.ops.aten.view.default(mm_112, [128, 901, 128]);  mm_112 = None
        add_179 = torch.ops.aten.add.Tensor(convert_element_type_618, view_288);  convert_element_type_618 = view_288 = None
        convert_element_type_627 = torch.ops.prims.convert_element_type.default(add_179, torch.float32);  add_179 = None
        pow_56 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_627, 2)
        mean_55 = torch.ops.aten.mean.dim(pow_56, [-1], True);  pow_56 = None
        add_180 = torch.ops.aten.add.Tensor(mean_55, 1e-05);  mean_55 = None
        rsqrt_55 = torch.ops.aten.rsqrt.default(add_180);  add_180 = None
        mul_226 = torch.ops.aten.mul.Tensor(convert_element_type_627, rsqrt_55);  convert_element_type_627 = rsqrt_55 = None
        convert_element_type_628 = torch.ops.prims.convert_element_type.default(mul_226, torch.bfloat16);  mul_226 = None
        add_182 = torch.ops.aten.add.Tensor(convert_element_type_628, add_155);  convert_element_type_628 = add_155 = None
        view_289 = torch.ops.aten.view.default(add_182, [115328, 128])
        mm_113 = torch.ops.aten.mm.default(view_289, permute_98);  view_289 = None
        view_290 = torch.ops.aten.view.default(mm_113, [128, 901, 384]);  mm_113 = None
        view_291 = torch.ops.aten.view.default(view_290, [128, 901, 6, 64]);  view_290 = None
        slice_370 = torch.ops.aten.slice.Tensor(view_291, 2, 0, 2)
        slice_373 = torch.ops.aten.slice.Tensor(view_291, 2, 2, 4)
        slice_376 = torch.ops.aten.slice.Tensor(view_291, 2, 4, 9223372036854775807);  view_291 = None
        convert_element_type_632 = torch.ops.prims.convert_element_type.default(slice_370, torch.float32);  slice_370 = None
        convert_element_type_633 = torch.ops.prims.convert_element_type.default(slice_373, torch.float32);  slice_373 = None
        mul_227 = torch.ops.aten.mul.Tensor(convert_element_type_632, unsqueeze_96)
        slice_377 = torch.ops.aten.slice.Tensor(convert_element_type_632, 3, 0, 32)
        slice_378 = torch.ops.aten.slice.Tensor(convert_element_type_632, 3, 32, 9223372036854775807);  convert_element_type_632 = None
        neg_56 = torch.ops.aten.neg.default(slice_378);  slice_378 = None
        cat_58 = torch.ops.aten.cat.default([neg_56, slice_377], -1);  neg_56 = slice_377 = None
        mul_228 = torch.ops.aten.mul.Tensor(cat_58, unsqueeze_97);  cat_58 = None
        add_183 = torch.ops.aten.add.Tensor(mul_227, mul_228);  mul_227 = mul_228 = None
        mul_229 = torch.ops.aten.mul.Tensor(convert_element_type_633, unsqueeze_96)
        slice_379 = torch.ops.aten.slice.Tensor(convert_element_type_633, 3, 0, 32)
        slice_380 = torch.ops.aten.slice.Tensor(convert_element_type_633, 3, 32, 9223372036854775807);  convert_element_type_633 = None
        neg_57 = torch.ops.aten.neg.default(slice_380);  slice_380 = None
        cat_59 = torch.ops.aten.cat.default([neg_57, slice_379], -1);  neg_57 = slice_379 = None
        mul_230 = torch.ops.aten.mul.Tensor(cat_59, unsqueeze_97);  cat_59 = None
        add_184 = torch.ops.aten.add.Tensor(mul_229, mul_230);  mul_229 = mul_230 = None
        convert_element_type_634 = torch.ops.prims.convert_element_type.default(add_183, torch.bfloat16);  add_183 = None
        convert_element_type_635 = torch.ops.prims.convert_element_type.default(add_184, torch.bfloat16);  add_184 = None
        _flash_attn_forward_28 = torch.ops.flash_attn._flash_attn_forward.default(convert_element_type_634, convert_element_type_635, slice_376, 0.0, 0.125, False, -1, -1, 0.0, None, False);  convert_element_type_634 = convert_element_type_635 = slice_376 = None
        getitem_168 = _flash_attn_forward_28[0];  _flash_attn_forward_28 = None
        view_292 = torch.ops.aten.view.default(getitem_168, [128, 901, 128]);  getitem_168 = None
        view_293 = torch.ops.aten.view.default(view_292, [115328, 128]);  view_292 = None
        mm_114 = torch.ops.aten.mm.default(view_293, permute_99);  view_293 = None
        view_294 = torch.ops.aten.view.default(mm_114, [128, 901, 128]);  mm_114 = None
        add_185 = torch.ops.aten.add.Tensor(add_182, view_294);  add_182 = view_294 = None
        convert_element_type_639 = torch.ops.prims.convert_element_type.default(add_185, torch.float32);  add_185 = None
        pow_57 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_639, 2)
        mean_56 = torch.ops.aten.mean.dim(pow_57, [-1], True);  pow_57 = None
        add_186 = torch.ops.aten.add.Tensor(mean_56, 1e-05);  mean_56 = None
        rsqrt_56 = torch.ops.aten.rsqrt.default(add_186);  add_186 = None
        mul_231 = torch.ops.aten.mul.Tensor(convert_element_type_639, rsqrt_56);  convert_element_type_639 = rsqrt_56 = None
        convert_element_type_640 = torch.ops.prims.convert_element_type.default(mul_231, torch.bfloat16);  mul_231 = None
        view_295 = torch.ops.aten.view.default(convert_element_type_640, [115328, 128])
        mm_115 = torch.ops.aten.mm.default(view_295, permute_100);  view_295 = None
        view_296 = torch.ops.aten.view.default(mm_115, [128, 901, 1024]);  mm_115 = None
        split_28 = torch.ops.aten.split.Tensor(view_296, 512, -1);  view_296 = None
        getitem_172 = split_28[0]
        getitem_173 = split_28[1];  split_28 = None
        convert_element_type_644 = torch.ops.prims.convert_element_type.default(getitem_172, torch.float32);  getitem_172 = None
        sigmoid_28 = torch.ops.aten.sigmoid.default(convert_element_type_644)
        mul_232 = torch.ops.aten.mul.Tensor(convert_element_type_644, sigmoid_28);  convert_element_type_644 = sigmoid_28 = None
        convert_element_type_645 = torch.ops.prims.convert_element_type.default(mul_232, torch.bfloat16);  mul_232 = None
        mul_233 = torch.ops.aten.mul.Tensor(convert_element_type_645, getitem_173);  convert_element_type_645 = getitem_173 = None
        view_297 = torch.ops.aten.view.default(mul_233, [115328, 512]);  mul_233 = None
        mm_116 = torch.ops.aten.mm.default(view_297, permute_101);  view_297 = None
        view_298 = torch.ops.aten.view.default(mm_116, [128, 901, 128]);  mm_116 = None
        add_187 = torch.ops.aten.add.Tensor(convert_element_type_640, view_298);  convert_element_type_640 = view_298 = None
        convert_element_type_649 = torch.ops.prims.convert_element_type.default(add_187, torch.float32);  add_187 = None
        pow_58 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_649, 2)
        mean_57 = torch.ops.aten.mean.dim(pow_58, [-1], True);  pow_58 = None
        add_188 = torch.ops.aten.add.Tensor(mean_57, 1e-05);  mean_57 = None
        rsqrt_57 = torch.ops.aten.rsqrt.default(add_188);  add_188 = None
        mul_234 = torch.ops.aten.mul.Tensor(convert_element_type_649, rsqrt_57);  convert_element_type_649 = rsqrt_57 = None
        convert_element_type_650 = torch.ops.prims.convert_element_type.default(mul_234, torch.bfloat16);  mul_234 = None
        view_299 = torch.ops.aten.view.default(convert_element_type_650, [115328, 128])
        mm_117 = torch.ops.aten.mm.default(view_299, permute_102);  view_299 = None
        view_300 = torch.ops.aten.view.default(mm_117, [128, 901, 384]);  mm_117 = None
        view_301 = torch.ops.aten.view.default(view_300, [128, 901, 6, 64]);  view_300 = None
        slice_383 = torch.ops.aten.slice.Tensor(view_301, 2, 0, 2)
        slice_386 = torch.ops.aten.slice.Tensor(view_301, 2, 2, 4)
        slice_389 = torch.ops.aten.slice.Tensor(view_301, 2, 4, 9223372036854775807);  view_301 = None
        convert_element_type_654 = torch.ops.prims.convert_element_type.default(slice_383, torch.float32);  slice_383 = None
        convert_element_type_655 = torch.ops.prims.convert_element_type.default(slice_386, torch.float32);  slice_386 = None
        mul_235 = torch.ops.aten.mul.Tensor(convert_element_type_654, unsqueeze_96)
        slice_390 = torch.ops.aten.slice.Tensor(convert_element_type_654, 3, 0, 32)
        slice_391 = torch.ops.aten.slice.Tensor(convert_element_type_654, 3, 32, 9223372036854775807);  convert_element_type_654 = None
        neg_58 = torch.ops.aten.neg.default(slice_391);  slice_391 = None
        cat_60 = torch.ops.aten.cat.default([neg_58, slice_390], -1);  neg_58 = slice_390 = None
        mul_236 = torch.ops.aten.mul.Tensor(cat_60, unsqueeze_97);  cat_60 = None
        add_189 = torch.ops.aten.add.Tensor(mul_235, mul_236);  mul_235 = mul_236 = None
        mul_237 = torch.ops.aten.mul.Tensor(convert_element_type_655, unsqueeze_96)
        slice_392 = torch.ops.aten.slice.Tensor(convert_element_type_655, 3, 0, 32)
        slice_393 = torch.ops.aten.slice.Tensor(convert_element_type_655, 3, 32, 9223372036854775807);  convert_element_type_655 = None
        neg_59 = torch.ops.aten.neg.default(slice_393);  slice_393 = None
        cat_61 = torch.ops.aten.cat.default([neg_59, slice_392], -1);  neg_59 = slice_392 = None
        mul_238 = torch.ops.aten.mul.Tensor(cat_61, unsqueeze_97);  cat_61 = None
        add_190 = torch.ops.aten.add.Tensor(mul_237, mul_238);  mul_237 = mul_238 = None
        convert_element_type_656 = torch.ops.prims.convert_element_type.default(add_189, torch.bfloat16);  add_189 = None
        convert_element_type_657 = torch.ops.prims.convert_element_type.default(add_190, torch.bfloat16);  add_190 = None
        _flash_attn_forward_29 = torch.ops.flash_attn._flash_attn_forward.default(convert_element_type_656, convert_element_type_657, slice_389, 0.0, 0.125, False, -1, -1, 0.0, None, False);  convert_element_type_656 = convert_element_type_657 = slice_389 = None
        getitem_174 = _flash_attn_forward_29[0];  _flash_attn_forward_29 = None
        view_302 = torch.ops.aten.view.default(getitem_174, [128, 901, 128]);  getitem_174 = None
        view_303 = torch.ops.aten.view.default(view_302, [115328, 128]);  view_302 = None
        mm_118 = torch.ops.aten.mm.default(view_303, permute_103);  view_303 = None
        view_304 = torch.ops.aten.view.default(mm_118, [128, 901, 128]);  mm_118 = None
        add_191 = torch.ops.aten.add.Tensor(convert_element_type_650, view_304);  convert_element_type_650 = view_304 = None
        convert_element_type_661 = torch.ops.prims.convert_element_type.default(add_191, torch.float32);  add_191 = None
        pow_59 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_661, 2)
        mean_58 = torch.ops.aten.mean.dim(pow_59, [-1], True);  pow_59 = None
        add_192 = torch.ops.aten.add.Tensor(mean_58, 1e-05);  mean_58 = None
        rsqrt_58 = torch.ops.aten.rsqrt.default(add_192);  add_192 = None
        mul_239 = torch.ops.aten.mul.Tensor(convert_element_type_661, rsqrt_58);  convert_element_type_661 = rsqrt_58 = None
        convert_element_type_662 = torch.ops.prims.convert_element_type.default(mul_239, torch.bfloat16);  mul_239 = None
        view_305 = torch.ops.aten.view.default(convert_element_type_662, [115328, 128])
        mm_119 = torch.ops.aten.mm.default(view_305, permute_104);  view_305 = None
        view_306 = torch.ops.aten.view.default(mm_119, [128, 901, 1024]);  mm_119 = None
        split_29 = torch.ops.aten.split.Tensor(view_306, 512, -1);  view_306 = None
        getitem_178 = split_29[0]
        getitem_179 = split_29[1];  split_29 = None
        convert_element_type_666 = torch.ops.prims.convert_element_type.default(getitem_178, torch.float32);  getitem_178 = None
        sigmoid_29 = torch.ops.aten.sigmoid.default(convert_element_type_666)
        mul_240 = torch.ops.aten.mul.Tensor(convert_element_type_666, sigmoid_29);  convert_element_type_666 = sigmoid_29 = None
        convert_element_type_667 = torch.ops.prims.convert_element_type.default(mul_240, torch.bfloat16);  mul_240 = None
        mul_241 = torch.ops.aten.mul.Tensor(convert_element_type_667, getitem_179);  convert_element_type_667 = getitem_179 = None
        view_307 = torch.ops.aten.view.default(mul_241, [115328, 512]);  mul_241 = None
        mm_120 = torch.ops.aten.mm.default(view_307, permute_105);  view_307 = None
        view_308 = torch.ops.aten.view.default(mm_120, [128, 901, 128]);  mm_120 = None
        add_193 = torch.ops.aten.add.Tensor(convert_element_type_662, view_308);  convert_element_type_662 = view_308 = None
        convert_element_type_671 = torch.ops.prims.convert_element_type.default(add_193, torch.float32);  add_193 = None
        pow_60 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_671, 2)
        mean_59 = torch.ops.aten.mean.dim(pow_60, [-1], True);  pow_60 = None
        add_194 = torch.ops.aten.add.Tensor(mean_59, 1e-05);  mean_59 = None
        rsqrt_59 = torch.ops.aten.rsqrt.default(add_194);  add_194 = None
        mul_242 = torch.ops.aten.mul.Tensor(convert_element_type_671, rsqrt_59);  convert_element_type_671 = rsqrt_59 = None
        convert_element_type_672 = torch.ops.prims.convert_element_type.default(mul_242, torch.bfloat16);  mul_242 = None
        view_309 = torch.ops.aten.view.default(convert_element_type_672, [115328, 128])
        mm_121 = torch.ops.aten.mm.default(view_309, permute_106);  view_309 = None
        view_310 = torch.ops.aten.view.default(mm_121, [128, 901, 384]);  mm_121 = None
        view_311 = torch.ops.aten.view.default(view_310, [128, 901, 6, 64]);  view_310 = None
        slice_396 = torch.ops.aten.slice.Tensor(view_311, 2, 0, 2)
        slice_399 = torch.ops.aten.slice.Tensor(view_311, 2, 2, 4)
        slice_402 = torch.ops.aten.slice.Tensor(view_311, 2, 4, 9223372036854775807);  view_311 = None
        convert_element_type_676 = torch.ops.prims.convert_element_type.default(slice_396, torch.float32);  slice_396 = None
        convert_element_type_677 = torch.ops.prims.convert_element_type.default(slice_399, torch.float32);  slice_399 = None
        mul_243 = torch.ops.aten.mul.Tensor(convert_element_type_676, unsqueeze_96)
        slice_403 = torch.ops.aten.slice.Tensor(convert_element_type_676, 3, 0, 32)
        slice_404 = torch.ops.aten.slice.Tensor(convert_element_type_676, 3, 32, 9223372036854775807);  convert_element_type_676 = None
        neg_60 = torch.ops.aten.neg.default(slice_404);  slice_404 = None
        cat_62 = torch.ops.aten.cat.default([neg_60, slice_403], -1);  neg_60 = slice_403 = None
        mul_244 = torch.ops.aten.mul.Tensor(cat_62, unsqueeze_97);  cat_62 = None
        add_195 = torch.ops.aten.add.Tensor(mul_243, mul_244);  mul_243 = mul_244 = None
        mul_245 = torch.ops.aten.mul.Tensor(convert_element_type_677, unsqueeze_96)
        slice_405 = torch.ops.aten.slice.Tensor(convert_element_type_677, 3, 0, 32)
        slice_406 = torch.ops.aten.slice.Tensor(convert_element_type_677, 3, 32, 9223372036854775807);  convert_element_type_677 = None
        neg_61 = torch.ops.aten.neg.default(slice_406);  slice_406 = None
        cat_63 = torch.ops.aten.cat.default([neg_61, slice_405], -1);  neg_61 = slice_405 = None
        mul_246 = torch.ops.aten.mul.Tensor(cat_63, unsqueeze_97);  cat_63 = None
        add_196 = torch.ops.aten.add.Tensor(mul_245, mul_246);  mul_245 = mul_246 = None
        convert_element_type_678 = torch.ops.prims.convert_element_type.default(add_195, torch.bfloat16);  add_195 = None
        convert_element_type_679 = torch.ops.prims.convert_element_type.default(add_196, torch.bfloat16);  add_196 = None
        _flash_attn_forward_30 = torch.ops.flash_attn._flash_attn_forward.default(convert_element_type_678, convert_element_type_679, slice_402, 0.0, 0.125, False, -1, -1, 0.0, None, False);  convert_element_type_678 = convert_element_type_679 = slice_402 = None
        getitem_180 = _flash_attn_forward_30[0];  _flash_attn_forward_30 = None
        view_312 = torch.ops.aten.view.default(getitem_180, [128, 901, 128]);  getitem_180 = None
        view_313 = torch.ops.aten.view.default(view_312, [115328, 128]);  view_312 = None
        mm_122 = torch.ops.aten.mm.default(view_313, permute_107);  view_313 = None
        view_314 = torch.ops.aten.view.default(mm_122, [128, 901, 128]);  mm_122 = None
        add_197 = torch.ops.aten.add.Tensor(convert_element_type_672, view_314);  convert_element_type_672 = view_314 = None
        convert_element_type_683 = torch.ops.prims.convert_element_type.default(add_197, torch.float32);  add_197 = None
        pow_61 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_683, 2)
        mean_60 = torch.ops.aten.mean.dim(pow_61, [-1], True);  pow_61 = None
        add_198 = torch.ops.aten.add.Tensor(mean_60, 1e-05);  mean_60 = None
        rsqrt_60 = torch.ops.aten.rsqrt.default(add_198);  add_198 = None
        mul_247 = torch.ops.aten.mul.Tensor(convert_element_type_683, rsqrt_60);  convert_element_type_683 = rsqrt_60 = None
        convert_element_type_684 = torch.ops.prims.convert_element_type.default(mul_247, torch.bfloat16);  mul_247 = None
        view_315 = torch.ops.aten.view.default(convert_element_type_684, [115328, 128])
        mm_123 = torch.ops.aten.mm.default(view_315, permute_108);  view_315 = None
        view_316 = torch.ops.aten.view.default(mm_123, [128, 901, 1024]);  mm_123 = None
        split_30 = torch.ops.aten.split.Tensor(view_316, 512, -1);  view_316 = None
        getitem_184 = split_30[0]
        getitem_185 = split_30[1];  split_30 = None
        convert_element_type_688 = torch.ops.prims.convert_element_type.default(getitem_184, torch.float32);  getitem_184 = None
        sigmoid_30 = torch.ops.aten.sigmoid.default(convert_element_type_688)
        mul_248 = torch.ops.aten.mul.Tensor(convert_element_type_688, sigmoid_30);  convert_element_type_688 = sigmoid_30 = None
        convert_element_type_689 = torch.ops.prims.convert_element_type.default(mul_248, torch.bfloat16);  mul_248 = None
        mul_249 = torch.ops.aten.mul.Tensor(convert_element_type_689, getitem_185);  convert_element_type_689 = getitem_185 = None
        view_317 = torch.ops.aten.view.default(mul_249, [115328, 512]);  mul_249 = None
        mm_124 = torch.ops.aten.mm.default(view_317, permute_109);  view_317 = None
        view_318 = torch.ops.aten.view.default(mm_124, [128, 901, 128]);  mm_124 = None
        add_199 = torch.ops.aten.add.Tensor(convert_element_type_684, view_318);  convert_element_type_684 = view_318 = None
        convert_element_type_693 = torch.ops.prims.convert_element_type.default(add_199, torch.float32);  add_199 = None
        pow_62 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_693, 2)
        mean_61 = torch.ops.aten.mean.dim(pow_62, [-1], True);  pow_62 = None
        add_200 = torch.ops.aten.add.Tensor(mean_61, 1e-05);  mean_61 = None
        rsqrt_61 = torch.ops.aten.rsqrt.default(add_200);  add_200 = None
        mul_250 = torch.ops.aten.mul.Tensor(convert_element_type_693, rsqrt_61);  convert_element_type_693 = rsqrt_61 = None
        convert_element_type_694 = torch.ops.prims.convert_element_type.default(mul_250, torch.bfloat16);  mul_250 = None
        view_319 = torch.ops.aten.view.default(convert_element_type_694, [115328, 128])
        mm_125 = torch.ops.aten.mm.default(view_319, permute_110);  view_319 = None
        view_320 = torch.ops.aten.view.default(mm_125, [128, 901, 384]);  mm_125 = None
        view_321 = torch.ops.aten.view.default(view_320, [128, 901, 6, 64]);  view_320 = None
        slice_409 = torch.ops.aten.slice.Tensor(view_321, 2, 0, 2)
        slice_412 = torch.ops.aten.slice.Tensor(view_321, 2, 2, 4)
        slice_415 = torch.ops.aten.slice.Tensor(view_321, 2, 4, 9223372036854775807);  view_321 = None
        convert_element_type_698 = torch.ops.prims.convert_element_type.default(slice_409, torch.float32);  slice_409 = None
        convert_element_type_699 = torch.ops.prims.convert_element_type.default(slice_412, torch.float32);  slice_412 = None
        mul_251 = torch.ops.aten.mul.Tensor(convert_element_type_698, unsqueeze_96)
        slice_416 = torch.ops.aten.slice.Tensor(convert_element_type_698, 3, 0, 32)
        slice_417 = torch.ops.aten.slice.Tensor(convert_element_type_698, 3, 32, 9223372036854775807);  convert_element_type_698 = None
        neg_62 = torch.ops.aten.neg.default(slice_417);  slice_417 = None
        cat_64 = torch.ops.aten.cat.default([neg_62, slice_416], -1);  neg_62 = slice_416 = None
        mul_252 = torch.ops.aten.mul.Tensor(cat_64, unsqueeze_97);  cat_64 = None
        add_201 = torch.ops.aten.add.Tensor(mul_251, mul_252);  mul_251 = mul_252 = None
        mul_253 = torch.ops.aten.mul.Tensor(convert_element_type_699, unsqueeze_96)
        slice_418 = torch.ops.aten.slice.Tensor(convert_element_type_699, 3, 0, 32)
        slice_419 = torch.ops.aten.slice.Tensor(convert_element_type_699, 3, 32, 9223372036854775807);  convert_element_type_699 = None
        neg_63 = torch.ops.aten.neg.default(slice_419);  slice_419 = None
        cat_65 = torch.ops.aten.cat.default([neg_63, slice_418], -1);  neg_63 = slice_418 = None
        mul_254 = torch.ops.aten.mul.Tensor(cat_65, unsqueeze_97);  cat_65 = None
        add_202 = torch.ops.aten.add.Tensor(mul_253, mul_254);  mul_253 = mul_254 = None
        convert_element_type_700 = torch.ops.prims.convert_element_type.default(add_201, torch.bfloat16);  add_201 = None
        convert_element_type_701 = torch.ops.prims.convert_element_type.default(add_202, torch.bfloat16);  add_202 = None
        _flash_attn_forward_31 = torch.ops.flash_attn._flash_attn_forward.default(convert_element_type_700, convert_element_type_701, slice_415, 0.0, 0.125, False, -1, -1, 0.0, None, False);  convert_element_type_700 = convert_element_type_701 = slice_415 = None
        getitem_186 = _flash_attn_forward_31[0];  _flash_attn_forward_31 = None
        view_322 = torch.ops.aten.view.default(getitem_186, [128, 901, 128]);  getitem_186 = None
        view_323 = torch.ops.aten.view.default(view_322, [115328, 128]);  view_322 = None
        mm_126 = torch.ops.aten.mm.default(view_323, permute_111);  view_323 = None
        view_324 = torch.ops.aten.view.default(mm_126, [128, 901, 128]);  mm_126 = None
        add_203 = torch.ops.aten.add.Tensor(convert_element_type_694, view_324);  convert_element_type_694 = view_324 = None
        convert_element_type_705 = torch.ops.prims.convert_element_type.default(add_203, torch.float32);  add_203 = None
        pow_63 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_705, 2)
        mean_62 = torch.ops.aten.mean.dim(pow_63, [-1], True);  pow_63 = None
        add_204 = torch.ops.aten.add.Tensor(mean_62, 1e-05);  mean_62 = None
        rsqrt_62 = torch.ops.aten.rsqrt.default(add_204);  add_204 = None
        mul_255 = torch.ops.aten.mul.Tensor(convert_element_type_705, rsqrt_62);  convert_element_type_705 = rsqrt_62 = None
        convert_element_type_706 = torch.ops.prims.convert_element_type.default(mul_255, torch.bfloat16);  mul_255 = None
        view_325 = torch.ops.aten.view.default(convert_element_type_706, [115328, 128])
        mm_127 = torch.ops.aten.mm.default(view_325, permute_112);  view_325 = None
        view_326 = torch.ops.aten.view.default(mm_127, [128, 901, 1024]);  mm_127 = None
        split_31 = torch.ops.aten.split.Tensor(view_326, 512, -1);  view_326 = None
        getitem_190 = split_31[0]
        getitem_191 = split_31[1];  split_31 = None
        convert_element_type_710 = torch.ops.prims.convert_element_type.default(getitem_190, torch.float32);  getitem_190 = None
        sigmoid_31 = torch.ops.aten.sigmoid.default(convert_element_type_710)
        mul_256 = torch.ops.aten.mul.Tensor(convert_element_type_710, sigmoid_31);  convert_element_type_710 = sigmoid_31 = None
        convert_element_type_711 = torch.ops.prims.convert_element_type.default(mul_256, torch.bfloat16);  mul_256 = None
        mul_257 = torch.ops.aten.mul.Tensor(convert_element_type_711, getitem_191);  convert_element_type_711 = getitem_191 = None
        view_327 = torch.ops.aten.view.default(mul_257, [115328, 512]);  mul_257 = None
        mm_128 = torch.ops.aten.mm.default(view_327, permute_113);  view_327 = None
        view_328 = torch.ops.aten.view.default(mm_128, [128, 901, 128]);  mm_128 = None
        add_205 = torch.ops.aten.add.Tensor(convert_element_type_706, view_328);  convert_element_type_706 = view_328 = None
        convert_element_type_715 = torch.ops.prims.convert_element_type.default(add_205, torch.float32);  add_205 = None
        pow_64 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_715, 2)
        mean_63 = torch.ops.aten.mean.dim(pow_64, [-1], True);  pow_64 = None
        add_206 = torch.ops.aten.add.Tensor(mean_63, 1e-05);  mean_63 = None
        rsqrt_63 = torch.ops.aten.rsqrt.default(add_206);  add_206 = None
        mul_258 = torch.ops.aten.mul.Tensor(convert_element_type_715, rsqrt_63);  convert_element_type_715 = rsqrt_63 = None
        convert_element_type_716 = torch.ops.prims.convert_element_type.default(mul_258, torch.bfloat16);  mul_258 = None
        add_207 = torch.ops.aten.add.Tensor(convert_element_type_529, convert_element_type_716)
        convert_element_type_717 = torch.ops.prims.convert_element_type.default(primals_35, torch.bfloat16);  primals_35 = None
        permute_130 = torch.ops.aten.permute.default(convert_element_type_717, [1, 0]);  convert_element_type_717 = None
        view_329 = torch.ops.aten.view.default(add_207, [115328, 128])
        mm_129 = torch.ops.aten.mm.default(view_329, permute_130);  view_329 = None
        view_330 = torch.ops.aten.view.default(mm_129, [128, 901, 384]);  mm_129 = None
        view_331 = torch.ops.aten.view.default(view_330, [128, 901, 6, 64]);  view_330 = None
        slice_422 = torch.ops.aten.slice.Tensor(view_331, 2, 0, 2)
        slice_425 = torch.ops.aten.slice.Tensor(view_331, 2, 2, 4)
        slice_428 = torch.ops.aten.slice.Tensor(view_331, 2, 4, 9223372036854775807);  view_331 = None
        convert_element_type_720 = torch.ops.prims.convert_element_type.default(slice_422, torch.float32);  slice_422 = None
        convert_element_type_721 = torch.ops.prims.convert_element_type.default(slice_425, torch.float32);  slice_425 = None
        mul_259 = torch.ops.aten.mul.Tensor(convert_element_type_720, unsqueeze_96)
        slice_429 = torch.ops.aten.slice.Tensor(convert_element_type_720, 3, 0, 32)
        slice_430 = torch.ops.aten.slice.Tensor(convert_element_type_720, 3, 32, 9223372036854775807);  convert_element_type_720 = None
        neg_64 = torch.ops.aten.neg.default(slice_430);  slice_430 = None
        cat_66 = torch.ops.aten.cat.default([neg_64, slice_429], -1);  neg_64 = slice_429 = None
        mul_260 = torch.ops.aten.mul.Tensor(cat_66, unsqueeze_97);  cat_66 = None
        add_208 = torch.ops.aten.add.Tensor(mul_259, mul_260);  mul_259 = mul_260 = None
        mul_261 = torch.ops.aten.mul.Tensor(convert_element_type_721, unsqueeze_96)
        slice_431 = torch.ops.aten.slice.Tensor(convert_element_type_721, 3, 0, 32)
        slice_432 = torch.ops.aten.slice.Tensor(convert_element_type_721, 3, 32, 9223372036854775807);  convert_element_type_721 = None
        neg_65 = torch.ops.aten.neg.default(slice_432);  slice_432 = None
        cat_67 = torch.ops.aten.cat.default([neg_65, slice_431], -1);  neg_65 = slice_431 = None
        mul_262 = torch.ops.aten.mul.Tensor(cat_67, unsqueeze_97);  cat_67 = None
        add_209 = torch.ops.aten.add.Tensor(mul_261, mul_262);  mul_261 = mul_262 = None
        convert_element_type_722 = torch.ops.prims.convert_element_type.default(add_208, torch.bfloat16);  add_208 = None
        convert_element_type_723 = torch.ops.prims.convert_element_type.default(add_209, torch.bfloat16);  add_209 = None
        _flash_attn_forward_32 = torch.ops.flash_attn._flash_attn_forward.default(convert_element_type_722, convert_element_type_723, slice_428, 0.0, 0.125, False, -1, -1, 0.0, None, False);  convert_element_type_722 = convert_element_type_723 = slice_428 = None
        getitem_192 = _flash_attn_forward_32[0];  _flash_attn_forward_32 = None
        view_332 = torch.ops.aten.view.default(getitem_192, [128, 901, 128]);  getitem_192 = None
        convert_element_type_724 = torch.ops.prims.convert_element_type.default(primals_36, torch.bfloat16);  primals_36 = None
        permute_131 = torch.ops.aten.permute.default(convert_element_type_724, [1, 0]);  convert_element_type_724 = None
        view_333 = torch.ops.aten.view.default(view_332, [115328, 128]);  view_332 = None
        mm_130 = torch.ops.aten.mm.default(view_333, permute_131);  view_333 = None
        view_334 = torch.ops.aten.view.default(mm_130, [128, 901, 128]);  mm_130 = None
        add_210 = torch.ops.aten.add.Tensor(add_207, view_334);  add_207 = view_334 = None
        convert_element_type_727 = torch.ops.prims.convert_element_type.default(add_210, torch.float32);  add_210 = None
        pow_65 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_727, 2)
        mean_64 = torch.ops.aten.mean.dim(pow_65, [-1], True);  pow_65 = None
        add_211 = torch.ops.aten.add.Tensor(mean_64, 1e-05);  mean_64 = None
        rsqrt_64 = torch.ops.aten.rsqrt.default(add_211);  add_211 = None
        mul_263 = torch.ops.aten.mul.Tensor(convert_element_type_727, rsqrt_64);  convert_element_type_727 = rsqrt_64 = None
        convert_element_type_728 = torch.ops.prims.convert_element_type.default(mul_263, torch.bfloat16);  mul_263 = None
        convert_element_type_729 = torch.ops.prims.convert_element_type.default(primals_37, torch.bfloat16);  primals_37 = None
        permute_132 = torch.ops.aten.permute.default(convert_element_type_729, [1, 0]);  convert_element_type_729 = None
        view_335 = torch.ops.aten.view.default(convert_element_type_728, [115328, 128])
        mm_131 = torch.ops.aten.mm.default(view_335, permute_132);  view_335 = None
        view_336 = torch.ops.aten.view.default(mm_131, [128, 901, 1024]);  mm_131 = None
        split_32 = torch.ops.aten.split.Tensor(view_336, 512, -1);  view_336 = None
        getitem_196 = split_32[0]
        getitem_197 = split_32[1];  split_32 = None
        convert_element_type_732 = torch.ops.prims.convert_element_type.default(getitem_196, torch.float32);  getitem_196 = None
        sigmoid_32 = torch.ops.aten.sigmoid.default(convert_element_type_732)
        mul_264 = torch.ops.aten.mul.Tensor(convert_element_type_732, sigmoid_32);  convert_element_type_732 = sigmoid_32 = None
        convert_element_type_733 = torch.ops.prims.convert_element_type.default(mul_264, torch.bfloat16);  mul_264 = None
        mul_265 = torch.ops.aten.mul.Tensor(convert_element_type_733, getitem_197);  convert_element_type_733 = getitem_197 = None
        convert_element_type_734 = torch.ops.prims.convert_element_type.default(primals_38, torch.bfloat16);  primals_38 = None
        permute_133 = torch.ops.aten.permute.default(convert_element_type_734, [1, 0]);  convert_element_type_734 = None
        view_337 = torch.ops.aten.view.default(mul_265, [115328, 512]);  mul_265 = None
        mm_132 = torch.ops.aten.mm.default(view_337, permute_133);  view_337 = None
        view_338 = torch.ops.aten.view.default(mm_132, [128, 901, 128]);  mm_132 = None
        add_212 = torch.ops.aten.add.Tensor(convert_element_type_728, view_338);  convert_element_type_728 = view_338 = None
        convert_element_type_737 = torch.ops.prims.convert_element_type.default(add_212, torch.float32);  add_212 = None
        pow_66 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_737, 2)
        mean_65 = torch.ops.aten.mean.dim(pow_66, [-1], True);  pow_66 = None
        add_213 = torch.ops.aten.add.Tensor(mean_65, 1e-05);  mean_65 = None
        rsqrt_65 = torch.ops.aten.rsqrt.default(add_213);  add_213 = None
        mul_266 = torch.ops.aten.mul.Tensor(convert_element_type_737, rsqrt_65);  convert_element_type_737 = rsqrt_65 = None
        convert_element_type_738 = torch.ops.prims.convert_element_type.default(mul_266, torch.bfloat16);  mul_266 = None
        convert_element_type_739 = torch.ops.prims.convert_element_type.default(primals_39, torch.bfloat16);  primals_39 = None
        permute_134 = torch.ops.aten.permute.default(convert_element_type_739, [1, 0]);  convert_element_type_739 = None
        view_339 = torch.ops.aten.view.default(convert_element_type_738, [115328, 128])
        mm_133 = torch.ops.aten.mm.default(view_339, permute_134);  view_339 = None
        view_340 = torch.ops.aten.view.default(mm_133, [128, 901, 384]);  mm_133 = None
        view_341 = torch.ops.aten.view.default(view_340, [128, 901, 6, 64]);  view_340 = None
        slice_435 = torch.ops.aten.slice.Tensor(view_341, 2, 0, 2)
        slice_438 = torch.ops.aten.slice.Tensor(view_341, 2, 2, 4)
        slice_441 = torch.ops.aten.slice.Tensor(view_341, 2, 4, 9223372036854775807);  view_341 = None
        convert_element_type_742 = torch.ops.prims.convert_element_type.default(slice_435, torch.float32);  slice_435 = None
        convert_element_type_743 = torch.ops.prims.convert_element_type.default(slice_438, torch.float32);  slice_438 = None
        mul_267 = torch.ops.aten.mul.Tensor(convert_element_type_742, unsqueeze_96)
        slice_442 = torch.ops.aten.slice.Tensor(convert_element_type_742, 3, 0, 32)
        slice_443 = torch.ops.aten.slice.Tensor(convert_element_type_742, 3, 32, 9223372036854775807);  convert_element_type_742 = None
        neg_66 = torch.ops.aten.neg.default(slice_443);  slice_443 = None
        cat_68 = torch.ops.aten.cat.default([neg_66, slice_442], -1);  neg_66 = slice_442 = None
        mul_268 = torch.ops.aten.mul.Tensor(cat_68, unsqueeze_97);  cat_68 = None
        add_214 = torch.ops.aten.add.Tensor(mul_267, mul_268);  mul_267 = mul_268 = None
        mul_269 = torch.ops.aten.mul.Tensor(convert_element_type_743, unsqueeze_96)
        slice_444 = torch.ops.aten.slice.Tensor(convert_element_type_743, 3, 0, 32)
        slice_445 = torch.ops.aten.slice.Tensor(convert_element_type_743, 3, 32, 9223372036854775807);  convert_element_type_743 = None
        neg_67 = torch.ops.aten.neg.default(slice_445);  slice_445 = None
        cat_69 = torch.ops.aten.cat.default([neg_67, slice_444], -1);  neg_67 = slice_444 = None
        mul_270 = torch.ops.aten.mul.Tensor(cat_69, unsqueeze_97);  cat_69 = None
        add_215 = torch.ops.aten.add.Tensor(mul_269, mul_270);  mul_269 = mul_270 = None
        convert_element_type_744 = torch.ops.prims.convert_element_type.default(add_214, torch.bfloat16);  add_214 = None
        convert_element_type_745 = torch.ops.prims.convert_element_type.default(add_215, torch.bfloat16);  add_215 = None
        _flash_attn_forward_33 = torch.ops.flash_attn._flash_attn_forward.default(convert_element_type_744, convert_element_type_745, slice_441, 0.0, 0.125, False, -1, -1, 0.0, None, False);  convert_element_type_744 = convert_element_type_745 = slice_441 = None
        getitem_198 = _flash_attn_forward_33[0];  _flash_attn_forward_33 = None
        view_342 = torch.ops.aten.view.default(getitem_198, [128, 901, 128]);  getitem_198 = None
        convert_element_type_746 = torch.ops.prims.convert_element_type.default(primals_40, torch.bfloat16);  primals_40 = None
        permute_135 = torch.ops.aten.permute.default(convert_element_type_746, [1, 0]);  convert_element_type_746 = None
        view_343 = torch.ops.aten.view.default(view_342, [115328, 128]);  view_342 = None
        mm_134 = torch.ops.aten.mm.default(view_343, permute_135);  view_343 = None
        view_344 = torch.ops.aten.view.default(mm_134, [128, 901, 128]);  mm_134 = None
        add_216 = torch.ops.aten.add.Tensor(convert_element_type_738, view_344);  convert_element_type_738 = view_344 = None
        convert_element_type_749 = torch.ops.prims.convert_element_type.default(add_216, torch.float32);  add_216 = None
        pow_67 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_749, 2)
        mean_66 = torch.ops.aten.mean.dim(pow_67, [-1], True);  pow_67 = None
        add_217 = torch.ops.aten.add.Tensor(mean_66, 1e-05);  mean_66 = None
        rsqrt_66 = torch.ops.aten.rsqrt.default(add_217);  add_217 = None
        mul_271 = torch.ops.aten.mul.Tensor(convert_element_type_749, rsqrt_66);  convert_element_type_749 = rsqrt_66 = None
        convert_element_type_750 = torch.ops.prims.convert_element_type.default(mul_271, torch.bfloat16);  mul_271 = None
        convert_element_type_751 = torch.ops.prims.convert_element_type.default(primals_41, torch.bfloat16);  primals_41 = None
        permute_136 = torch.ops.aten.permute.default(convert_element_type_751, [1, 0]);  convert_element_type_751 = None
        view_345 = torch.ops.aten.view.default(convert_element_type_750, [115328, 128])
        mm_135 = torch.ops.aten.mm.default(view_345, permute_136);  view_345 = None
        view_346 = torch.ops.aten.view.default(mm_135, [128, 901, 1024]);  mm_135 = None
        split_33 = torch.ops.aten.split.Tensor(view_346, 512, -1);  view_346 = None
        getitem_202 = split_33[0]
        getitem_203 = split_33[1];  split_33 = None
        convert_element_type_754 = torch.ops.prims.convert_element_type.default(getitem_202, torch.float32);  getitem_202 = None
        sigmoid_33 = torch.ops.aten.sigmoid.default(convert_element_type_754)
        mul_272 = torch.ops.aten.mul.Tensor(convert_element_type_754, sigmoid_33);  convert_element_type_754 = sigmoid_33 = None
        convert_element_type_755 = torch.ops.prims.convert_element_type.default(mul_272, torch.bfloat16);  mul_272 = None
        mul_273 = torch.ops.aten.mul.Tensor(convert_element_type_755, getitem_203);  convert_element_type_755 = getitem_203 = None
        convert_element_type_756 = torch.ops.prims.convert_element_type.default(primals_42, torch.bfloat16);  primals_42 = None
        permute_137 = torch.ops.aten.permute.default(convert_element_type_756, [1, 0]);  convert_element_type_756 = None
        view_347 = torch.ops.aten.view.default(mul_273, [115328, 512]);  mul_273 = None
        mm_136 = torch.ops.aten.mm.default(view_347, permute_137);  view_347 = None
        view_348 = torch.ops.aten.view.default(mm_136, [128, 901, 128]);  mm_136 = None
        add_218 = torch.ops.aten.add.Tensor(convert_element_type_750, view_348);  convert_element_type_750 = view_348 = None
        convert_element_type_759 = torch.ops.prims.convert_element_type.default(add_218, torch.float32);  add_218 = None
        pow_68 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_759, 2)
        mean_67 = torch.ops.aten.mean.dim(pow_68, [-1], True);  pow_68 = None
        add_219 = torch.ops.aten.add.Tensor(mean_67, 1e-05);  mean_67 = None
        rsqrt_67 = torch.ops.aten.rsqrt.default(add_219);  add_219 = None
        mul_274 = torch.ops.aten.mul.Tensor(convert_element_type_759, rsqrt_67);  convert_element_type_759 = rsqrt_67 = None
        convert_element_type_760 = torch.ops.prims.convert_element_type.default(mul_274, torch.bfloat16);  mul_274 = None
        convert_element_type_761 = torch.ops.prims.convert_element_type.default(primals_43, torch.bfloat16);  primals_43 = None
        permute_138 = torch.ops.aten.permute.default(convert_element_type_761, [1, 0]);  convert_element_type_761 = None
        view_349 = torch.ops.aten.view.default(convert_element_type_760, [115328, 128])
        mm_137 = torch.ops.aten.mm.default(view_349, permute_138);  view_349 = None
        view_350 = torch.ops.aten.view.default(mm_137, [128, 901, 384]);  mm_137 = None
        view_351 = torch.ops.aten.view.default(view_350, [128, 901, 6, 64]);  view_350 = None
        slice_448 = torch.ops.aten.slice.Tensor(view_351, 2, 0, 2)
        slice_451 = torch.ops.aten.slice.Tensor(view_351, 2, 2, 4)
        slice_454 = torch.ops.aten.slice.Tensor(view_351, 2, 4, 9223372036854775807);  view_351 = None
        convert_element_type_764 = torch.ops.prims.convert_element_type.default(slice_448, torch.float32);  slice_448 = None
        convert_element_type_765 = torch.ops.prims.convert_element_type.default(slice_451, torch.float32);  slice_451 = None
        mul_275 = torch.ops.aten.mul.Tensor(convert_element_type_764, unsqueeze_96)
        slice_455 = torch.ops.aten.slice.Tensor(convert_element_type_764, 3, 0, 32)
        slice_456 = torch.ops.aten.slice.Tensor(convert_element_type_764, 3, 32, 9223372036854775807);  convert_element_type_764 = None
        neg_68 = torch.ops.aten.neg.default(slice_456);  slice_456 = None
        cat_70 = torch.ops.aten.cat.default([neg_68, slice_455], -1);  neg_68 = slice_455 = None
        mul_276 = torch.ops.aten.mul.Tensor(cat_70, unsqueeze_97);  cat_70 = None
        add_220 = torch.ops.aten.add.Tensor(mul_275, mul_276);  mul_275 = mul_276 = None
        mul_277 = torch.ops.aten.mul.Tensor(convert_element_type_765, unsqueeze_96)
        slice_457 = torch.ops.aten.slice.Tensor(convert_element_type_765, 3, 0, 32)
        slice_458 = torch.ops.aten.slice.Tensor(convert_element_type_765, 3, 32, 9223372036854775807);  convert_element_type_765 = None
        neg_69 = torch.ops.aten.neg.default(slice_458);  slice_458 = None
        cat_71 = torch.ops.aten.cat.default([neg_69, slice_457], -1);  neg_69 = slice_457 = None
        mul_278 = torch.ops.aten.mul.Tensor(cat_71, unsqueeze_97);  cat_71 = None
        add_221 = torch.ops.aten.add.Tensor(mul_277, mul_278);  mul_277 = mul_278 = None
        convert_element_type_766 = torch.ops.prims.convert_element_type.default(add_220, torch.bfloat16);  add_220 = None
        convert_element_type_767 = torch.ops.prims.convert_element_type.default(add_221, torch.bfloat16);  add_221 = None
        _flash_attn_forward_34 = torch.ops.flash_attn._flash_attn_forward.default(convert_element_type_766, convert_element_type_767, slice_454, 0.0, 0.125, False, -1, -1, 0.0, None, False);  convert_element_type_766 = convert_element_type_767 = slice_454 = None
        getitem_204 = _flash_attn_forward_34[0];  _flash_attn_forward_34 = None
        view_352 = torch.ops.aten.view.default(getitem_204, [128, 901, 128]);  getitem_204 = None
        convert_element_type_768 = torch.ops.prims.convert_element_type.default(primals_44, torch.bfloat16);  primals_44 = None
        permute_139 = torch.ops.aten.permute.default(convert_element_type_768, [1, 0]);  convert_element_type_768 = None
        view_353 = torch.ops.aten.view.default(view_352, [115328, 128]);  view_352 = None
        mm_138 = torch.ops.aten.mm.default(view_353, permute_139);  view_353 = None
        view_354 = torch.ops.aten.view.default(mm_138, [128, 901, 128]);  mm_138 = None
        add_222 = torch.ops.aten.add.Tensor(convert_element_type_760, view_354);  convert_element_type_760 = view_354 = None
        convert_element_type_771 = torch.ops.prims.convert_element_type.default(add_222, torch.float32);  add_222 = None
        pow_69 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_771, 2)
        mean_68 = torch.ops.aten.mean.dim(pow_69, [-1], True);  pow_69 = None
        add_223 = torch.ops.aten.add.Tensor(mean_68, 1e-05);  mean_68 = None
        rsqrt_68 = torch.ops.aten.rsqrt.default(add_223);  add_223 = None
        mul_279 = torch.ops.aten.mul.Tensor(convert_element_type_771, rsqrt_68);  convert_element_type_771 = rsqrt_68 = None
        convert_element_type_772 = torch.ops.prims.convert_element_type.default(mul_279, torch.bfloat16);  mul_279 = None
        convert_element_type_773 = torch.ops.prims.convert_element_type.default(primals_45, torch.bfloat16);  primals_45 = None
        permute_140 = torch.ops.aten.permute.default(convert_element_type_773, [1, 0]);  convert_element_type_773 = None
        view_355 = torch.ops.aten.view.default(convert_element_type_772, [115328, 128])
        mm_139 = torch.ops.aten.mm.default(view_355, permute_140);  view_355 = None
        view_356 = torch.ops.aten.view.default(mm_139, [128, 901, 1024]);  mm_139 = None
        split_34 = torch.ops.aten.split.Tensor(view_356, 512, -1);  view_356 = None
        getitem_208 = split_34[0]
        getitem_209 = split_34[1];  split_34 = None
        convert_element_type_776 = torch.ops.prims.convert_element_type.default(getitem_208, torch.float32);  getitem_208 = None
        sigmoid_34 = torch.ops.aten.sigmoid.default(convert_element_type_776)
        mul_280 = torch.ops.aten.mul.Tensor(convert_element_type_776, sigmoid_34);  convert_element_type_776 = sigmoid_34 = None
        convert_element_type_777 = torch.ops.prims.convert_element_type.default(mul_280, torch.bfloat16);  mul_280 = None
        mul_281 = torch.ops.aten.mul.Tensor(convert_element_type_777, getitem_209);  convert_element_type_777 = getitem_209 = None
        convert_element_type_778 = torch.ops.prims.convert_element_type.default(primals_46, torch.bfloat16);  primals_46 = None
        permute_141 = torch.ops.aten.permute.default(convert_element_type_778, [1, 0]);  convert_element_type_778 = None
        view_357 = torch.ops.aten.view.default(mul_281, [115328, 512]);  mul_281 = None
        mm_140 = torch.ops.aten.mm.default(view_357, permute_141);  view_357 = None
        view_358 = torch.ops.aten.view.default(mm_140, [128, 901, 128]);  mm_140 = None
        add_224 = torch.ops.aten.add.Tensor(convert_element_type_772, view_358);  convert_element_type_772 = view_358 = None
        convert_element_type_781 = torch.ops.prims.convert_element_type.default(add_224, torch.float32);  add_224 = None
        pow_70 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_781, 2)
        mean_69 = torch.ops.aten.mean.dim(pow_70, [-1], True);  pow_70 = None
        add_225 = torch.ops.aten.add.Tensor(mean_69, 1e-05);  mean_69 = None
        rsqrt_69 = torch.ops.aten.rsqrt.default(add_225);  add_225 = None
        mul_282 = torch.ops.aten.mul.Tensor(convert_element_type_781, rsqrt_69);  convert_element_type_781 = rsqrt_69 = None
        convert_element_type_782 = torch.ops.prims.convert_element_type.default(mul_282, torch.bfloat16);  mul_282 = None
        convert_element_type_783 = torch.ops.prims.convert_element_type.default(primals_47, torch.bfloat16);  primals_47 = None
        permute_142 = torch.ops.aten.permute.default(convert_element_type_783, [1, 0]);  convert_element_type_783 = None
        view_359 = torch.ops.aten.view.default(convert_element_type_782, [115328, 128])
        mm_141 = torch.ops.aten.mm.default(view_359, permute_142);  view_359 = None
        view_360 = torch.ops.aten.view.default(mm_141, [128, 901, 384]);  mm_141 = None
        view_361 = torch.ops.aten.view.default(view_360, [128, 901, 6, 64]);  view_360 = None
        slice_461 = torch.ops.aten.slice.Tensor(view_361, 2, 0, 2)
        slice_464 = torch.ops.aten.slice.Tensor(view_361, 2, 2, 4)
        slice_467 = torch.ops.aten.slice.Tensor(view_361, 2, 4, 9223372036854775807);  view_361 = None
        convert_element_type_786 = torch.ops.prims.convert_element_type.default(slice_461, torch.float32);  slice_461 = None
        convert_element_type_787 = torch.ops.prims.convert_element_type.default(slice_464, torch.float32);  slice_464 = None
        mul_283 = torch.ops.aten.mul.Tensor(convert_element_type_786, unsqueeze_96)
        slice_468 = torch.ops.aten.slice.Tensor(convert_element_type_786, 3, 0, 32)
        slice_469 = torch.ops.aten.slice.Tensor(convert_element_type_786, 3, 32, 9223372036854775807);  convert_element_type_786 = None
        neg_70 = torch.ops.aten.neg.default(slice_469);  slice_469 = None
        cat_72 = torch.ops.aten.cat.default([neg_70, slice_468], -1);  neg_70 = slice_468 = None
        mul_284 = torch.ops.aten.mul.Tensor(cat_72, unsqueeze_97);  cat_72 = None
        add_226 = torch.ops.aten.add.Tensor(mul_283, mul_284);  mul_283 = mul_284 = None
        mul_285 = torch.ops.aten.mul.Tensor(convert_element_type_787, unsqueeze_96)
        slice_470 = torch.ops.aten.slice.Tensor(convert_element_type_787, 3, 0, 32)
        slice_471 = torch.ops.aten.slice.Tensor(convert_element_type_787, 3, 32, 9223372036854775807);  convert_element_type_787 = None
        neg_71 = torch.ops.aten.neg.default(slice_471);  slice_471 = None
        cat_73 = torch.ops.aten.cat.default([neg_71, slice_470], -1);  neg_71 = slice_470 = None
        mul_286 = torch.ops.aten.mul.Tensor(cat_73, unsqueeze_97);  cat_73 = None
        add_227 = torch.ops.aten.add.Tensor(mul_285, mul_286);  mul_285 = mul_286 = None
        convert_element_type_788 = torch.ops.prims.convert_element_type.default(add_226, torch.bfloat16);  add_226 = None
        convert_element_type_789 = torch.ops.prims.convert_element_type.default(add_227, torch.bfloat16);  add_227 = None
        _flash_attn_forward_35 = torch.ops.flash_attn._flash_attn_forward.default(convert_element_type_788, convert_element_type_789, slice_467, 0.0, 0.125, False, -1, -1, 0.0, None, False);  convert_element_type_788 = convert_element_type_789 = slice_467 = None
        getitem_210 = _flash_attn_forward_35[0];  _flash_attn_forward_35 = None
        view_362 = torch.ops.aten.view.default(getitem_210, [128, 901, 128]);  getitem_210 = None
        convert_element_type_790 = torch.ops.prims.convert_element_type.default(primals_48, torch.bfloat16);  primals_48 = None
        permute_143 = torch.ops.aten.permute.default(convert_element_type_790, [1, 0]);  convert_element_type_790 = None
        view_363 = torch.ops.aten.view.default(view_362, [115328, 128]);  view_362 = None
        mm_142 = torch.ops.aten.mm.default(view_363, permute_143);  view_363 = None
        view_364 = torch.ops.aten.view.default(mm_142, [128, 901, 128]);  mm_142 = None
        add_228 = torch.ops.aten.add.Tensor(convert_element_type_782, view_364);  convert_element_type_782 = view_364 = None
        convert_element_type_793 = torch.ops.prims.convert_element_type.default(add_228, torch.float32);  add_228 = None
        pow_71 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_793, 2)
        mean_70 = torch.ops.aten.mean.dim(pow_71, [-1], True);  pow_71 = None
        add_229 = torch.ops.aten.add.Tensor(mean_70, 1e-05);  mean_70 = None
        rsqrt_70 = torch.ops.aten.rsqrt.default(add_229);  add_229 = None
        mul_287 = torch.ops.aten.mul.Tensor(convert_element_type_793, rsqrt_70);  convert_element_type_793 = rsqrt_70 = None
        convert_element_type_794 = torch.ops.prims.convert_element_type.default(mul_287, torch.bfloat16);  mul_287 = None
        convert_element_type_795 = torch.ops.prims.convert_element_type.default(primals_49, torch.bfloat16);  primals_49 = None
        permute_144 = torch.ops.aten.permute.default(convert_element_type_795, [1, 0]);  convert_element_type_795 = None
        view_365 = torch.ops.aten.view.default(convert_element_type_794, [115328, 128])
        mm_143 = torch.ops.aten.mm.default(view_365, permute_144);  view_365 = None
        view_366 = torch.ops.aten.view.default(mm_143, [128, 901, 1024]);  mm_143 = None
        split_35 = torch.ops.aten.split.Tensor(view_366, 512, -1);  view_366 = None
        getitem_214 = split_35[0]
        getitem_215 = split_35[1];  split_35 = None
        convert_element_type_798 = torch.ops.prims.convert_element_type.default(getitem_214, torch.float32);  getitem_214 = None
        sigmoid_35 = torch.ops.aten.sigmoid.default(convert_element_type_798)
        mul_288 = torch.ops.aten.mul.Tensor(convert_element_type_798, sigmoid_35);  convert_element_type_798 = sigmoid_35 = None
        convert_element_type_799 = torch.ops.prims.convert_element_type.default(mul_288, torch.bfloat16);  mul_288 = None
        mul_289 = torch.ops.aten.mul.Tensor(convert_element_type_799, getitem_215);  convert_element_type_799 = getitem_215 = None
        convert_element_type_800 = torch.ops.prims.convert_element_type.default(primals_50, torch.bfloat16);  primals_50 = None
        permute_145 = torch.ops.aten.permute.default(convert_element_type_800, [1, 0]);  convert_element_type_800 = None
        view_367 = torch.ops.aten.view.default(mul_289, [115328, 512]);  mul_289 = None
        mm_144 = torch.ops.aten.mm.default(view_367, permute_145);  view_367 = None
        view_368 = torch.ops.aten.view.default(mm_144, [128, 901, 128]);  mm_144 = None
        add_230 = torch.ops.aten.add.Tensor(convert_element_type_794, view_368);  convert_element_type_794 = view_368 = None
        convert_element_type_803 = torch.ops.prims.convert_element_type.default(add_230, torch.float32);  add_230 = None
        pow_72 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_803, 2)
        mean_71 = torch.ops.aten.mean.dim(pow_72, [-1], True);  pow_72 = None
        add_231 = torch.ops.aten.add.Tensor(mean_71, 1e-05);  mean_71 = None
        rsqrt_71 = torch.ops.aten.rsqrt.default(add_231);  add_231 = None
        mul_290 = torch.ops.aten.mul.Tensor(convert_element_type_803, rsqrt_71);  convert_element_type_803 = rsqrt_71 = None
        convert_element_type_804 = torch.ops.prims.convert_element_type.default(mul_290, torch.bfloat16);  mul_290 = None
        add_232 = torch.ops.aten.add.Tensor(convert_element_type_804, mul_194);  mul_194 = None
        add_233 = torch.ops.aten.add.Tensor(convert_element_type_716, add_232);  convert_element_type_716 = None
        view_369 = torch.ops.aten.view.default(add_233, [115328, 128])
        mm_145 = torch.ops.aten.mm.default(view_369, permute_98);  view_369 = None
        view_370 = torch.ops.aten.view.default(mm_145, [128, 901, 384]);  mm_145 = None
        view_371 = torch.ops.aten.view.default(view_370, [128, 901, 6, 64]);  view_370 = None
        slice_474 = torch.ops.aten.slice.Tensor(view_371, 2, 0, 2)
        slice_477 = torch.ops.aten.slice.Tensor(view_371, 2, 2, 4)
        slice_480 = torch.ops.aten.slice.Tensor(view_371, 2, 4, 9223372036854775807);  view_371 = None
        convert_element_type_808 = torch.ops.prims.convert_element_type.default(slice_474, torch.float32);  slice_474 = None
        convert_element_type_809 = torch.ops.prims.convert_element_type.default(slice_477, torch.float32);  slice_477 = None
        mul_291 = torch.ops.aten.mul.Tensor(convert_element_type_808, unsqueeze_96)
        slice_481 = torch.ops.aten.slice.Tensor(convert_element_type_808, 3, 0, 32)
        slice_482 = torch.ops.aten.slice.Tensor(convert_element_type_808, 3, 32, 9223372036854775807);  convert_element_type_808 = None
        neg_72 = torch.ops.aten.neg.default(slice_482);  slice_482 = None
        cat_74 = torch.ops.aten.cat.default([neg_72, slice_481], -1);  neg_72 = slice_481 = None
        mul_292 = torch.ops.aten.mul.Tensor(cat_74, unsqueeze_97);  cat_74 = None
        add_234 = torch.ops.aten.add.Tensor(mul_291, mul_292);  mul_291 = mul_292 = None
        mul_293 = torch.ops.aten.mul.Tensor(convert_element_type_809, unsqueeze_96)
        slice_483 = torch.ops.aten.slice.Tensor(convert_element_type_809, 3, 0, 32)
        slice_484 = torch.ops.aten.slice.Tensor(convert_element_type_809, 3, 32, 9223372036854775807);  convert_element_type_809 = None
        neg_73 = torch.ops.aten.neg.default(slice_484);  slice_484 = None
        cat_75 = torch.ops.aten.cat.default([neg_73, slice_483], -1);  neg_73 = slice_483 = None
        mul_294 = torch.ops.aten.mul.Tensor(cat_75, unsqueeze_97);  cat_75 = None
        add_235 = torch.ops.aten.add.Tensor(mul_293, mul_294);  mul_293 = mul_294 = None
        convert_element_type_810 = torch.ops.prims.convert_element_type.default(add_234, torch.bfloat16);  add_234 = None
        convert_element_type_811 = torch.ops.prims.convert_element_type.default(add_235, torch.bfloat16);  add_235 = None
        _flash_attn_forward_36 = torch.ops.flash_attn._flash_attn_forward.default(convert_element_type_810, convert_element_type_811, slice_480, 0.0, 0.125, False, -1, -1, 0.0, None, False);  convert_element_type_810 = convert_element_type_811 = slice_480 = None
        getitem_216 = _flash_attn_forward_36[0];  _flash_attn_forward_36 = None
        view_372 = torch.ops.aten.view.default(getitem_216, [128, 901, 128]);  getitem_216 = None
        view_373 = torch.ops.aten.view.default(view_372, [115328, 128]);  view_372 = None
        mm_146 = torch.ops.aten.mm.default(view_373, permute_99);  view_373 = None
        view_374 = torch.ops.aten.view.default(mm_146, [128, 901, 128]);  mm_146 = None
        add_236 = torch.ops.aten.add.Tensor(add_233, view_374);  add_233 = view_374 = None
        convert_element_type_815 = torch.ops.prims.convert_element_type.default(add_236, torch.float32);  add_236 = None
        pow_73 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_815, 2)
        mean_72 = torch.ops.aten.mean.dim(pow_73, [-1], True);  pow_73 = None
        add_237 = torch.ops.aten.add.Tensor(mean_72, 1e-05);  mean_72 = None
        rsqrt_72 = torch.ops.aten.rsqrt.default(add_237);  add_237 = None
        mul_295 = torch.ops.aten.mul.Tensor(convert_element_type_815, rsqrt_72);  convert_element_type_815 = rsqrt_72 = None
        convert_element_type_816 = torch.ops.prims.convert_element_type.default(mul_295, torch.bfloat16);  mul_295 = None
        view_375 = torch.ops.aten.view.default(convert_element_type_816, [115328, 128])
        mm_147 = torch.ops.aten.mm.default(view_375, permute_100);  view_375 = None
        view_376 = torch.ops.aten.view.default(mm_147, [128, 901, 1024]);  mm_147 = None
        split_36 = torch.ops.aten.split.Tensor(view_376, 512, -1);  view_376 = None
        getitem_220 = split_36[0]
        getitem_221 = split_36[1];  split_36 = None
        convert_element_type_820 = torch.ops.prims.convert_element_type.default(getitem_220, torch.float32);  getitem_220 = None
        sigmoid_36 = torch.ops.aten.sigmoid.default(convert_element_type_820)
        mul_296 = torch.ops.aten.mul.Tensor(convert_element_type_820, sigmoid_36);  convert_element_type_820 = sigmoid_36 = None
        convert_element_type_821 = torch.ops.prims.convert_element_type.default(mul_296, torch.bfloat16);  mul_296 = None
        mul_297 = torch.ops.aten.mul.Tensor(convert_element_type_821, getitem_221);  convert_element_type_821 = getitem_221 = None
        view_377 = torch.ops.aten.view.default(mul_297, [115328, 512]);  mul_297 = None
        mm_148 = torch.ops.aten.mm.default(view_377, permute_101);  view_377 = None
        view_378 = torch.ops.aten.view.default(mm_148, [128, 901, 128]);  mm_148 = None
        add_238 = torch.ops.aten.add.Tensor(convert_element_type_816, view_378);  convert_element_type_816 = view_378 = None
        convert_element_type_825 = torch.ops.prims.convert_element_type.default(add_238, torch.float32);  add_238 = None
        pow_74 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_825, 2)
        mean_73 = torch.ops.aten.mean.dim(pow_74, [-1], True);  pow_74 = None
        add_239 = torch.ops.aten.add.Tensor(mean_73, 1e-05);  mean_73 = None
        rsqrt_73 = torch.ops.aten.rsqrt.default(add_239);  add_239 = None
        mul_298 = torch.ops.aten.mul.Tensor(convert_element_type_825, rsqrt_73);  convert_element_type_825 = rsqrt_73 = None
        convert_element_type_826 = torch.ops.prims.convert_element_type.default(mul_298, torch.bfloat16);  mul_298 = None
        view_379 = torch.ops.aten.view.default(convert_element_type_826, [115328, 128])
        mm_149 = torch.ops.aten.mm.default(view_379, permute_102);  view_379 = None
        view_380 = torch.ops.aten.view.default(mm_149, [128, 901, 384]);  mm_149 = None
        view_381 = torch.ops.aten.view.default(view_380, [128, 901, 6, 64]);  view_380 = None
        slice_487 = torch.ops.aten.slice.Tensor(view_381, 2, 0, 2)
        slice_490 = torch.ops.aten.slice.Tensor(view_381, 2, 2, 4)
        slice_493 = torch.ops.aten.slice.Tensor(view_381, 2, 4, 9223372036854775807);  view_381 = None
        convert_element_type_830 = torch.ops.prims.convert_element_type.default(slice_487, torch.float32);  slice_487 = None
        convert_element_type_831 = torch.ops.prims.convert_element_type.default(slice_490, torch.float32);  slice_490 = None
        mul_299 = torch.ops.aten.mul.Tensor(convert_element_type_830, unsqueeze_96)
        slice_494 = torch.ops.aten.slice.Tensor(convert_element_type_830, 3, 0, 32)
        slice_495 = torch.ops.aten.slice.Tensor(convert_element_type_830, 3, 32, 9223372036854775807);  convert_element_type_830 = None
        neg_74 = torch.ops.aten.neg.default(slice_495);  slice_495 = None
        cat_76 = torch.ops.aten.cat.default([neg_74, slice_494], -1);  neg_74 = slice_494 = None
        mul_300 = torch.ops.aten.mul.Tensor(cat_76, unsqueeze_97);  cat_76 = None
        add_240 = torch.ops.aten.add.Tensor(mul_299, mul_300);  mul_299 = mul_300 = None
        mul_301 = torch.ops.aten.mul.Tensor(convert_element_type_831, unsqueeze_96)
        slice_496 = torch.ops.aten.slice.Tensor(convert_element_type_831, 3, 0, 32)
        slice_497 = torch.ops.aten.slice.Tensor(convert_element_type_831, 3, 32, 9223372036854775807);  convert_element_type_831 = None
        neg_75 = torch.ops.aten.neg.default(slice_497);  slice_497 = None
        cat_77 = torch.ops.aten.cat.default([neg_75, slice_496], -1);  neg_75 = slice_496 = None
        mul_302 = torch.ops.aten.mul.Tensor(cat_77, unsqueeze_97);  cat_77 = None
        add_241 = torch.ops.aten.add.Tensor(mul_301, mul_302);  mul_301 = mul_302 = None
        convert_element_type_832 = torch.ops.prims.convert_element_type.default(add_240, torch.bfloat16);  add_240 = None
        convert_element_type_833 = torch.ops.prims.convert_element_type.default(add_241, torch.bfloat16);  add_241 = None
        _flash_attn_forward_37 = torch.ops.flash_attn._flash_attn_forward.default(convert_element_type_832, convert_element_type_833, slice_493, 0.0, 0.125, False, -1, -1, 0.0, None, False);  convert_element_type_832 = convert_element_type_833 = slice_493 = None
        getitem_222 = _flash_attn_forward_37[0];  _flash_attn_forward_37 = None
        view_382 = torch.ops.aten.view.default(getitem_222, [128, 901, 128]);  getitem_222 = None
        view_383 = torch.ops.aten.view.default(view_382, [115328, 128]);  view_382 = None
        mm_150 = torch.ops.aten.mm.default(view_383, permute_103);  view_383 = None
        view_384 = torch.ops.aten.view.default(mm_150, [128, 901, 128]);  mm_150 = None
        add_242 = torch.ops.aten.add.Tensor(convert_element_type_826, view_384);  convert_element_type_826 = view_384 = None
        convert_element_type_837 = torch.ops.prims.convert_element_type.default(add_242, torch.float32);  add_242 = None
        pow_75 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_837, 2)
        mean_74 = torch.ops.aten.mean.dim(pow_75, [-1], True);  pow_75 = None
        add_243 = torch.ops.aten.add.Tensor(mean_74, 1e-05);  mean_74 = None
        rsqrt_74 = torch.ops.aten.rsqrt.default(add_243);  add_243 = None
        mul_303 = torch.ops.aten.mul.Tensor(convert_element_type_837, rsqrt_74);  convert_element_type_837 = rsqrt_74 = None
        convert_element_type_838 = torch.ops.prims.convert_element_type.default(mul_303, torch.bfloat16);  mul_303 = None
        view_385 = torch.ops.aten.view.default(convert_element_type_838, [115328, 128])
        mm_151 = torch.ops.aten.mm.default(view_385, permute_104);  view_385 = None
        view_386 = torch.ops.aten.view.default(mm_151, [128, 901, 1024]);  mm_151 = None
        split_37 = torch.ops.aten.split.Tensor(view_386, 512, -1);  view_386 = None
        getitem_226 = split_37[0]
        getitem_227 = split_37[1];  split_37 = None
        convert_element_type_842 = torch.ops.prims.convert_element_type.default(getitem_226, torch.float32);  getitem_226 = None
        sigmoid_37 = torch.ops.aten.sigmoid.default(convert_element_type_842)
        mul_304 = torch.ops.aten.mul.Tensor(convert_element_type_842, sigmoid_37);  convert_element_type_842 = sigmoid_37 = None
        convert_element_type_843 = torch.ops.prims.convert_element_type.default(mul_304, torch.bfloat16);  mul_304 = None
        mul_305 = torch.ops.aten.mul.Tensor(convert_element_type_843, getitem_227);  convert_element_type_843 = getitem_227 = None
        view_387 = torch.ops.aten.view.default(mul_305, [115328, 512]);  mul_305 = None
        mm_152 = torch.ops.aten.mm.default(view_387, permute_105);  view_387 = None
        view_388 = torch.ops.aten.view.default(mm_152, [128, 901, 128]);  mm_152 = None
        add_244 = torch.ops.aten.add.Tensor(convert_element_type_838, view_388);  convert_element_type_838 = view_388 = None
        convert_element_type_847 = torch.ops.prims.convert_element_type.default(add_244, torch.float32);  add_244 = None
        pow_76 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_847, 2)
        mean_75 = torch.ops.aten.mean.dim(pow_76, [-1], True);  pow_76 = None
        add_245 = torch.ops.aten.add.Tensor(mean_75, 1e-05);  mean_75 = None
        rsqrt_75 = torch.ops.aten.rsqrt.default(add_245);  add_245 = None
        mul_306 = torch.ops.aten.mul.Tensor(convert_element_type_847, rsqrt_75);  convert_element_type_847 = rsqrt_75 = None
        convert_element_type_848 = torch.ops.prims.convert_element_type.default(mul_306, torch.bfloat16);  mul_306 = None
        view_389 = torch.ops.aten.view.default(convert_element_type_848, [115328, 128])
        mm_153 = torch.ops.aten.mm.default(view_389, permute_106);  view_389 = None
        view_390 = torch.ops.aten.view.default(mm_153, [128, 901, 384]);  mm_153 = None
        view_391 = torch.ops.aten.view.default(view_390, [128, 901, 6, 64]);  view_390 = None
        slice_500 = torch.ops.aten.slice.Tensor(view_391, 2, 0, 2)
        slice_503 = torch.ops.aten.slice.Tensor(view_391, 2, 2, 4)
        slice_506 = torch.ops.aten.slice.Tensor(view_391, 2, 4, 9223372036854775807);  view_391 = None
        convert_element_type_852 = torch.ops.prims.convert_element_type.default(slice_500, torch.float32);  slice_500 = None
        convert_element_type_853 = torch.ops.prims.convert_element_type.default(slice_503, torch.float32);  slice_503 = None
        mul_307 = torch.ops.aten.mul.Tensor(convert_element_type_852, unsqueeze_96)
        slice_507 = torch.ops.aten.slice.Tensor(convert_element_type_852, 3, 0, 32)
        slice_508 = torch.ops.aten.slice.Tensor(convert_element_type_852, 3, 32, 9223372036854775807);  convert_element_type_852 = None
        neg_76 = torch.ops.aten.neg.default(slice_508);  slice_508 = None
        cat_78 = torch.ops.aten.cat.default([neg_76, slice_507], -1);  neg_76 = slice_507 = None
        mul_308 = torch.ops.aten.mul.Tensor(cat_78, unsqueeze_97);  cat_78 = None
        add_246 = torch.ops.aten.add.Tensor(mul_307, mul_308);  mul_307 = mul_308 = None
        mul_309 = torch.ops.aten.mul.Tensor(convert_element_type_853, unsqueeze_96)
        slice_509 = torch.ops.aten.slice.Tensor(convert_element_type_853, 3, 0, 32)
        slice_510 = torch.ops.aten.slice.Tensor(convert_element_type_853, 3, 32, 9223372036854775807);  convert_element_type_853 = None
        neg_77 = torch.ops.aten.neg.default(slice_510);  slice_510 = None
        cat_79 = torch.ops.aten.cat.default([neg_77, slice_509], -1);  neg_77 = slice_509 = None
        mul_310 = torch.ops.aten.mul.Tensor(cat_79, unsqueeze_97);  cat_79 = None
        add_247 = torch.ops.aten.add.Tensor(mul_309, mul_310);  mul_309 = mul_310 = None
        convert_element_type_854 = torch.ops.prims.convert_element_type.default(add_246, torch.bfloat16);  add_246 = None
        convert_element_type_855 = torch.ops.prims.convert_element_type.default(add_247, torch.bfloat16);  add_247 = None
        _flash_attn_forward_38 = torch.ops.flash_attn._flash_attn_forward.default(convert_element_type_854, convert_element_type_855, slice_506, 0.0, 0.125, False, -1, -1, 0.0, None, False);  convert_element_type_854 = convert_element_type_855 = slice_506 = None
        getitem_228 = _flash_attn_forward_38[0];  _flash_attn_forward_38 = None
        view_392 = torch.ops.aten.view.default(getitem_228, [128, 901, 128]);  getitem_228 = None
        view_393 = torch.ops.aten.view.default(view_392, [115328, 128]);  view_392 = None
        mm_154 = torch.ops.aten.mm.default(view_393, permute_107);  view_393 = None
        view_394 = torch.ops.aten.view.default(mm_154, [128, 901, 128]);  mm_154 = None
        add_248 = torch.ops.aten.add.Tensor(convert_element_type_848, view_394);  convert_element_type_848 = view_394 = None
        convert_element_type_859 = torch.ops.prims.convert_element_type.default(add_248, torch.float32);  add_248 = None
        pow_77 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_859, 2)
        mean_76 = torch.ops.aten.mean.dim(pow_77, [-1], True);  pow_77 = None
        add_249 = torch.ops.aten.add.Tensor(mean_76, 1e-05);  mean_76 = None
        rsqrt_76 = torch.ops.aten.rsqrt.default(add_249);  add_249 = None
        mul_311 = torch.ops.aten.mul.Tensor(convert_element_type_859, rsqrt_76);  convert_element_type_859 = rsqrt_76 = None
        convert_element_type_860 = torch.ops.prims.convert_element_type.default(mul_311, torch.bfloat16);  mul_311 = None
        view_395 = torch.ops.aten.view.default(convert_element_type_860, [115328, 128])
        mm_155 = torch.ops.aten.mm.default(view_395, permute_108);  view_395 = None
        view_396 = torch.ops.aten.view.default(mm_155, [128, 901, 1024]);  mm_155 = None
        split_38 = torch.ops.aten.split.Tensor(view_396, 512, -1);  view_396 = None
        getitem_232 = split_38[0]
        getitem_233 = split_38[1];  split_38 = None
        convert_element_type_864 = torch.ops.prims.convert_element_type.default(getitem_232, torch.float32);  getitem_232 = None
        sigmoid_38 = torch.ops.aten.sigmoid.default(convert_element_type_864)
        mul_312 = torch.ops.aten.mul.Tensor(convert_element_type_864, sigmoid_38);  convert_element_type_864 = sigmoid_38 = None
        convert_element_type_865 = torch.ops.prims.convert_element_type.default(mul_312, torch.bfloat16);  mul_312 = None
        mul_313 = torch.ops.aten.mul.Tensor(convert_element_type_865, getitem_233);  convert_element_type_865 = getitem_233 = None
        view_397 = torch.ops.aten.view.default(mul_313, [115328, 512]);  mul_313 = None
        mm_156 = torch.ops.aten.mm.default(view_397, permute_109);  view_397 = None
        view_398 = torch.ops.aten.view.default(mm_156, [128, 901, 128]);  mm_156 = None
        add_250 = torch.ops.aten.add.Tensor(convert_element_type_860, view_398);  convert_element_type_860 = view_398 = None
        convert_element_type_869 = torch.ops.prims.convert_element_type.default(add_250, torch.float32);  add_250 = None
        pow_78 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_869, 2)
        mean_77 = torch.ops.aten.mean.dim(pow_78, [-1], True);  pow_78 = None
        add_251 = torch.ops.aten.add.Tensor(mean_77, 1e-05);  mean_77 = None
        rsqrt_77 = torch.ops.aten.rsqrt.default(add_251);  add_251 = None
        mul_314 = torch.ops.aten.mul.Tensor(convert_element_type_869, rsqrt_77);  convert_element_type_869 = rsqrt_77 = None
        convert_element_type_870 = torch.ops.prims.convert_element_type.default(mul_314, torch.bfloat16);  mul_314 = None
        view_399 = torch.ops.aten.view.default(convert_element_type_870, [115328, 128])
        mm_157 = torch.ops.aten.mm.default(view_399, permute_110);  view_399 = None
        view_400 = torch.ops.aten.view.default(mm_157, [128, 901, 384]);  mm_157 = None
        view_401 = torch.ops.aten.view.default(view_400, [128, 901, 6, 64]);  view_400 = None
        slice_513 = torch.ops.aten.slice.Tensor(view_401, 2, 0, 2)
        slice_516 = torch.ops.aten.slice.Tensor(view_401, 2, 2, 4)
        slice_519 = torch.ops.aten.slice.Tensor(view_401, 2, 4, 9223372036854775807);  view_401 = None
        convert_element_type_874 = torch.ops.prims.convert_element_type.default(slice_513, torch.float32);  slice_513 = None
        convert_element_type_875 = torch.ops.prims.convert_element_type.default(slice_516, torch.float32);  slice_516 = None
        mul_315 = torch.ops.aten.mul.Tensor(convert_element_type_874, unsqueeze_96)
        slice_520 = torch.ops.aten.slice.Tensor(convert_element_type_874, 3, 0, 32)
        slice_521 = torch.ops.aten.slice.Tensor(convert_element_type_874, 3, 32, 9223372036854775807);  convert_element_type_874 = None
        neg_78 = torch.ops.aten.neg.default(slice_521);  slice_521 = None
        cat_80 = torch.ops.aten.cat.default([neg_78, slice_520], -1);  neg_78 = slice_520 = None
        mul_316 = torch.ops.aten.mul.Tensor(cat_80, unsqueeze_97);  cat_80 = None
        add_252 = torch.ops.aten.add.Tensor(mul_315, mul_316);  mul_315 = mul_316 = None
        mul_317 = torch.ops.aten.mul.Tensor(convert_element_type_875, unsqueeze_96)
        slice_522 = torch.ops.aten.slice.Tensor(convert_element_type_875, 3, 0, 32)
        slice_523 = torch.ops.aten.slice.Tensor(convert_element_type_875, 3, 32, 9223372036854775807);  convert_element_type_875 = None
        neg_79 = torch.ops.aten.neg.default(slice_523);  slice_523 = None
        cat_81 = torch.ops.aten.cat.default([neg_79, slice_522], -1);  neg_79 = slice_522 = None
        mul_318 = torch.ops.aten.mul.Tensor(cat_81, unsqueeze_97);  cat_81 = None
        add_253 = torch.ops.aten.add.Tensor(mul_317, mul_318);  mul_317 = mul_318 = None
        convert_element_type_876 = torch.ops.prims.convert_element_type.default(add_252, torch.bfloat16);  add_252 = None
        convert_element_type_877 = torch.ops.prims.convert_element_type.default(add_253, torch.bfloat16);  add_253 = None
        _flash_attn_forward_39 = torch.ops.flash_attn._flash_attn_forward.default(convert_element_type_876, convert_element_type_877, slice_519, 0.0, 0.125, False, -1, -1, 0.0, None, False);  convert_element_type_876 = convert_element_type_877 = slice_519 = None
        getitem_234 = _flash_attn_forward_39[0];  _flash_attn_forward_39 = None
        view_402 = torch.ops.aten.view.default(getitem_234, [128, 901, 128]);  getitem_234 = None
        view_403 = torch.ops.aten.view.default(view_402, [115328, 128]);  view_402 = None
        mm_158 = torch.ops.aten.mm.default(view_403, permute_111);  view_403 = None
        view_404 = torch.ops.aten.view.default(mm_158, [128, 901, 128]);  mm_158 = None
        add_254 = torch.ops.aten.add.Tensor(convert_element_type_870, view_404);  convert_element_type_870 = view_404 = None
        convert_element_type_881 = torch.ops.prims.convert_element_type.default(add_254, torch.float32);  add_254 = None
        pow_79 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_881, 2)
        mean_78 = torch.ops.aten.mean.dim(pow_79, [-1], True);  pow_79 = None
        add_255 = torch.ops.aten.add.Tensor(mean_78, 1e-05);  mean_78 = None
        rsqrt_78 = torch.ops.aten.rsqrt.default(add_255);  add_255 = None
        mul_319 = torch.ops.aten.mul.Tensor(convert_element_type_881, rsqrt_78);  convert_element_type_881 = rsqrt_78 = None
        convert_element_type_882 = torch.ops.prims.convert_element_type.default(mul_319, torch.bfloat16);  mul_319 = None
        view_405 = torch.ops.aten.view.default(convert_element_type_882, [115328, 128])
        mm_159 = torch.ops.aten.mm.default(view_405, permute_112);  view_405 = None
        view_406 = torch.ops.aten.view.default(mm_159, [128, 901, 1024]);  mm_159 = None
        split_39 = torch.ops.aten.split.Tensor(view_406, 512, -1);  view_406 = None
        getitem_238 = split_39[0]
        getitem_239 = split_39[1];  split_39 = None
        convert_element_type_886 = torch.ops.prims.convert_element_type.default(getitem_238, torch.float32);  getitem_238 = None
        sigmoid_39 = torch.ops.aten.sigmoid.default(convert_element_type_886)
        mul_320 = torch.ops.aten.mul.Tensor(convert_element_type_886, sigmoid_39);  convert_element_type_886 = sigmoid_39 = None
        convert_element_type_887 = torch.ops.prims.convert_element_type.default(mul_320, torch.bfloat16);  mul_320 = None
        mul_321 = torch.ops.aten.mul.Tensor(convert_element_type_887, getitem_239);  convert_element_type_887 = getitem_239 = None
        view_407 = torch.ops.aten.view.default(mul_321, [115328, 512]);  mul_321 = None
        mm_160 = torch.ops.aten.mm.default(view_407, permute_113);  view_407 = None
        view_408 = torch.ops.aten.view.default(mm_160, [128, 901, 128]);  mm_160 = None
        add_256 = torch.ops.aten.add.Tensor(convert_element_type_882, view_408);  convert_element_type_882 = view_408 = None
        convert_element_type_891 = torch.ops.prims.convert_element_type.default(add_256, torch.float32);  add_256 = None
        pow_80 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_891, 2)
        mean_79 = torch.ops.aten.mean.dim(pow_80, [-1], True);  pow_80 = None
        add_257 = torch.ops.aten.add.Tensor(mean_79, 1e-05);  mean_79 = None
        rsqrt_79 = torch.ops.aten.rsqrt.default(add_257);  add_257 = None
        mul_322 = torch.ops.aten.mul.Tensor(convert_element_type_891, rsqrt_79);  convert_element_type_891 = rsqrt_79 = None
        convert_element_type_892 = torch.ops.prims.convert_element_type.default(mul_322, torch.bfloat16);  mul_322 = None
        add_259 = torch.ops.aten.add.Tensor(convert_element_type_892, add_232);  convert_element_type_892 = add_232 = None
        view_409 = torch.ops.aten.view.default(add_259, [115328, 128])
        mm_161 = torch.ops.aten.mm.default(view_409, permute_98);  view_409 = permute_98 = None
        view_410 = torch.ops.aten.view.default(mm_161, [128, 901, 384]);  mm_161 = None
        view_411 = torch.ops.aten.view.default(view_410, [128, 901, 6, 64]);  view_410 = None
        slice_526 = torch.ops.aten.slice.Tensor(view_411, 2, 0, 2)
        slice_529 = torch.ops.aten.slice.Tensor(view_411, 2, 2, 4)
        slice_532 = torch.ops.aten.slice.Tensor(view_411, 2, 4, 9223372036854775807);  view_411 = None
        convert_element_type_896 = torch.ops.prims.convert_element_type.default(slice_526, torch.float32);  slice_526 = None
        convert_element_type_897 = torch.ops.prims.convert_element_type.default(slice_529, torch.float32);  slice_529 = None
        mul_323 = torch.ops.aten.mul.Tensor(convert_element_type_896, unsqueeze_96)
        slice_533 = torch.ops.aten.slice.Tensor(convert_element_type_896, 3, 0, 32)
        slice_534 = torch.ops.aten.slice.Tensor(convert_element_type_896, 3, 32, 9223372036854775807);  convert_element_type_896 = None
        neg_80 = torch.ops.aten.neg.default(slice_534);  slice_534 = None
        cat_82 = torch.ops.aten.cat.default([neg_80, slice_533], -1);  neg_80 = slice_533 = None
        mul_324 = torch.ops.aten.mul.Tensor(cat_82, unsqueeze_97);  cat_82 = None
        add_260 = torch.ops.aten.add.Tensor(mul_323, mul_324);  mul_323 = mul_324 = None
        mul_325 = torch.ops.aten.mul.Tensor(convert_element_type_897, unsqueeze_96)
        slice_535 = torch.ops.aten.slice.Tensor(convert_element_type_897, 3, 0, 32)
        slice_536 = torch.ops.aten.slice.Tensor(convert_element_type_897, 3, 32, 9223372036854775807);  convert_element_type_897 = None
        neg_81 = torch.ops.aten.neg.default(slice_536);  slice_536 = None
        cat_83 = torch.ops.aten.cat.default([neg_81, slice_535], -1);  neg_81 = slice_535 = None
        mul_326 = torch.ops.aten.mul.Tensor(cat_83, unsqueeze_97);  cat_83 = None
        add_261 = torch.ops.aten.add.Tensor(mul_325, mul_326);  mul_325 = mul_326 = None
        convert_element_type_898 = torch.ops.prims.convert_element_type.default(add_260, torch.bfloat16);  add_260 = None
        convert_element_type_899 = torch.ops.prims.convert_element_type.default(add_261, torch.bfloat16);  add_261 = None
        _flash_attn_forward_40 = torch.ops.flash_attn._flash_attn_forward.default(convert_element_type_898, convert_element_type_899, slice_532, 0.0, 0.125, False, -1, -1, 0.0, None, False);  convert_element_type_898 = convert_element_type_899 = slice_532 = None
        getitem_240 = _flash_attn_forward_40[0];  _flash_attn_forward_40 = None
        view_412 = torch.ops.aten.view.default(getitem_240, [128, 901, 128]);  getitem_240 = None
        view_413 = torch.ops.aten.view.default(view_412, [115328, 128]);  view_412 = None
        mm_162 = torch.ops.aten.mm.default(view_413, permute_99);  view_413 = permute_99 = None
        view_414 = torch.ops.aten.view.default(mm_162, [128, 901, 128]);  mm_162 = None
        add_262 = torch.ops.aten.add.Tensor(add_259, view_414);  add_259 = view_414 = None
        convert_element_type_903 = torch.ops.prims.convert_element_type.default(add_262, torch.float32);  add_262 = None
        pow_81 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_903, 2)
        mean_80 = torch.ops.aten.mean.dim(pow_81, [-1], True);  pow_81 = None
        add_263 = torch.ops.aten.add.Tensor(mean_80, 1e-05);  mean_80 = None
        rsqrt_80 = torch.ops.aten.rsqrt.default(add_263);  add_263 = None
        mul_327 = torch.ops.aten.mul.Tensor(convert_element_type_903, rsqrt_80);  convert_element_type_903 = rsqrt_80 = None
        convert_element_type_904 = torch.ops.prims.convert_element_type.default(mul_327, torch.bfloat16);  mul_327 = None
        view_415 = torch.ops.aten.view.default(convert_element_type_904, [115328, 128])
        mm_163 = torch.ops.aten.mm.default(view_415, permute_100);  view_415 = permute_100 = None
        view_416 = torch.ops.aten.view.default(mm_163, [128, 901, 1024]);  mm_163 = None
        split_40 = torch.ops.aten.split.Tensor(view_416, 512, -1);  view_416 = None
        getitem_244 = split_40[0]
        getitem_245 = split_40[1];  split_40 = None
        convert_element_type_908 = torch.ops.prims.convert_element_type.default(getitem_244, torch.float32);  getitem_244 = None
        sigmoid_40 = torch.ops.aten.sigmoid.default(convert_element_type_908)
        mul_328 = torch.ops.aten.mul.Tensor(convert_element_type_908, sigmoid_40);  convert_element_type_908 = sigmoid_40 = None
        convert_element_type_909 = torch.ops.prims.convert_element_type.default(mul_328, torch.bfloat16);  mul_328 = None
        mul_329 = torch.ops.aten.mul.Tensor(convert_element_type_909, getitem_245);  convert_element_type_909 = getitem_245 = None
        view_417 = torch.ops.aten.view.default(mul_329, [115328, 512]);  mul_329 = None
        mm_164 = torch.ops.aten.mm.default(view_417, permute_101);  view_417 = permute_101 = None
        view_418 = torch.ops.aten.view.default(mm_164, [128, 901, 128]);  mm_164 = None
        add_264 = torch.ops.aten.add.Tensor(convert_element_type_904, view_418);  convert_element_type_904 = view_418 = None
        convert_element_type_913 = torch.ops.prims.convert_element_type.default(add_264, torch.float32);  add_264 = None
        pow_82 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_913, 2)
        mean_81 = torch.ops.aten.mean.dim(pow_82, [-1], True);  pow_82 = None
        add_265 = torch.ops.aten.add.Tensor(mean_81, 1e-05);  mean_81 = None
        rsqrt_81 = torch.ops.aten.rsqrt.default(add_265);  add_265 = None
        mul_330 = torch.ops.aten.mul.Tensor(convert_element_type_913, rsqrt_81);  convert_element_type_913 = rsqrt_81 = None
        convert_element_type_914 = torch.ops.prims.convert_element_type.default(mul_330, torch.bfloat16);  mul_330 = None
        view_419 = torch.ops.aten.view.default(convert_element_type_914, [115328, 128])
        mm_165 = torch.ops.aten.mm.default(view_419, permute_102);  view_419 = permute_102 = None
        view_420 = torch.ops.aten.view.default(mm_165, [128, 901, 384]);  mm_165 = None
        view_421 = torch.ops.aten.view.default(view_420, [128, 901, 6, 64]);  view_420 = None
        slice_539 = torch.ops.aten.slice.Tensor(view_421, 2, 0, 2)
        slice_542 = torch.ops.aten.slice.Tensor(view_421, 2, 2, 4)
        slice_545 = torch.ops.aten.slice.Tensor(view_421, 2, 4, 9223372036854775807);  view_421 = None
        convert_element_type_918 = torch.ops.prims.convert_element_type.default(slice_539, torch.float32);  slice_539 = None
        convert_element_type_919 = torch.ops.prims.convert_element_type.default(slice_542, torch.float32);  slice_542 = None
        mul_331 = torch.ops.aten.mul.Tensor(convert_element_type_918, unsqueeze_96)
        slice_546 = torch.ops.aten.slice.Tensor(convert_element_type_918, 3, 0, 32)
        slice_547 = torch.ops.aten.slice.Tensor(convert_element_type_918, 3, 32, 9223372036854775807);  convert_element_type_918 = None
        neg_82 = torch.ops.aten.neg.default(slice_547);  slice_547 = None
        cat_84 = torch.ops.aten.cat.default([neg_82, slice_546], -1);  neg_82 = slice_546 = None
        mul_332 = torch.ops.aten.mul.Tensor(cat_84, unsqueeze_97);  cat_84 = None
        add_266 = torch.ops.aten.add.Tensor(mul_331, mul_332);  mul_331 = mul_332 = None
        mul_333 = torch.ops.aten.mul.Tensor(convert_element_type_919, unsqueeze_96)
        slice_548 = torch.ops.aten.slice.Tensor(convert_element_type_919, 3, 0, 32)
        slice_549 = torch.ops.aten.slice.Tensor(convert_element_type_919, 3, 32, 9223372036854775807);  convert_element_type_919 = None
        neg_83 = torch.ops.aten.neg.default(slice_549);  slice_549 = None
        cat_85 = torch.ops.aten.cat.default([neg_83, slice_548], -1);  neg_83 = slice_548 = None
        mul_334 = torch.ops.aten.mul.Tensor(cat_85, unsqueeze_97);  cat_85 = None
        add_267 = torch.ops.aten.add.Tensor(mul_333, mul_334);  mul_333 = mul_334 = None
        convert_element_type_920 = torch.ops.prims.convert_element_type.default(add_266, torch.bfloat16);  add_266 = None
        convert_element_type_921 = torch.ops.prims.convert_element_type.default(add_267, torch.bfloat16);  add_267 = None
        _flash_attn_forward_41 = torch.ops.flash_attn._flash_attn_forward.default(convert_element_type_920, convert_element_type_921, slice_545, 0.0, 0.125, False, -1, -1, 0.0, None, False);  convert_element_type_920 = convert_element_type_921 = slice_545 = None
        getitem_246 = _flash_attn_forward_41[0];  _flash_attn_forward_41 = None
        view_422 = torch.ops.aten.view.default(getitem_246, [128, 901, 128]);  getitem_246 = None
        view_423 = torch.ops.aten.view.default(view_422, [115328, 128]);  view_422 = None
        mm_166 = torch.ops.aten.mm.default(view_423, permute_103);  view_423 = permute_103 = None
        view_424 = torch.ops.aten.view.default(mm_166, [128, 901, 128]);  mm_166 = None
        add_268 = torch.ops.aten.add.Tensor(convert_element_type_914, view_424);  convert_element_type_914 = view_424 = None
        convert_element_type_925 = torch.ops.prims.convert_element_type.default(add_268, torch.float32);  add_268 = None
        pow_83 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_925, 2)
        mean_82 = torch.ops.aten.mean.dim(pow_83, [-1], True);  pow_83 = None
        add_269 = torch.ops.aten.add.Tensor(mean_82, 1e-05);  mean_82 = None
        rsqrt_82 = torch.ops.aten.rsqrt.default(add_269);  add_269 = None
        mul_335 = torch.ops.aten.mul.Tensor(convert_element_type_925, rsqrt_82);  convert_element_type_925 = rsqrt_82 = None
        convert_element_type_926 = torch.ops.prims.convert_element_type.default(mul_335, torch.bfloat16);  mul_335 = None
        view_425 = torch.ops.aten.view.default(convert_element_type_926, [115328, 128])
        mm_167 = torch.ops.aten.mm.default(view_425, permute_104);  view_425 = permute_104 = None
        view_426 = torch.ops.aten.view.default(mm_167, [128, 901, 1024]);  mm_167 = None
        split_41 = torch.ops.aten.split.Tensor(view_426, 512, -1);  view_426 = None
        getitem_250 = split_41[0]
        getitem_251 = split_41[1];  split_41 = None
        convert_element_type_930 = torch.ops.prims.convert_element_type.default(getitem_250, torch.float32);  getitem_250 = None
        sigmoid_41 = torch.ops.aten.sigmoid.default(convert_element_type_930)
        mul_336 = torch.ops.aten.mul.Tensor(convert_element_type_930, sigmoid_41);  convert_element_type_930 = sigmoid_41 = None
        convert_element_type_931 = torch.ops.prims.convert_element_type.default(mul_336, torch.bfloat16);  mul_336 = None
        mul_337 = torch.ops.aten.mul.Tensor(convert_element_type_931, getitem_251);  convert_element_type_931 = getitem_251 = None
        view_427 = torch.ops.aten.view.default(mul_337, [115328, 512]);  mul_337 = None
        mm_168 = torch.ops.aten.mm.default(view_427, permute_105);  view_427 = permute_105 = None
        view_428 = torch.ops.aten.view.default(mm_168, [128, 901, 128]);  mm_168 = None
        add_270 = torch.ops.aten.add.Tensor(convert_element_type_926, view_428);  convert_element_type_926 = view_428 = None
        convert_element_type_935 = torch.ops.prims.convert_element_type.default(add_270, torch.float32);  add_270 = None
        pow_84 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_935, 2)
        mean_83 = torch.ops.aten.mean.dim(pow_84, [-1], True);  pow_84 = None
        add_271 = torch.ops.aten.add.Tensor(mean_83, 1e-05);  mean_83 = None
        rsqrt_83 = torch.ops.aten.rsqrt.default(add_271);  add_271 = None
        mul_338 = torch.ops.aten.mul.Tensor(convert_element_type_935, rsqrt_83);  convert_element_type_935 = rsqrt_83 = None
        convert_element_type_936 = torch.ops.prims.convert_element_type.default(mul_338, torch.bfloat16);  mul_338 = None
        view_429 = torch.ops.aten.view.default(convert_element_type_936, [115328, 128])
        mm_169 = torch.ops.aten.mm.default(view_429, permute_106);  view_429 = permute_106 = None
        view_430 = torch.ops.aten.view.default(mm_169, [128, 901, 384]);  mm_169 = None
        view_431 = torch.ops.aten.view.default(view_430, [128, 901, 6, 64]);  view_430 = None
        slice_552 = torch.ops.aten.slice.Tensor(view_431, 2, 0, 2)
        slice_555 = torch.ops.aten.slice.Tensor(view_431, 2, 2, 4)
        slice_558 = torch.ops.aten.slice.Tensor(view_431, 2, 4, 9223372036854775807);  view_431 = None
        convert_element_type_940 = torch.ops.prims.convert_element_type.default(slice_552, torch.float32);  slice_552 = None
        convert_element_type_941 = torch.ops.prims.convert_element_type.default(slice_555, torch.float32);  slice_555 = None
        mul_339 = torch.ops.aten.mul.Tensor(convert_element_type_940, unsqueeze_96)
        slice_559 = torch.ops.aten.slice.Tensor(convert_element_type_940, 3, 0, 32)
        slice_560 = torch.ops.aten.slice.Tensor(convert_element_type_940, 3, 32, 9223372036854775807);  convert_element_type_940 = None
        neg_84 = torch.ops.aten.neg.default(slice_560);  slice_560 = None
        cat_86 = torch.ops.aten.cat.default([neg_84, slice_559], -1);  neg_84 = slice_559 = None
        mul_340 = torch.ops.aten.mul.Tensor(cat_86, unsqueeze_97);  cat_86 = None
        add_272 = torch.ops.aten.add.Tensor(mul_339, mul_340);  mul_339 = mul_340 = None
        mul_341 = torch.ops.aten.mul.Tensor(convert_element_type_941, unsqueeze_96)
        slice_561 = torch.ops.aten.slice.Tensor(convert_element_type_941, 3, 0, 32)
        slice_562 = torch.ops.aten.slice.Tensor(convert_element_type_941, 3, 32, 9223372036854775807);  convert_element_type_941 = None
        neg_85 = torch.ops.aten.neg.default(slice_562);  slice_562 = None
        cat_87 = torch.ops.aten.cat.default([neg_85, slice_561], -1);  neg_85 = slice_561 = None
        mul_342 = torch.ops.aten.mul.Tensor(cat_87, unsqueeze_97);  cat_87 = None
        add_273 = torch.ops.aten.add.Tensor(mul_341, mul_342);  mul_341 = mul_342 = None
        convert_element_type_942 = torch.ops.prims.convert_element_type.default(add_272, torch.bfloat16);  add_272 = None
        convert_element_type_943 = torch.ops.prims.convert_element_type.default(add_273, torch.bfloat16);  add_273 = None
        _flash_attn_forward_42 = torch.ops.flash_attn._flash_attn_forward.default(convert_element_type_942, convert_element_type_943, slice_558, 0.0, 0.125, False, -1, -1, 0.0, None, False);  convert_element_type_942 = convert_element_type_943 = slice_558 = None
        getitem_252 = _flash_attn_forward_42[0];  _flash_attn_forward_42 = None
        view_432 = torch.ops.aten.view.default(getitem_252, [128, 901, 128]);  getitem_252 = None
        view_433 = torch.ops.aten.view.default(view_432, [115328, 128]);  view_432 = None
        mm_170 = torch.ops.aten.mm.default(view_433, permute_107);  view_433 = permute_107 = None
        view_434 = torch.ops.aten.view.default(mm_170, [128, 901, 128]);  mm_170 = None
        add_274 = torch.ops.aten.add.Tensor(convert_element_type_936, view_434);  convert_element_type_936 = view_434 = None
        convert_element_type_947 = torch.ops.prims.convert_element_type.default(add_274, torch.float32);  add_274 = None
        pow_85 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_947, 2)
        mean_84 = torch.ops.aten.mean.dim(pow_85, [-1], True);  pow_85 = None
        add_275 = torch.ops.aten.add.Tensor(mean_84, 1e-05);  mean_84 = None
        rsqrt_84 = torch.ops.aten.rsqrt.default(add_275);  add_275 = None
        mul_343 = torch.ops.aten.mul.Tensor(convert_element_type_947, rsqrt_84);  convert_element_type_947 = rsqrt_84 = None
        convert_element_type_948 = torch.ops.prims.convert_element_type.default(mul_343, torch.bfloat16);  mul_343 = None
        view_435 = torch.ops.aten.view.default(convert_element_type_948, [115328, 128])
        mm_171 = torch.ops.aten.mm.default(view_435, permute_108);  view_435 = permute_108 = None
        view_436 = torch.ops.aten.view.default(mm_171, [128, 901, 1024]);  mm_171 = None
        split_42 = torch.ops.aten.split.Tensor(view_436, 512, -1);  view_436 = None
        getitem_256 = split_42[0]
        getitem_257 = split_42[1];  split_42 = None
        convert_element_type_952 = torch.ops.prims.convert_element_type.default(getitem_256, torch.float32);  getitem_256 = None
        sigmoid_42 = torch.ops.aten.sigmoid.default(convert_element_type_952)
        mul_344 = torch.ops.aten.mul.Tensor(convert_element_type_952, sigmoid_42);  convert_element_type_952 = sigmoid_42 = None
        convert_element_type_953 = torch.ops.prims.convert_element_type.default(mul_344, torch.bfloat16);  mul_344 = None
        mul_345 = torch.ops.aten.mul.Tensor(convert_element_type_953, getitem_257);  convert_element_type_953 = getitem_257 = None
        view_437 = torch.ops.aten.view.default(mul_345, [115328, 512]);  mul_345 = None
        mm_172 = torch.ops.aten.mm.default(view_437, permute_109);  view_437 = permute_109 = None
        view_438 = torch.ops.aten.view.default(mm_172, [128, 901, 128]);  mm_172 = None
        add_276 = torch.ops.aten.add.Tensor(convert_element_type_948, view_438);  convert_element_type_948 = view_438 = None
        convert_element_type_957 = torch.ops.prims.convert_element_type.default(add_276, torch.float32);  add_276 = None
        pow_86 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_957, 2)
        mean_85 = torch.ops.aten.mean.dim(pow_86, [-1], True);  pow_86 = None
        add_277 = torch.ops.aten.add.Tensor(mean_85, 1e-05);  mean_85 = None
        rsqrt_85 = torch.ops.aten.rsqrt.default(add_277);  add_277 = None
        mul_346 = torch.ops.aten.mul.Tensor(convert_element_type_957, rsqrt_85);  convert_element_type_957 = rsqrt_85 = None
        convert_element_type_958 = torch.ops.prims.convert_element_type.default(mul_346, torch.bfloat16);  mul_346 = None
        view_439 = torch.ops.aten.view.default(convert_element_type_958, [115328, 128])
        mm_173 = torch.ops.aten.mm.default(view_439, permute_110);  view_439 = permute_110 = None
        view_440 = torch.ops.aten.view.default(mm_173, [128, 901, 384]);  mm_173 = None
        view_441 = torch.ops.aten.view.default(view_440, [128, 901, 6, 64]);  view_440 = None
        slice_565 = torch.ops.aten.slice.Tensor(view_441, 2, 0, 2)
        slice_568 = torch.ops.aten.slice.Tensor(view_441, 2, 2, 4)
        slice_571 = torch.ops.aten.slice.Tensor(view_441, 2, 4, 9223372036854775807);  view_441 = None
        convert_element_type_962 = torch.ops.prims.convert_element_type.default(slice_565, torch.float32);  slice_565 = None
        convert_element_type_963 = torch.ops.prims.convert_element_type.default(slice_568, torch.float32);  slice_568 = None
        mul_347 = torch.ops.aten.mul.Tensor(convert_element_type_962, unsqueeze_96)
        slice_572 = torch.ops.aten.slice.Tensor(convert_element_type_962, 3, 0, 32)
        slice_573 = torch.ops.aten.slice.Tensor(convert_element_type_962, 3, 32, 9223372036854775807);  convert_element_type_962 = None
        neg_86 = torch.ops.aten.neg.default(slice_573);  slice_573 = None
        cat_88 = torch.ops.aten.cat.default([neg_86, slice_572], -1);  neg_86 = slice_572 = None
        mul_348 = torch.ops.aten.mul.Tensor(cat_88, unsqueeze_97);  cat_88 = None
        add_278 = torch.ops.aten.add.Tensor(mul_347, mul_348);  mul_347 = mul_348 = None
        mul_349 = torch.ops.aten.mul.Tensor(convert_element_type_963, unsqueeze_96)
        slice_574 = torch.ops.aten.slice.Tensor(convert_element_type_963, 3, 0, 32)
        slice_575 = torch.ops.aten.slice.Tensor(convert_element_type_963, 3, 32, 9223372036854775807);  convert_element_type_963 = None
        neg_87 = torch.ops.aten.neg.default(slice_575);  slice_575 = None
        cat_89 = torch.ops.aten.cat.default([neg_87, slice_574], -1);  neg_87 = slice_574 = None
        mul_350 = torch.ops.aten.mul.Tensor(cat_89, unsqueeze_97);  cat_89 = None
        add_279 = torch.ops.aten.add.Tensor(mul_349, mul_350);  mul_349 = mul_350 = None
        convert_element_type_964 = torch.ops.prims.convert_element_type.default(add_278, torch.bfloat16);  add_278 = None
        convert_element_type_965 = torch.ops.prims.convert_element_type.default(add_279, torch.bfloat16);  add_279 = None
        _flash_attn_forward_43 = torch.ops.flash_attn._flash_attn_forward.default(convert_element_type_964, convert_element_type_965, slice_571, 0.0, 0.125, False, -1, -1, 0.0, None, False);  convert_element_type_964 = convert_element_type_965 = slice_571 = None
        getitem_258 = _flash_attn_forward_43[0];  _flash_attn_forward_43 = None
        view_442 = torch.ops.aten.view.default(getitem_258, [128, 901, 128]);  getitem_258 = None
        view_443 = torch.ops.aten.view.default(view_442, [115328, 128]);  view_442 = None
        mm_174 = torch.ops.aten.mm.default(view_443, permute_111);  view_443 = permute_111 = None
        view_444 = torch.ops.aten.view.default(mm_174, [128, 901, 128]);  mm_174 = None
        add_280 = torch.ops.aten.add.Tensor(convert_element_type_958, view_444);  convert_element_type_958 = view_444 = None
        convert_element_type_969 = torch.ops.prims.convert_element_type.default(add_280, torch.float32);  add_280 = None
        pow_87 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_969, 2)
        mean_86 = torch.ops.aten.mean.dim(pow_87, [-1], True);  pow_87 = None
        add_281 = torch.ops.aten.add.Tensor(mean_86, 1e-05);  mean_86 = None
        rsqrt_86 = torch.ops.aten.rsqrt.default(add_281);  add_281 = None
        mul_351 = torch.ops.aten.mul.Tensor(convert_element_type_969, rsqrt_86);  convert_element_type_969 = rsqrt_86 = None
        convert_element_type_970 = torch.ops.prims.convert_element_type.default(mul_351, torch.bfloat16);  mul_351 = None
        view_445 = torch.ops.aten.view.default(convert_element_type_970, [115328, 128])
        mm_175 = torch.ops.aten.mm.default(view_445, permute_112);  view_445 = permute_112 = None
        view_446 = torch.ops.aten.view.default(mm_175, [128, 901, 1024]);  mm_175 = None
        split_43 = torch.ops.aten.split.Tensor(view_446, 512, -1);  view_446 = None
        getitem_262 = split_43[0]
        getitem_263 = split_43[1];  split_43 = None
        convert_element_type_974 = torch.ops.prims.convert_element_type.default(getitem_262, torch.float32);  getitem_262 = None
        sigmoid_43 = torch.ops.aten.sigmoid.default(convert_element_type_974)
        mul_352 = torch.ops.aten.mul.Tensor(convert_element_type_974, sigmoid_43);  convert_element_type_974 = sigmoid_43 = None
        convert_element_type_975 = torch.ops.prims.convert_element_type.default(mul_352, torch.bfloat16);  mul_352 = None
        mul_353 = torch.ops.aten.mul.Tensor(convert_element_type_975, getitem_263);  convert_element_type_975 = getitem_263 = None
        view_447 = torch.ops.aten.view.default(mul_353, [115328, 512]);  mul_353 = None
        mm_176 = torch.ops.aten.mm.default(view_447, permute_113);  view_447 = permute_113 = None
        view_448 = torch.ops.aten.view.default(mm_176, [128, 901, 128]);  mm_176 = None
        add_282 = torch.ops.aten.add.Tensor(convert_element_type_970, view_448);  convert_element_type_970 = view_448 = None
        convert_element_type_979 = torch.ops.prims.convert_element_type.default(add_282, torch.float32);  add_282 = None
        pow_88 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_979, 2)
        mean_87 = torch.ops.aten.mean.dim(pow_88, [-1], True);  pow_88 = None
        add_283 = torch.ops.aten.add.Tensor(mean_87, 1e-05);  mean_87 = None
        rsqrt_87 = torch.ops.aten.rsqrt.default(add_283);  add_283 = None
        mul_354 = torch.ops.aten.mul.Tensor(convert_element_type_979, rsqrt_87);  convert_element_type_979 = rsqrt_87 = None
        convert_element_type_980 = torch.ops.prims.convert_element_type.default(mul_354, torch.bfloat16);  mul_354 = None
        add_284 = torch.ops.aten.add.Tensor(convert_element_type_804, convert_element_type_980);  convert_element_type_804 = convert_element_type_980 = None
        view_449 = torch.ops.aten.view.default(add_284, [115328, 128])
        mm_177 = torch.ops.aten.mm.default(view_449, permute_130);  view_449 = permute_130 = None
        view_450 = torch.ops.aten.view.default(mm_177, [128, 901, 384]);  mm_177 = None
        view_451 = torch.ops.aten.view.default(view_450, [128, 901, 6, 64]);  view_450 = None
        slice_578 = torch.ops.aten.slice.Tensor(view_451, 2, 0, 2)
        slice_581 = torch.ops.aten.slice.Tensor(view_451, 2, 2, 4)
        slice_584 = torch.ops.aten.slice.Tensor(view_451, 2, 4, 9223372036854775807);  view_451 = None
        convert_element_type_984 = torch.ops.prims.convert_element_type.default(slice_578, torch.float32);  slice_578 = None
        convert_element_type_985 = torch.ops.prims.convert_element_type.default(slice_581, torch.float32);  slice_581 = None
        mul_355 = torch.ops.aten.mul.Tensor(convert_element_type_984, unsqueeze_96)
        slice_585 = torch.ops.aten.slice.Tensor(convert_element_type_984, 3, 0, 32)
        slice_586 = torch.ops.aten.slice.Tensor(convert_element_type_984, 3, 32, 9223372036854775807);  convert_element_type_984 = None
        neg_88 = torch.ops.aten.neg.default(slice_586);  slice_586 = None
        cat_90 = torch.ops.aten.cat.default([neg_88, slice_585], -1);  neg_88 = slice_585 = None
        mul_356 = torch.ops.aten.mul.Tensor(cat_90, unsqueeze_97);  cat_90 = None
        add_285 = torch.ops.aten.add.Tensor(mul_355, mul_356);  mul_355 = mul_356 = None
        mul_357 = torch.ops.aten.mul.Tensor(convert_element_type_985, unsqueeze_96)
        slice_587 = torch.ops.aten.slice.Tensor(convert_element_type_985, 3, 0, 32)
        slice_588 = torch.ops.aten.slice.Tensor(convert_element_type_985, 3, 32, 9223372036854775807);  convert_element_type_985 = None
        neg_89 = torch.ops.aten.neg.default(slice_588);  slice_588 = None
        cat_91 = torch.ops.aten.cat.default([neg_89, slice_587], -1);  neg_89 = slice_587 = None
        mul_358 = torch.ops.aten.mul.Tensor(cat_91, unsqueeze_97);  cat_91 = None
        add_286 = torch.ops.aten.add.Tensor(mul_357, mul_358);  mul_357 = mul_358 = None
        convert_element_type_986 = torch.ops.prims.convert_element_type.default(add_285, torch.bfloat16);  add_285 = None
        convert_element_type_987 = torch.ops.prims.convert_element_type.default(add_286, torch.bfloat16);  add_286 = None
        _flash_attn_forward_44 = torch.ops.flash_attn._flash_attn_forward.default(convert_element_type_986, convert_element_type_987, slice_584, 0.0, 0.125, False, -1, -1, 0.0, None, False);  convert_element_type_986 = convert_element_type_987 = slice_584 = None
        getitem_264 = _flash_attn_forward_44[0];  _flash_attn_forward_44 = None
        view_452 = torch.ops.aten.view.default(getitem_264, [128, 901, 128]);  getitem_264 = None
        view_453 = torch.ops.aten.view.default(view_452, [115328, 128]);  view_452 = None
        mm_178 = torch.ops.aten.mm.default(view_453, permute_131);  view_453 = permute_131 = None
        view_454 = torch.ops.aten.view.default(mm_178, [128, 901, 128]);  mm_178 = None
        add_287 = torch.ops.aten.add.Tensor(add_284, view_454);  add_284 = view_454 = None
        convert_element_type_991 = torch.ops.prims.convert_element_type.default(add_287, torch.float32);  add_287 = None
        pow_89 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_991, 2)
        mean_88 = torch.ops.aten.mean.dim(pow_89, [-1], True);  pow_89 = None
        add_288 = torch.ops.aten.add.Tensor(mean_88, 1e-05);  mean_88 = None
        rsqrt_88 = torch.ops.aten.rsqrt.default(add_288);  add_288 = None
        mul_359 = torch.ops.aten.mul.Tensor(convert_element_type_991, rsqrt_88);  convert_element_type_991 = rsqrt_88 = None
        convert_element_type_992 = torch.ops.prims.convert_element_type.default(mul_359, torch.bfloat16);  mul_359 = None
        view_455 = torch.ops.aten.view.default(convert_element_type_992, [115328, 128])
        mm_179 = torch.ops.aten.mm.default(view_455, permute_132);  view_455 = permute_132 = None
        view_456 = torch.ops.aten.view.default(mm_179, [128, 901, 1024]);  mm_179 = None
        split_44 = torch.ops.aten.split.Tensor(view_456, 512, -1);  view_456 = None
        getitem_268 = split_44[0]
        getitem_269 = split_44[1];  split_44 = None
        convert_element_type_996 = torch.ops.prims.convert_element_type.default(getitem_268, torch.float32);  getitem_268 = None
        sigmoid_44 = torch.ops.aten.sigmoid.default(convert_element_type_996)
        mul_360 = torch.ops.aten.mul.Tensor(convert_element_type_996, sigmoid_44);  convert_element_type_996 = sigmoid_44 = None
        convert_element_type_997 = torch.ops.prims.convert_element_type.default(mul_360, torch.bfloat16);  mul_360 = None
        mul_361 = torch.ops.aten.mul.Tensor(convert_element_type_997, getitem_269);  convert_element_type_997 = getitem_269 = None
        view_457 = torch.ops.aten.view.default(mul_361, [115328, 512]);  mul_361 = None
        mm_180 = torch.ops.aten.mm.default(view_457, permute_133);  view_457 = permute_133 = None
        view_458 = torch.ops.aten.view.default(mm_180, [128, 901, 128]);  mm_180 = None
        add_289 = torch.ops.aten.add.Tensor(convert_element_type_992, view_458);  convert_element_type_992 = view_458 = None
        convert_element_type_1001 = torch.ops.prims.convert_element_type.default(add_289, torch.float32);  add_289 = None
        pow_90 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_1001, 2)
        mean_89 = torch.ops.aten.mean.dim(pow_90, [-1], True);  pow_90 = None
        add_290 = torch.ops.aten.add.Tensor(mean_89, 1e-05);  mean_89 = None
        rsqrt_89 = torch.ops.aten.rsqrt.default(add_290);  add_290 = None
        mul_362 = torch.ops.aten.mul.Tensor(convert_element_type_1001, rsqrt_89);  convert_element_type_1001 = rsqrt_89 = None
        convert_element_type_1002 = torch.ops.prims.convert_element_type.default(mul_362, torch.bfloat16);  mul_362 = None
        view_459 = torch.ops.aten.view.default(convert_element_type_1002, [115328, 128])
        mm_181 = torch.ops.aten.mm.default(view_459, permute_134);  view_459 = permute_134 = None
        view_460 = torch.ops.aten.view.default(mm_181, [128, 901, 384]);  mm_181 = None
        view_461 = torch.ops.aten.view.default(view_460, [128, 901, 6, 64]);  view_460 = None
        slice_591 = torch.ops.aten.slice.Tensor(view_461, 2, 0, 2)
        slice_594 = torch.ops.aten.slice.Tensor(view_461, 2, 2, 4)
        slice_597 = torch.ops.aten.slice.Tensor(view_461, 2, 4, 9223372036854775807);  view_461 = None
        convert_element_type_1006 = torch.ops.prims.convert_element_type.default(slice_591, torch.float32);  slice_591 = None
        convert_element_type_1007 = torch.ops.prims.convert_element_type.default(slice_594, torch.float32);  slice_594 = None
        mul_363 = torch.ops.aten.mul.Tensor(convert_element_type_1006, unsqueeze_96)
        slice_598 = torch.ops.aten.slice.Tensor(convert_element_type_1006, 3, 0, 32)
        slice_599 = torch.ops.aten.slice.Tensor(convert_element_type_1006, 3, 32, 9223372036854775807);  convert_element_type_1006 = None
        neg_90 = torch.ops.aten.neg.default(slice_599);  slice_599 = None
        cat_92 = torch.ops.aten.cat.default([neg_90, slice_598], -1);  neg_90 = slice_598 = None
        mul_364 = torch.ops.aten.mul.Tensor(cat_92, unsqueeze_97);  cat_92 = None
        add_291 = torch.ops.aten.add.Tensor(mul_363, mul_364);  mul_363 = mul_364 = None
        mul_365 = torch.ops.aten.mul.Tensor(convert_element_type_1007, unsqueeze_96)
        slice_600 = torch.ops.aten.slice.Tensor(convert_element_type_1007, 3, 0, 32)
        slice_601 = torch.ops.aten.slice.Tensor(convert_element_type_1007, 3, 32, 9223372036854775807);  convert_element_type_1007 = None
        neg_91 = torch.ops.aten.neg.default(slice_601);  slice_601 = None
        cat_93 = torch.ops.aten.cat.default([neg_91, slice_600], -1);  neg_91 = slice_600 = None
        mul_366 = torch.ops.aten.mul.Tensor(cat_93, unsqueeze_97);  cat_93 = None
        add_292 = torch.ops.aten.add.Tensor(mul_365, mul_366);  mul_365 = mul_366 = None
        convert_element_type_1008 = torch.ops.prims.convert_element_type.default(add_291, torch.bfloat16);  add_291 = None
        convert_element_type_1009 = torch.ops.prims.convert_element_type.default(add_292, torch.bfloat16);  add_292 = None
        _flash_attn_forward_45 = torch.ops.flash_attn._flash_attn_forward.default(convert_element_type_1008, convert_element_type_1009, slice_597, 0.0, 0.125, False, -1, -1, 0.0, None, False);  convert_element_type_1008 = convert_element_type_1009 = slice_597 = None
        getitem_270 = _flash_attn_forward_45[0];  _flash_attn_forward_45 = None
        view_462 = torch.ops.aten.view.default(getitem_270, [128, 901, 128]);  getitem_270 = None
        view_463 = torch.ops.aten.view.default(view_462, [115328, 128]);  view_462 = None
        mm_182 = torch.ops.aten.mm.default(view_463, permute_135);  view_463 = permute_135 = None
        view_464 = torch.ops.aten.view.default(mm_182, [128, 901, 128]);  mm_182 = None
        add_293 = torch.ops.aten.add.Tensor(convert_element_type_1002, view_464);  convert_element_type_1002 = view_464 = None
        convert_element_type_1013 = torch.ops.prims.convert_element_type.default(add_293, torch.float32);  add_293 = None
        pow_91 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_1013, 2)
        mean_90 = torch.ops.aten.mean.dim(pow_91, [-1], True);  pow_91 = None
        add_294 = torch.ops.aten.add.Tensor(mean_90, 1e-05);  mean_90 = None
        rsqrt_90 = torch.ops.aten.rsqrt.default(add_294);  add_294 = None
        mul_367 = torch.ops.aten.mul.Tensor(convert_element_type_1013, rsqrt_90);  convert_element_type_1013 = rsqrt_90 = None
        convert_element_type_1014 = torch.ops.prims.convert_element_type.default(mul_367, torch.bfloat16);  mul_367 = None
        view_465 = torch.ops.aten.view.default(convert_element_type_1014, [115328, 128])
        mm_183 = torch.ops.aten.mm.default(view_465, permute_136);  view_465 = permute_136 = None
        view_466 = torch.ops.aten.view.default(mm_183, [128, 901, 1024]);  mm_183 = None
        split_45 = torch.ops.aten.split.Tensor(view_466, 512, -1);  view_466 = None
        getitem_274 = split_45[0]
        getitem_275 = split_45[1];  split_45 = None
        convert_element_type_1018 = torch.ops.prims.convert_element_type.default(getitem_274, torch.float32);  getitem_274 = None
        sigmoid_45 = torch.ops.aten.sigmoid.default(convert_element_type_1018)
        mul_368 = torch.ops.aten.mul.Tensor(convert_element_type_1018, sigmoid_45);  convert_element_type_1018 = sigmoid_45 = None
        convert_element_type_1019 = torch.ops.prims.convert_element_type.default(mul_368, torch.bfloat16);  mul_368 = None
        mul_369 = torch.ops.aten.mul.Tensor(convert_element_type_1019, getitem_275);  convert_element_type_1019 = getitem_275 = None
        view_467 = torch.ops.aten.view.default(mul_369, [115328, 512]);  mul_369 = None
        mm_184 = torch.ops.aten.mm.default(view_467, permute_137);  view_467 = permute_137 = None
        view_468 = torch.ops.aten.view.default(mm_184, [128, 901, 128]);  mm_184 = None
        add_295 = torch.ops.aten.add.Tensor(convert_element_type_1014, view_468);  convert_element_type_1014 = view_468 = None
        convert_element_type_1023 = torch.ops.prims.convert_element_type.default(add_295, torch.float32);  add_295 = None
        pow_92 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_1023, 2)
        mean_91 = torch.ops.aten.mean.dim(pow_92, [-1], True);  pow_92 = None
        add_296 = torch.ops.aten.add.Tensor(mean_91, 1e-05);  mean_91 = None
        rsqrt_91 = torch.ops.aten.rsqrt.default(add_296);  add_296 = None
        mul_370 = torch.ops.aten.mul.Tensor(convert_element_type_1023, rsqrt_91);  convert_element_type_1023 = rsqrt_91 = None
        convert_element_type_1024 = torch.ops.prims.convert_element_type.default(mul_370, torch.bfloat16);  mul_370 = None
        view_469 = torch.ops.aten.view.default(convert_element_type_1024, [115328, 128])
        mm_185 = torch.ops.aten.mm.default(view_469, permute_138);  view_469 = permute_138 = None
        view_470 = torch.ops.aten.view.default(mm_185, [128, 901, 384]);  mm_185 = None
        view_471 = torch.ops.aten.view.default(view_470, [128, 901, 6, 64]);  view_470 = None
        slice_604 = torch.ops.aten.slice.Tensor(view_471, 2, 0, 2)
        slice_607 = torch.ops.aten.slice.Tensor(view_471, 2, 2, 4)
        slice_610 = torch.ops.aten.slice.Tensor(view_471, 2, 4, 9223372036854775807);  view_471 = None
        convert_element_type_1028 = torch.ops.prims.convert_element_type.default(slice_604, torch.float32);  slice_604 = None
        convert_element_type_1029 = torch.ops.prims.convert_element_type.default(slice_607, torch.float32);  slice_607 = None
        mul_371 = torch.ops.aten.mul.Tensor(convert_element_type_1028, unsqueeze_96)
        slice_611 = torch.ops.aten.slice.Tensor(convert_element_type_1028, 3, 0, 32)
        slice_612 = torch.ops.aten.slice.Tensor(convert_element_type_1028, 3, 32, 9223372036854775807);  convert_element_type_1028 = None
        neg_92 = torch.ops.aten.neg.default(slice_612);  slice_612 = None
        cat_94 = torch.ops.aten.cat.default([neg_92, slice_611], -1);  neg_92 = slice_611 = None
        mul_372 = torch.ops.aten.mul.Tensor(cat_94, unsqueeze_97);  cat_94 = None
        add_297 = torch.ops.aten.add.Tensor(mul_371, mul_372);  mul_371 = mul_372 = None
        mul_373 = torch.ops.aten.mul.Tensor(convert_element_type_1029, unsqueeze_96)
        slice_613 = torch.ops.aten.slice.Tensor(convert_element_type_1029, 3, 0, 32)
        slice_614 = torch.ops.aten.slice.Tensor(convert_element_type_1029, 3, 32, 9223372036854775807);  convert_element_type_1029 = None
        neg_93 = torch.ops.aten.neg.default(slice_614);  slice_614 = None
        cat_95 = torch.ops.aten.cat.default([neg_93, slice_613], -1);  neg_93 = slice_613 = None
        mul_374 = torch.ops.aten.mul.Tensor(cat_95, unsqueeze_97);  cat_95 = None
        add_298 = torch.ops.aten.add.Tensor(mul_373, mul_374);  mul_373 = mul_374 = None
        convert_element_type_1030 = torch.ops.prims.convert_element_type.default(add_297, torch.bfloat16);  add_297 = None
        convert_element_type_1031 = torch.ops.prims.convert_element_type.default(add_298, torch.bfloat16);  add_298 = None
        _flash_attn_forward_46 = torch.ops.flash_attn._flash_attn_forward.default(convert_element_type_1030, convert_element_type_1031, slice_610, 0.0, 0.125, False, -1, -1, 0.0, None, False);  convert_element_type_1030 = convert_element_type_1031 = slice_610 = None
        getitem_276 = _flash_attn_forward_46[0];  _flash_attn_forward_46 = None
        view_472 = torch.ops.aten.view.default(getitem_276, [128, 901, 128]);  getitem_276 = None
        view_473 = torch.ops.aten.view.default(view_472, [115328, 128]);  view_472 = None
        mm_186 = torch.ops.aten.mm.default(view_473, permute_139);  view_473 = permute_139 = None
        view_474 = torch.ops.aten.view.default(mm_186, [128, 901, 128]);  mm_186 = None
        add_299 = torch.ops.aten.add.Tensor(convert_element_type_1024, view_474);  convert_element_type_1024 = view_474 = None
        convert_element_type_1035 = torch.ops.prims.convert_element_type.default(add_299, torch.float32);  add_299 = None
        pow_93 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_1035, 2)
        mean_92 = torch.ops.aten.mean.dim(pow_93, [-1], True);  pow_93 = None
        add_300 = torch.ops.aten.add.Tensor(mean_92, 1e-05);  mean_92 = None
        rsqrt_92 = torch.ops.aten.rsqrt.default(add_300);  add_300 = None
        mul_375 = torch.ops.aten.mul.Tensor(convert_element_type_1035, rsqrt_92);  convert_element_type_1035 = rsqrt_92 = None
        convert_element_type_1036 = torch.ops.prims.convert_element_type.default(mul_375, torch.bfloat16);  mul_375 = None
        view_475 = torch.ops.aten.view.default(convert_element_type_1036, [115328, 128])
        mm_187 = torch.ops.aten.mm.default(view_475, permute_140);  view_475 = permute_140 = None
        view_476 = torch.ops.aten.view.default(mm_187, [128, 901, 1024]);  mm_187 = None
        split_46 = torch.ops.aten.split.Tensor(view_476, 512, -1);  view_476 = None
        getitem_280 = split_46[0]
        getitem_281 = split_46[1];  split_46 = None
        convert_element_type_1040 = torch.ops.prims.convert_element_type.default(getitem_280, torch.float32);  getitem_280 = None
        sigmoid_46 = torch.ops.aten.sigmoid.default(convert_element_type_1040)
        mul_376 = torch.ops.aten.mul.Tensor(convert_element_type_1040, sigmoid_46);  convert_element_type_1040 = sigmoid_46 = None
        convert_element_type_1041 = torch.ops.prims.convert_element_type.default(mul_376, torch.bfloat16);  mul_376 = None
        mul_377 = torch.ops.aten.mul.Tensor(convert_element_type_1041, getitem_281);  convert_element_type_1041 = getitem_281 = None
        view_477 = torch.ops.aten.view.default(mul_377, [115328, 512]);  mul_377 = None
        mm_188 = torch.ops.aten.mm.default(view_477, permute_141);  view_477 = permute_141 = None
        view_478 = torch.ops.aten.view.default(mm_188, [128, 901, 128]);  mm_188 = None
        add_301 = torch.ops.aten.add.Tensor(convert_element_type_1036, view_478);  convert_element_type_1036 = view_478 = None
        convert_element_type_1045 = torch.ops.prims.convert_element_type.default(add_301, torch.float32);  add_301 = None
        pow_94 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_1045, 2)
        mean_93 = torch.ops.aten.mean.dim(pow_94, [-1], True);  pow_94 = None
        add_302 = torch.ops.aten.add.Tensor(mean_93, 1e-05);  mean_93 = None
        rsqrt_93 = torch.ops.aten.rsqrt.default(add_302);  add_302 = None
        mul_378 = torch.ops.aten.mul.Tensor(convert_element_type_1045, rsqrt_93);  convert_element_type_1045 = rsqrt_93 = None
        convert_element_type_1046 = torch.ops.prims.convert_element_type.default(mul_378, torch.bfloat16);  mul_378 = None
        view_479 = torch.ops.aten.view.default(convert_element_type_1046, [115328, 128])
        mm_189 = torch.ops.aten.mm.default(view_479, permute_142);  view_479 = permute_142 = None
        view_480 = torch.ops.aten.view.default(mm_189, [128, 901, 384]);  mm_189 = None
        view_481 = torch.ops.aten.view.default(view_480, [128, 901, 6, 64]);  view_480 = None
        slice_617 = torch.ops.aten.slice.Tensor(view_481, 2, 0, 2)
        slice_620 = torch.ops.aten.slice.Tensor(view_481, 2, 2, 4)
        slice_623 = torch.ops.aten.slice.Tensor(view_481, 2, 4, 9223372036854775807);  view_481 = None
        convert_element_type_1050 = torch.ops.prims.convert_element_type.default(slice_617, torch.float32);  slice_617 = None
        convert_element_type_1051 = torch.ops.prims.convert_element_type.default(slice_620, torch.float32);  slice_620 = None
        mul_379 = torch.ops.aten.mul.Tensor(convert_element_type_1050, unsqueeze_96)
        slice_624 = torch.ops.aten.slice.Tensor(convert_element_type_1050, 3, 0, 32)
        slice_625 = torch.ops.aten.slice.Tensor(convert_element_type_1050, 3, 32, 9223372036854775807);  convert_element_type_1050 = None
        neg_94 = torch.ops.aten.neg.default(slice_625);  slice_625 = None
        cat_96 = torch.ops.aten.cat.default([neg_94, slice_624], -1);  neg_94 = slice_624 = None
        mul_380 = torch.ops.aten.mul.Tensor(cat_96, unsqueeze_97);  cat_96 = None
        add_303 = torch.ops.aten.add.Tensor(mul_379, mul_380);  mul_379 = mul_380 = None
        mul_381 = torch.ops.aten.mul.Tensor(convert_element_type_1051, unsqueeze_96);  unsqueeze_96 = None
        slice_626 = torch.ops.aten.slice.Tensor(convert_element_type_1051, 3, 0, 32)
        slice_627 = torch.ops.aten.slice.Tensor(convert_element_type_1051, 3, 32, 9223372036854775807);  convert_element_type_1051 = None
        neg_95 = torch.ops.aten.neg.default(slice_627);  slice_627 = None
        cat_97 = torch.ops.aten.cat.default([neg_95, slice_626], -1);  neg_95 = slice_626 = None
        mul_382 = torch.ops.aten.mul.Tensor(cat_97, unsqueeze_97);  cat_97 = unsqueeze_97 = None
        add_304 = torch.ops.aten.add.Tensor(mul_381, mul_382);  mul_381 = mul_382 = None
        convert_element_type_1052 = torch.ops.prims.convert_element_type.default(add_303, torch.bfloat16);  add_303 = None
        convert_element_type_1053 = torch.ops.prims.convert_element_type.default(add_304, torch.bfloat16);  add_304 = None
        _flash_attn_forward_47 = torch.ops.flash_attn._flash_attn_forward.default(convert_element_type_1052, convert_element_type_1053, slice_623, 0.0, 0.125, False, -1, -1, 0.0, None, False);  convert_element_type_1052 = convert_element_type_1053 = slice_623 = None
        getitem_282 = _flash_attn_forward_47[0];  _flash_attn_forward_47 = None
        view_482 = torch.ops.aten.view.default(getitem_282, [128, 901, 128]);  getitem_282 = None
        view_483 = torch.ops.aten.view.default(view_482, [115328, 128]);  view_482 = None
        mm_190 = torch.ops.aten.mm.default(view_483, permute_143);  view_483 = permute_143 = None
        view_484 = torch.ops.aten.view.default(mm_190, [128, 901, 128]);  mm_190 = None
        add_305 = torch.ops.aten.add.Tensor(convert_element_type_1046, view_484);  convert_element_type_1046 = view_484 = None
        convert_element_type_1057 = torch.ops.prims.convert_element_type.default(add_305, torch.float32);  add_305 = None
        pow_95 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_1057, 2)
        mean_94 = torch.ops.aten.mean.dim(pow_95, [-1], True);  pow_95 = None
        add_306 = torch.ops.aten.add.Tensor(mean_94, 1e-05);  mean_94 = None
        rsqrt_94 = torch.ops.aten.rsqrt.default(add_306);  add_306 = None
        mul_383 = torch.ops.aten.mul.Tensor(convert_element_type_1057, rsqrt_94);  convert_element_type_1057 = rsqrt_94 = None
        convert_element_type_1058 = torch.ops.prims.convert_element_type.default(mul_383, torch.bfloat16);  mul_383 = None
        view_485 = torch.ops.aten.view.default(convert_element_type_1058, [115328, 128])
        mm_191 = torch.ops.aten.mm.default(view_485, permute_144);  view_485 = permute_144 = None
        view_486 = torch.ops.aten.view.default(mm_191, [128, 901, 1024]);  mm_191 = None
        split_47 = torch.ops.aten.split.Tensor(view_486, 512, -1);  view_486 = None
        getitem_286 = split_47[0]
        getitem_287 = split_47[1];  split_47 = None
        convert_element_type_1062 = torch.ops.prims.convert_element_type.default(getitem_286, torch.float32);  getitem_286 = None
        sigmoid_47 = torch.ops.aten.sigmoid.default(convert_element_type_1062)
        mul_384 = torch.ops.aten.mul.Tensor(convert_element_type_1062, sigmoid_47);  convert_element_type_1062 = sigmoid_47 = None
        convert_element_type_1063 = torch.ops.prims.convert_element_type.default(mul_384, torch.bfloat16);  mul_384 = None
        mul_385 = torch.ops.aten.mul.Tensor(convert_element_type_1063, getitem_287);  convert_element_type_1063 = getitem_287 = None
        view_487 = torch.ops.aten.view.default(mul_385, [115328, 512]);  mul_385 = None
        mm_192 = torch.ops.aten.mm.default(view_487, permute_145);  view_487 = permute_145 = None
        view_488 = torch.ops.aten.view.default(mm_192, [128, 901, 128]);  mm_192 = None
        add_307 = torch.ops.aten.add.Tensor(convert_element_type_1058, view_488);  convert_element_type_1058 = view_488 = None
        convert_element_type_1067 = torch.ops.prims.convert_element_type.default(add_307, torch.float32);  add_307 = None
        pow_96 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_1067, 2)
        mean_95 = torch.ops.aten.mean.dim(pow_96, [-1], True);  pow_96 = None
        add_308 = torch.ops.aten.add.Tensor(mean_95, 1e-05);  mean_95 = None
        rsqrt_95 = torch.ops.aten.rsqrt.default(add_308);  add_308 = None
        mul_386 = torch.ops.aten.mul.Tensor(convert_element_type_1067, rsqrt_95);  convert_element_type_1067 = rsqrt_95 = None
        convert_element_type_1068 = torch.ops.prims.convert_element_type.default(mul_386, torch.bfloat16);  mul_386 = None
        select_3 = torch.ops.aten.select.int(convert_element_type_1068, 1, 0);  convert_element_type_1068 = None
        convert_element_type_1072 = torch.ops.prims.convert_element_type.default(primals_52, torch.bfloat16);  primals_52 = None
        convert_element_type_1073 = torch.ops.prims.convert_element_type.default(primals_53, torch.bfloat16);  primals_53 = None
        permute_195 = torch.ops.aten.permute.default(convert_element_type_1072, [1, 0]);  convert_element_type_1072 = None
        addmm_1 = torch.ops.aten.addmm.default(convert_element_type_1073, select_3, permute_195);  convert_element_type_1073 = select_3 = permute_195 = None
        convert_element_type_1077 = torch.ops.prims.convert_element_type.default(addmm_1, torch.float32);  addmm_1 = None
        select_4 = torch.ops.aten.select.int(convert_element_type_1077, 1, 0)
        select_5 = torch.ops.aten.select.int(convert_element_type_1077, 1, 1);  convert_element_type_1077 = None
        maximum = torch.ops.aten.maximum.default(select_4, select_5);  select_5 = None
        where_6 = torch.ops.aten.where.self(ge, select_4, maximum);  ge = select_4 = maximum = None
        sigmoid_48 = torch.ops.aten.sigmoid.default(where_6);  where_6 = None
        ne = torch.ops.aten.ne.Scalar(where_4, -100)
        sum_1 = torch.ops.aten.sum.dim_IntList(ne, [-1])
        clamp_min = torch.ops.aten.clamp_min.default(sum_1, 1)
        unsqueeze_192 = torch.ops.aten.unsqueeze.default(clamp_min, -1);  clamp_min = None
        argmax = torch.ops.aten.argmax.default(slice_314, -1)
        eq = torch.ops.aten.eq.Tensor(argmax, where_4);  argmax = None
        bitwise_and_1 = torch.ops.aten.bitwise_and.Tensor(ne, eq);  eq = None
        sum_2 = torch.ops.aten.sum.dim_IntList(bitwise_and_1, [-1])
        eq_1 = torch.ops.aten.eq.Tensor(sum_2, sum_1);  sum_2 = None
        gt_1 = torch.ops.aten.gt.Scalar(sum_1, 0);  sum_1 = None
        bitwise_and_2 = torch.ops.aten.bitwise_and.Tensor(bitwise_and, gt_1);  gt_1 = None
        sum_3 = torch.ops.aten.sum.default(bitwise_and_2)
        convert_element_type_1078 = torch.ops.prims.convert_element_type.default(bitwise_and_1, torch.float32);  bitwise_and_1 = None
        div = torch.ops.aten.div.Tensor(convert_element_type_1078, unsqueeze_192);  convert_element_type_1078 = None
        sum_4 = torch.ops.aten.sum.dim_IntList(div, [-1]);  div = None
        full_default_1 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_7 = torch.ops.aten.where.self(bitwise_and_2, sum_4, full_default_1);  sum_4 = full_default_1 = None
        sum_5 = torch.ops.aten.sum.default(where_7);  where_7 = None
        bitwise_and_3 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_2, eq_1)
        sum_6 = torch.ops.aten.sum.default(bitwise_and_3);  bitwise_and_3 = None
        ge_2 = torch.ops.aten.ge.Scalar(select_1, 0)
        eq_2 = torch.ops.aten.eq.Tensor(ge_2, eq_1);  ge_2 = None
        bitwise_and_4 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_2, eq_2);  eq_2 = None
        sum_7 = torch.ops.aten.sum.default(bitwise_and_4);  bitwise_and_4 = None
        full_default_2 = torch.ops.aten.full.default([], 0, dtype = torch.int32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_8 = torch.ops.aten.where.self(bitwise_and_2, add_154, full_default_2);  bitwise_and_2 = None
        sum_8 = torch.ops.aten.sum.default(where_8);  where_8 = None
        convert_element_type_1079 = torch.ops.prims.convert_element_type.default(slice_314, torch.float64)
        lt_1 = torch.ops.aten.lt.Scalar(convert_element_type_1079, 0)
        sub = torch.ops.aten.sub.Tensor(1, convert_element_type_1079)
        add_309 = torch.ops.aten.add.Tensor(sub, 1e-30);  sub = None
        reciprocal = torch.ops.aten.reciprocal.default(add_309);  add_309 = None
        mul_387 = torch.ops.aten.mul.Tensor(reciprocal, 1);  reciprocal = None
        add_310 = torch.ops.aten.add.Tensor(convert_element_type_1079, 1)
        where_9 = torch.ops.aten.where.self(lt_1, mul_387, add_310);  mul_387 = add_310 = None
        sum_9 = torch.ops.aten.sum.dim_IntList(where_9, [-1], True)
        div_1 = torch.ops.aten.div.Tensor(where_9, sum_9);  where_9 = None
        log = torch.ops.aten.log.default(div_1);  div_1 = None
        where_10 = torch.ops.aten.where.self(ne, where_4, full_default_2);  full_default_2 = None
        convert_element_type_1080 = torch.ops.prims.convert_element_type.default(where_10, torch.int64);  where_10 = None
        unsqueeze_193 = torch.ops.aten.unsqueeze.default(convert_element_type_1080, -1);  convert_element_type_1080 = None
        gather = torch.ops.aten.gather.default(log, -1, unsqueeze_193);  log = None
        squeeze = torch.ops.aten.squeeze.dim(gather, -1);  gather = None
        full_default_4 = torch.ops.aten.full.default([], 0.0, dtype = torch.float64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_11 = torch.ops.aten.where.self(ne, squeeze, full_default_4);  squeeze = full_default_4 = None
        neg_96 = torch.ops.aten.neg.default(where_11);  where_11 = None
        div_2 = torch.ops.aten.div.Tensor(neg_96, unsqueeze_192);  neg_96 = None
        sum_10 = torch.ops.aten.sum.default(div_2);  div_2 = None
        convert_element_type_1081 = torch.ops.prims.convert_element_type.default(eq_1, torch.float32)
        sub_1 = torch.ops.aten.sub.Tensor(1, convert_element_type_1081);  convert_element_type_1081 = None
        mul_388 = torch.ops.aten.mul.Tensor(sub_1, select_1);  sub_1 = None
        full_default_5 = torch.ops.aten.full.default([], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        minimum = torch.ops.aten.minimum.default(full_default_5, select_1)
        abs_1 = torch.ops.aten.abs.default(select_1)
        neg_97 = torch.ops.aten.neg.default(abs_1);  abs_1 = None
        exp = torch.ops.aten.exp.default(neg_97);  neg_97 = None
        log1p = torch.ops.aten.log1p.default(exp);  exp = None
        sub_2 = torch.ops.aten.sub.Tensor(minimum, log1p);  minimum = log1p = None
        sub_3 = torch.ops.aten.sub.Tensor(mul_388, sub_2);  mul_388 = sub_2 = None
        sum_11 = torch.ops.aten.sum.default(sub_3);  sub_3 = None
        sub_4 = torch.ops.aten.sub.Tensor(1, sigmoid_48)
        mul_389 = torch.ops.aten.mul.Tensor(sub_4, select_2);  sub_4 = None
        minimum_1 = torch.ops.aten.minimum.default(full_default_5, select_2);  full_default_5 = None
        abs_2 = torch.ops.aten.abs.default(select_2)
        neg_98 = torch.ops.aten.neg.default(abs_2);  abs_2 = None
        exp_1 = torch.ops.aten.exp.default(neg_98);  neg_98 = None
        log1p_1 = torch.ops.aten.log1p.default(exp_1);  exp_1 = None
        sub_5 = torch.ops.aten.sub.Tensor(minimum_1, log1p_1);  minimum_1 = log1p_1 = None
        sub_6 = torch.ops.aten.sub.Tensor(mul_389, sub_5);  mul_389 = sub_5 = None
        sum_12 = torch.ops.aten.sum.default(sub_6);  sub_6 = None
        add_311 = torch.ops.aten.add.Tensor(sum_11, sum_12)
        mul_390 = torch.ops.aten.mul.Tensor(add_311, 0.5);  add_311 = None
        add_312 = torch.ops.aten.add.Tensor(sum_10, mul_390);  mul_390 = None
        logical_not = torch.ops.aten.logical_not.default(bitwise_and)
        any_1 = torch.ops.aten.any.dims(logical_not);  logical_not = None
        logical_not_1 = torch.ops.aten.logical_not.default(any_1);  any_1 = None
        sigmoid_49 = torch.ops.aten.sigmoid.default(select_2)
        sub_7 = torch.ops.aten.sub.Tensor(sigmoid_49, sigmoid_48);  sigmoid_49 = None
        permute_196 = torch.ops.aten.permute.default(permute_97, [1, 0]);  permute_97 = None
        permute_206 = torch.ops.aten.permute.default(permute_47, [1, 0]);  permute_47 = None
        permute_211 = torch.ops.aten.permute.default(permute_46, [1, 0]);  permute_46 = None
        permute_215 = torch.ops.aten.permute.default(permute_45, [1, 0]);  permute_45 = None
        permute_222 = torch.ops.aten.permute.default(permute_44, [1, 0]);  permute_44 = None
        permute_226 = torch.ops.aten.permute.default(permute_43, [1, 0]);  permute_43 = None
        permute_231 = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None
        permute_235 = torch.ops.aten.permute.default(permute_41, [1, 0]);  permute_41 = None
        permute_242 = torch.ops.aten.permute.default(permute_40, [1, 0]);  permute_40 = None
        permute_246 = torch.ops.aten.permute.default(permute_39, [1, 0]);  permute_39 = None
        permute_251 = torch.ops.aten.permute.default(permute_38, [1, 0]);  permute_38 = None
        permute_255 = torch.ops.aten.permute.default(permute_37, [1, 0]);  permute_37 = None
        permute_262 = torch.ops.aten.permute.default(permute_36, [1, 0]);  permute_36 = None
        permute_266 = torch.ops.aten.permute.default(permute_35, [1, 0]);  permute_35 = None
        permute_271 = torch.ops.aten.permute.default(permute_34, [1, 0]);  permute_34 = None
        permute_275 = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None
        permute_282 = torch.ops.aten.permute.default(permute_32, [1, 0]);  permute_32 = None
        permute_286 = torch.ops.aten.permute.default(permute_15, [1, 0]);  permute_15 = None
        permute_291 = torch.ops.aten.permute.default(permute_14, [1, 0]);  permute_14 = None
        permute_295 = torch.ops.aten.permute.default(permute_13, [1, 0]);  permute_13 = None
        permute_302 = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
        permute_306 = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
        permute_311 = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
        permute_315 = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
        permute_322 = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
        permute_326 = torch.ops.aten.permute.default(permute_7, [1, 0]);  permute_7 = None
        permute_331 = torch.ops.aten.permute.default(permute_6, [1, 0]);  permute_6 = None
        permute_335 = torch.ops.aten.permute.default(permute_5, [1, 0]);  permute_5 = None
        permute_342 = torch.ops.aten.permute.default(permute_4, [1, 0]);  permute_4 = None
        permute_346 = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
        permute_351 = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        permute_355 = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
        permute_362 = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        copy_ = torch.ops.aten.copy_.default(primals_16, index_1);  primals_16 = index_1 = copy_ = None
        copy__1 = torch.ops.aten.copy_.default(primals_18, where_5);  primals_18 = copy__1 = None
        return (add_312, sum_3, sum_5, sum_6, sum_7, sum_8, sum_10, sum_11, sum_12, logical_not_1, slice_314, select_1, select_2, sigmoid_48, convert_element_type_441, convert_element_type_529, where_3, where_4, where_5, bitwise_and, add_154, convert_element_type_538, primals_13, primals_14, where_3, view_166, slice_217, convert_element_type_359, convert_element_type_360, getitem_96, getitem_97, getitem_99, add_107, rsqrt_32, view_172, getitem_100, getitem_101, view_174, add_109, rsqrt_33, view_176, slice_230, convert_element_type_381, convert_element_type_382, getitem_102, getitem_103, getitem_105, add_113, rsqrt_34, view_182, getitem_106, getitem_107, view_184, add_115, rsqrt_35, view_186, slice_243, convert_element_type_403, convert_element_type_404, getitem_108, getitem_109, getitem_111, add_119, rsqrt_36, view_192, getitem_112, getitem_113, view_194, add_121, rsqrt_37, view_196, slice_256, convert_element_type_425, convert_element_type_426, getitem_114, getitem_115, getitem_117, add_125, rsqrt_38, view_202, getitem_118, getitem_119, view_204, add_127, rsqrt_39, view_206, slice_269, convert_element_type_447, convert_element_type_448, getitem_120, getitem_121, getitem_123, add_132, rsqrt_40, view_212, getitem_124, getitem_125, view_214, add_134, rsqrt_41, view_216, slice_282, convert_element_type_469, convert_element_type_470, getitem_126, getitem_127, getitem_129, add_138, rsqrt_42, view_222, getitem_130, getitem_131, view_224, add_140, rsqrt_43, view_226, slice_295, convert_element_type_491, convert_element_type_492, getitem_132, getitem_133, getitem_135, add_144, rsqrt_44, view_232, getitem_136, getitem_137, view_234, add_146, rsqrt_45, view_236, slice_308, convert_element_type_513, convert_element_type_514, getitem_138, getitem_139, getitem_141, add_150, rsqrt_46, view_242, getitem_142, getitem_143, view_244, add_152, rsqrt_47, permute_96, view_246, select, select_1, ne, unsqueeze_192, eq_1, convert_element_type_1079, lt_1, sum_9, unsqueeze_193, sub_7, permute_196, permute_206, permute_211, permute_215, permute_222, permute_226, permute_231, permute_235, permute_242, permute_246, permute_251, permute_255, permute_262, permute_266, permute_271, permute_275, permute_282, permute_286, permute_291, permute_295, permute_302, permute_306, permute_311, permute_315, permute_322, permute_326, permute_331, permute_335, permute_342, permute_346, permute_351, permute_355, permute_362)
        
def load_args(reader):
    buf0 = reader.storage(None, 128, device=device(type='cuda', index=0), dtype_hint=torch.bool)
    reader.tensor(buf0, (128,), dtype=torch.bool, is_leaf=True)  # primals_1
    buf1 = reader.storage(None, 256, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf1, (128,), dtype=torch.bfloat16, is_leaf=True)  # primals_2
    buf2 = reader.storage(None, 29523968, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf2, (128, 901, 128), dtype=torch.bfloat16, is_leaf=True)  # primals_3
    buf3 = reader.storage(None, 256, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf3, (128,), dtype=torch.bfloat16, is_leaf=True)  # primals_4
    buf4 = reader.storage(None, 29523968, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf4, (128, 901, 128), dtype=torch.bfloat16, is_leaf=True)  # primals_5
    buf5 = reader.storage(None, 512, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf5, (128,), dtype=torch.int32, is_leaf=True)  # primals_6
    buf6 = reader.storage(None, 460800, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf6, (128, 900), dtype=torch.int32, is_leaf=True)  # primals_7
    buf7 = reader.storage(None, 460800, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf7, (128, 900), dtype=torch.int32, is_leaf=True)  # primals_8
    buf8 = reader.storage(None, 460800, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf8, (128, 900), dtype=torch.int32, is_leaf=True)  # primals_9
    buf9 = reader.storage(None, 460800, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf9, (128, 900), dtype=torch.int32, is_leaf=True)  # primals_10
    buf10 = reader.storage(None, 512, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf10, (128,), dtype=torch.int32, is_leaf=True)  # primals_11
    buf11 = reader.storage(None, 512, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf11, (128,), dtype=torch.int32, is_leaf=True)  # primals_12
    buf12 = reader.storage(None, 230656, device=device(type='cuda', index=0))
    reader.tensor(buf12, (901, 64), is_leaf=True)  # primals_13
    buf13 = reader.storage(None, 230656, device=device(type='cuda', index=0))
    reader.tensor(buf13, (901, 64), is_leaf=True)  # primals_14
    buf14 = reader.storage(None, 5120, device=device(type='cuda', index=0))
    reader.tensor(buf14, (10, 128), is_leaf=True)  # primals_15
    buf15 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf15, (128, 128), is_leaf=True)  # primals_16
    buf16 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf16, (1, 128), is_leaf=True)  # primals_17
    buf17 = reader.storage(None, 512, device=device(type='cuda', index=0), dtype_hint=torch.int32)
    reader.tensor(buf17, (128,), dtype=torch.int32, is_leaf=True)  # primals_18
    buf18 = reader.storage(None, 196608, device=device(type='cuda', index=0))
    reader.tensor(buf18, (384, 128), is_leaf=True)  # primals_19
    buf19 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf19, (128, 128), is_leaf=True)  # primals_20
    buf20 = reader.storage(None, 524288, device=device(type='cuda', index=0))
    reader.tensor(buf20, (1024, 128), is_leaf=True)  # primals_21
    buf21 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf21, (128, 512), is_leaf=True)  # primals_22
    buf22 = reader.storage(None, 196608, device=device(type='cuda', index=0))
    reader.tensor(buf22, (384, 128), is_leaf=True)  # primals_23
    buf23 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf23, (128, 128), is_leaf=True)  # primals_24
    buf24 = reader.storage(None, 524288, device=device(type='cuda', index=0))
    reader.tensor(buf24, (1024, 128), is_leaf=True)  # primals_25
    buf25 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf25, (128, 512), is_leaf=True)  # primals_26
    buf26 = reader.storage(None, 196608, device=device(type='cuda', index=0))
    reader.tensor(buf26, (384, 128), is_leaf=True)  # primals_27
    buf27 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf27, (128, 128), is_leaf=True)  # primals_28
    buf28 = reader.storage(None, 524288, device=device(type='cuda', index=0))
    reader.tensor(buf28, (1024, 128), is_leaf=True)  # primals_29
    buf29 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf29, (128, 512), is_leaf=True)  # primals_30
    buf30 = reader.storage(None, 196608, device=device(type='cuda', index=0))
    reader.tensor(buf30, (384, 128), is_leaf=True)  # primals_31
    buf31 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf31, (128, 128), is_leaf=True)  # primals_32
    buf32 = reader.storage(None, 524288, device=device(type='cuda', index=0))
    reader.tensor(buf32, (1024, 128), is_leaf=True)  # primals_33
    buf33 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf33, (128, 512), is_leaf=True)  # primals_34
    buf34 = reader.storage(None, 196608, device=device(type='cuda', index=0))
    reader.tensor(buf34, (384, 128), is_leaf=True)  # primals_35
    buf35 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf35, (128, 128), is_leaf=True)  # primals_36
    buf36 = reader.storage(None, 524288, device=device(type='cuda', index=0))
    reader.tensor(buf36, (1024, 128), is_leaf=True)  # primals_37
    buf37 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf37, (128, 512), is_leaf=True)  # primals_38
    buf38 = reader.storage(None, 196608, device=device(type='cuda', index=0))
    reader.tensor(buf38, (384, 128), is_leaf=True)  # primals_39
    buf39 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf39, (128, 128), is_leaf=True)  # primals_40
    buf40 = reader.storage(None, 524288, device=device(type='cuda', index=0))
    reader.tensor(buf40, (1024, 128), is_leaf=True)  # primals_41
    buf41 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf41, (128, 512), is_leaf=True)  # primals_42
    buf42 = reader.storage(None, 196608, device=device(type='cuda', index=0))
    reader.tensor(buf42, (384, 128), is_leaf=True)  # primals_43
    buf43 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf43, (128, 128), is_leaf=True)  # primals_44
    buf44 = reader.storage(None, 524288, device=device(type='cuda', index=0))
    reader.tensor(buf44, (1024, 128), is_leaf=True)  # primals_45
    buf45 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf45, (128, 512), is_leaf=True)  # primals_46
    buf46 = reader.storage(None, 196608, device=device(type='cuda', index=0))
    reader.tensor(buf46, (384, 128), is_leaf=True)  # primals_47
    buf47 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf47, (128, 128), is_leaf=True)  # primals_48
    buf48 = reader.storage(None, 524288, device=device(type='cuda', index=0))
    reader.tensor(buf48, (1024, 128), is_leaf=True)  # primals_49
    buf49 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf49, (128, 512), is_leaf=True)  # primals_50
    buf50 = reader.storage(None, 5120, device=device(type='cuda', index=0))
    reader.tensor(buf50, (10, 128), is_leaf=True)  # primals_51
    buf51 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf51, (2, 128), is_leaf=True)  # primals_52
    buf52 = reader.storage(None, 8, device=device(type='cuda', index=0))
    reader.tensor(buf52, (2,), is_leaf=True)  # primals_53
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)