class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "b8[128]", primals_2: "bf16[128]", primals_3: "bf16[128, 901, 128]", primals_4: "bf16[128]", primals_5: "bf16[128, 901, 128]", primals_6: "i32[128]", primals_7: "i32[128, 900]", primals_8: "i32[128, 900]", primals_9: "i32[128, 900]", primals_10: "i32[128, 900]", primals_11: "i32[128]", primals_12: "i32[128]", primals_13: "f32[901, 64]", primals_14: "f32[901, 64]", primals_15: "f32[10, 128]", primals_16: "f32[128, 128]", primals_17: "f32[1, 128]", primals_18: "i32[128]", primals_19: "f32[384, 128]", primals_20: "f32[128, 128]", primals_21: "f32[1024, 128]", primals_22: "f32[128, 512]", primals_23: "f32[384, 128]", primals_24: "f32[128, 128]", primals_25: "f32[1024, 128]", primals_26: "f32[128, 512]", primals_27: "f32[384, 128]", primals_28: "f32[128, 128]", primals_29: "f32[1024, 128]", primals_30: "f32[128, 512]", primals_31: "f32[384, 128]", primals_32: "f32[128, 128]", primals_33: "f32[1024, 128]", primals_34: "f32[128, 512]", primals_35: "f32[384, 128]", primals_36: "f32[128, 128]", primals_37: "f32[1024, 128]", primals_38: "f32[128, 512]", primals_39: "f32[384, 128]", primals_40: "f32[128, 128]", primals_41: "f32[1024, 128]", primals_42: "f32[128, 512]", primals_43: "f32[384, 128]", primals_44: "f32[128, 128]", primals_45: "f32[1024, 128]", primals_46: "f32[128, 512]", primals_47: "f32[384, 128]", primals_48: "f32[128, 128]", primals_49: "f32[1024, 128]", primals_50: "f32[128, 512]", primals_51: "f32[10, 128]", primals_52: "f32[2, 128]", primals_53: "f32[2]"):
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:176 in reset_carry, code: z_H=torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_H),
        view: "b8[128, 1, 1]" = torch.ops.aten.reshape.default(primals_1, [-1, 1, 1])
        where: "bf16[128, 901, 128]" = torch.ops.aten.where.self(view, primals_2, primals_3);  primals_2 = primals_3 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:177 in reset_carry, code: z_L=torch.where(reset_flag.view(-1, 1, 1), self.L_init, carry.z_L),
        where_1: "bf16[128, 901, 128]" = torch.ops.aten.where.self(view, primals_4, primals_5);  view = primals_4 = primals_5 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:244 in forward, code: new_steps = torch.where(carry.halted, 0, carry.steps)
        full_default: "i32[]" = torch.ops.aten.full.default([], 0, dtype = torch.int32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2: "i32[128]" = torch.ops.aten.where.self(primals_1, full_default, primals_6);  full_default = primals_6 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:246 in <dictcomp>, code: new_current_data = {k: torch.where(carry.halted.view((-1, ) + (1, ) * (batch[k].ndim - 1)), batch[k], v) for k, v in carry.current_data.items()}
        view_2: "b8[128, 1]" = torch.ops.aten.reshape.default(primals_1, [-1, 1])
        where_3: "i32[128, 900]" = torch.ops.aten.where.self(view_2, primals_8, primals_7);  primals_8 = primals_7 = None
        where_4: "i32[128, 900]" = torch.ops.aten.where.self(view_2, primals_10, primals_9);  view_2 = primals_10 = primals_9 = None
        view_4: "b8[128]" = torch.ops.aten.reshape.default(primals_1, [-1]);  primals_1 = None
        where_5: "i32[128]" = torch.ops.aten.where.self(view_4, primals_12, primals_11);  view_4 = primals_12 = primals_11 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:77 in forward, code: return F.embedding(input, self.embedding_weight.to(self.cast_to))
        convert_element_type: "bf16[10, 128]" = torch.ops.prims.convert_element_type.default(primals_15, torch.bfloat16)
        embedding: "bf16[128, 900, 128]" = torch.ops.aten.embedding.default(convert_element_type, where_3);  convert_element_type = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\sparse_embedding.py:35 in forward, code: self.local_weights.copy_(self.weights[inputs])
        index: "f32[128, 128]" = torch.ops.aten.index.Tensor(primals_17, [where_5])
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\sparse_embedding.py:38 in forward, code: return self.local_weights.to(self.cast_to)
        convert_element_type_1: "bf16[128, 128]" = torch.ops.prims.convert_element_type.default(index, torch.bfloat16);  index = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:158 in _input_embeddings, code: embedding = torch.cat((puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding), dim=-2)
        view_5: "bf16[128, 1, 128]" = torch.ops.aten.reshape.default(convert_element_type_1, [-1, 1, 128]);  convert_element_type_1 = None
        cat: "bf16[128, 901, 128]" = torch.ops.aten.cat.default([view_5, embedding], -2);  view_5 = embedding = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:166 in _input_embeddings, code: return self.embed_scale * embedding
        mul: "bf16[128, 901, 128]" = torch.ops.aten.mul.Tensor(cat, 11.313708498984761);  cat = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:195 in forward, code: z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
        add: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(where, mul)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:94 in forward, code: hidden_states = hidden_states + input_injection
        add_1: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(where_1, add);  where_1 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        convert_element_type_2: "bf16[384, 128]" = torch.ops.prims.convert_element_type.default(primals_19, torch.bfloat16)
        permute: "bf16[128, 384]" = torch.ops.aten.permute.default(convert_element_type_2, [1, 0]);  convert_element_type_2 = None
        view_6: "bf16[115328, 128]" = torch.ops.aten.reshape.default(add_1, [115328, 128])
        mm: "bf16[115328, 384]" = torch.ops.aten.mm.default(view_6, permute);  view_6 = None
        view_7: "bf16[128, 901, 384]" = torch.ops.aten.reshape.default(mm, [128, 901, 384]);  mm = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:119 in forward, code: qkv = qkv.view(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        view_8: "bf16[128, 901, 6, 64]" = torch.ops.aten.reshape.default(view_7, [128, 901, 6, 64]);  view_7 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:120 in forward, code: query = qkv[:, :, :self.num_heads]
        slice_3: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_8, 2, 0, 2)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:121 in forward, code: key = qkv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
        slice_6: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_8, 2, 2, 4)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:122 in forward, code: value = qkv[:, :, self.num_heads + self.num_key_value_heads:]
        slice_9: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_8, 2, 4, 9223372036854775807);  view_8 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:34 in apply_rotary_pos_emb, code: q = q.to(cos.dtype)
        convert_element_type_5: "f32[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(slice_3, torch.float32);  slice_3 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:35 in apply_rotary_pos_emb, code: k = k.to(cos.dtype)
        convert_element_type_6: "f32[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(slice_6, torch.float32);  slice_6 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:37 in apply_rotary_pos_emb, code: q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
        unsqueeze: "f32[901, 1, 64]" = torch.ops.aten.unsqueeze.default(primals_13, -2)
        mul_1: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_5, unsqueeze)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:25 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_10: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_5, 3, 0, 32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:26 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_11: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_5, 3, 32, 9223372036854775807);  convert_element_type_5 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:27 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg: "f32[128, 901, 2, 32]" = torch.ops.aten.neg.default(slice_11);  slice_11 = None
        cat_1: "f32[128, 901, 2, 64]" = torch.ops.aten.cat.default([neg, slice_10], -1);  neg = slice_10 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:37 in apply_rotary_pos_emb, code: q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
        unsqueeze_1: "f32[901, 1, 64]" = torch.ops.aten.unsqueeze.default(primals_14, -2)
        mul_2: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(cat_1, unsqueeze_1);  cat_1 = None
        add_2: "f32[128, 901, 2, 64]" = torch.ops.aten.add.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:38 in apply_rotary_pos_emb, code: k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
        mul_3: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_6, unsqueeze)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:25 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_12: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_6, 3, 0, 32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:26 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_13: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_6, 3, 32, 9223372036854775807);  convert_element_type_6 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:27 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_1: "f32[128, 901, 2, 32]" = torch.ops.aten.neg.default(slice_13);  slice_13 = None
        cat_2: "f32[128, 901, 2, 64]" = torch.ops.aten.cat.default([neg_1, slice_12], -1);  neg_1 = slice_12 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:38 in apply_rotary_pos_emb, code: k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
        mul_4: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(cat_2, unsqueeze_1);  cat_2 = None
        add_3: "f32[128, 901, 2, 64]" = torch.ops.aten.add.Tensor(mul_3, mul_4);  mul_3 = mul_4 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:40 in apply_rotary_pos_emb, code: return q_embed.to(orig_dtype), k_embed.to(orig_dtype)
        convert_element_type_7: "bf16[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(add_2, torch.bfloat16);  add_2 = None
        convert_element_type_8: "bf16[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(add_3, torch.bfloat16);  add_3 = None
        
         # File: C:\Users\arroy\anaconda3\envs\hrm_v1\lib\site-packages\flash_attn\flash_attn_interface.py:839 in forward, code: out_padded, softmax_lse, S_dmask, rng_state = _wrapped_flash_attn_forward(
        _flash_attn_forward = torch.ops.flash_attn._flash_attn_forward.default(convert_element_type_7, convert_element_type_8, slice_9, 0.0, 0.125, False, -1, -1, 0.0, None, False);  convert_element_type_7 = convert_element_type_8 = slice_9 = None
        getitem: "bf16[128, 901, 2, 64]" = _flash_attn_forward[0];  _flash_attn_forward = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:135 in forward, code: attn_output = attn_output.view(batch_size, seq_len, self.output_size)  # type: ignore
        view_9: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(getitem, [128, 901, 128]);  getitem = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        convert_element_type_9: "bf16[128, 128]" = torch.ops.prims.convert_element_type.default(primals_20, torch.bfloat16)
        permute_1: "bf16[128, 128]" = torch.ops.aten.permute.default(convert_element_type_9, [1, 0]);  convert_element_type_9 = None
        view_10: "bf16[115328, 128]" = torch.ops.aten.reshape.default(view_9, [115328, 128]);  view_9 = None
        mm_1: "bf16[115328, 128]" = torch.ops.aten.mm.default(view_10, permute_1);  view_10 = None
        view_11: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(mm_1, [128, 901, 128]);  mm_1 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:80 in forward, code: hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        add_4: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(add_1, view_11);  add_1 = view_11 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:154 in rms_norm, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_12: "f32[128, 901, 128]" = torch.ops.prims.convert_element_type.default(add_4, torch.float32);  add_4 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:156 in rms_norm, code: variance = hidden_states.square().mean(-1, keepdim=True)
        pow_1: "f32[128, 901, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_12, 2)
        mean: "f32[128, 901, 1]" = torch.ops.aten.mean.dim(pow_1, [-1], True);  pow_1 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:157 in rms_norm, code: hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        add_5: "f32[128, 901, 1]" = torch.ops.aten.add.Tensor(mean, 1e-05);  mean = None
        rsqrt: "f32[128, 901, 1]" = torch.ops.aten.rsqrt.default(add_5);  add_5 = None
        mul_5: "f32[128, 901, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_12, rsqrt);  convert_element_type_12 = rsqrt = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:158 in rms_norm, code: return hidden_states.to(input_dtype)
        convert_element_type_13: "bf16[128, 901, 128]" = torch.ops.prims.convert_element_type.default(mul_5, torch.bfloat16);  mul_5 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        convert_element_type_14: "bf16[1024, 128]" = torch.ops.prims.convert_element_type.default(primals_21, torch.bfloat16)
        permute_2: "bf16[128, 1024]" = torch.ops.aten.permute.default(convert_element_type_14, [1, 0]);  convert_element_type_14 = None
        view_12: "bf16[115328, 128]" = torch.ops.aten.reshape.default(convert_element_type_13, [115328, 128])
        mm_2: "bf16[115328, 1024]" = torch.ops.aten.mm.default(view_12, permute_2);  view_12 = None
        view_13: "bf16[128, 901, 1024]" = torch.ops.aten.reshape.default(mm_2, [128, 901, 1024]);  mm_2 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:148 in forward, code: gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        split = torch.ops.aten.split.Tensor(view_13, 512, -1);  view_13 = None
        getitem_4: "bf16[128, 901, 512]" = split[0]
        getitem_5: "bf16[128, 901, 512]" = split[1];  split = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:149 in forward, code: return self.down_proj(F.silu(gate) * up)
        convert_element_type_17: "f32[128, 901, 512]" = torch.ops.prims.convert_element_type.default(getitem_4, torch.float32);  getitem_4 = None
        sigmoid: "f32[128, 901, 512]" = torch.ops.aten.sigmoid.default(convert_element_type_17)
        mul_6: "f32[128, 901, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_17, sigmoid);  convert_element_type_17 = sigmoid = None
        convert_element_type_18: "bf16[128, 901, 512]" = torch.ops.prims.convert_element_type.default(mul_6, torch.bfloat16);  mul_6 = None
        mul_7: "bf16[128, 901, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_18, getitem_5);  convert_element_type_18 = getitem_5 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        convert_element_type_19: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(primals_22, torch.bfloat16)
        permute_3: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_19, [1, 0]);  convert_element_type_19 = None
        view_14: "bf16[115328, 512]" = torch.ops.aten.reshape.default(mul_7, [115328, 512]);  mul_7 = None
        mm_3: "bf16[115328, 128]" = torch.ops.aten.mm.default(view_14, permute_3);  view_14 = None
        view_15: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(mm_3, [128, 901, 128]);  mm_3 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:82 in forward, code: hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)
        add_6: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(convert_element_type_13, view_15);  convert_element_type_13 = view_15 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:154 in rms_norm, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_22: "f32[128, 901, 128]" = torch.ops.prims.convert_element_type.default(add_6, torch.float32);  add_6 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:156 in rms_norm, code: variance = hidden_states.square().mean(-1, keepdim=True)
        pow_2: "f32[128, 901, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_22, 2)
        mean_1: "f32[128, 901, 1]" = torch.ops.aten.mean.dim(pow_2, [-1], True);  pow_2 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:157 in rms_norm, code: hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        add_7: "f32[128, 901, 1]" = torch.ops.aten.add.Tensor(mean_1, 1e-05);  mean_1 = None
        rsqrt_1: "f32[128, 901, 1]" = torch.ops.aten.rsqrt.default(add_7);  add_7 = None
        mul_8: "f32[128, 901, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_22, rsqrt_1);  convert_element_type_22 = rsqrt_1 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:158 in rms_norm, code: return hidden_states.to(input_dtype)
        convert_element_type_23: "bf16[128, 901, 128]" = torch.ops.prims.convert_element_type.default(mul_8, torch.bfloat16);  mul_8 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        convert_element_type_24: "bf16[384, 128]" = torch.ops.prims.convert_element_type.default(primals_23, torch.bfloat16)
        permute_4: "bf16[128, 384]" = torch.ops.aten.permute.default(convert_element_type_24, [1, 0]);  convert_element_type_24 = None
        view_16: "bf16[115328, 128]" = torch.ops.aten.reshape.default(convert_element_type_23, [115328, 128])
        mm_4: "bf16[115328, 384]" = torch.ops.aten.mm.default(view_16, permute_4);  view_16 = None
        view_17: "bf16[128, 901, 384]" = torch.ops.aten.reshape.default(mm_4, [128, 901, 384]);  mm_4 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:119 in forward, code: qkv = qkv.view(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        view_18: "bf16[128, 901, 6, 64]" = torch.ops.aten.reshape.default(view_17, [128, 901, 6, 64]);  view_17 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:120 in forward, code: query = qkv[:, :, :self.num_heads]
        slice_16: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_18, 2, 0, 2)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:121 in forward, code: key = qkv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
        slice_19: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_18, 2, 2, 4)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:122 in forward, code: value = qkv[:, :, self.num_heads + self.num_key_value_heads:]
        slice_22: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_18, 2, 4, 9223372036854775807);  view_18 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:34 in apply_rotary_pos_emb, code: q = q.to(cos.dtype)
        convert_element_type_27: "f32[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(slice_16, torch.float32);  slice_16 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:35 in apply_rotary_pos_emb, code: k = k.to(cos.dtype)
        convert_element_type_28: "f32[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(slice_19, torch.float32);  slice_19 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:37 in apply_rotary_pos_emb, code: q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
        mul_9: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_27, unsqueeze)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:25 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_23: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_27, 3, 0, 32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:26 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_24: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_27, 3, 32, 9223372036854775807);  convert_element_type_27 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:27 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_2: "f32[128, 901, 2, 32]" = torch.ops.aten.neg.default(slice_24);  slice_24 = None
        cat_3: "f32[128, 901, 2, 64]" = torch.ops.aten.cat.default([neg_2, slice_23], -1);  neg_2 = slice_23 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:37 in apply_rotary_pos_emb, code: q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
        mul_10: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(cat_3, unsqueeze_1);  cat_3 = None
        add_8: "f32[128, 901, 2, 64]" = torch.ops.aten.add.Tensor(mul_9, mul_10);  mul_9 = mul_10 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:38 in apply_rotary_pos_emb, code: k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
        mul_11: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_28, unsqueeze)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:25 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_25: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_28, 3, 0, 32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:26 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_26: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_28, 3, 32, 9223372036854775807);  convert_element_type_28 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:27 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_3: "f32[128, 901, 2, 32]" = torch.ops.aten.neg.default(slice_26);  slice_26 = None
        cat_4: "f32[128, 901, 2, 64]" = torch.ops.aten.cat.default([neg_3, slice_25], -1);  neg_3 = slice_25 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:38 in apply_rotary_pos_emb, code: k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
        mul_12: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(cat_4, unsqueeze_1);  cat_4 = None
        add_9: "f32[128, 901, 2, 64]" = torch.ops.aten.add.Tensor(mul_11, mul_12);  mul_11 = mul_12 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:40 in apply_rotary_pos_emb, code: return q_embed.to(orig_dtype), k_embed.to(orig_dtype)
        convert_element_type_29: "bf16[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(add_8, torch.bfloat16);  add_8 = None
        convert_element_type_30: "bf16[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(add_9, torch.bfloat16);  add_9 = None
        
         # File: C:\Users\arroy\anaconda3\envs\hrm_v1\lib\site-packages\flash_attn\flash_attn_interface.py:839 in forward, code: out_padded, softmax_lse, S_dmask, rng_state = _wrapped_flash_attn_forward(
        _flash_attn_forward_1 = torch.ops.flash_attn._flash_attn_forward.default(convert_element_type_29, convert_element_type_30, slice_22, 0.0, 0.125, False, -1, -1, 0.0, None, False);  convert_element_type_29 = convert_element_type_30 = slice_22 = None
        getitem_6: "bf16[128, 901, 2, 64]" = _flash_attn_forward_1[0];  _flash_attn_forward_1 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:135 in forward, code: attn_output = attn_output.view(batch_size, seq_len, self.output_size)  # type: ignore
        view_19: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(getitem_6, [128, 901, 128]);  getitem_6 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        convert_element_type_31: "bf16[128, 128]" = torch.ops.prims.convert_element_type.default(primals_24, torch.bfloat16)
        permute_5: "bf16[128, 128]" = torch.ops.aten.permute.default(convert_element_type_31, [1, 0]);  convert_element_type_31 = None
        view_20: "bf16[115328, 128]" = torch.ops.aten.reshape.default(view_19, [115328, 128]);  view_19 = None
        mm_5: "bf16[115328, 128]" = torch.ops.aten.mm.default(view_20, permute_5);  view_20 = None
        view_21: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(mm_5, [128, 901, 128]);  mm_5 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:80 in forward, code: hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        add_10: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(convert_element_type_23, view_21);  convert_element_type_23 = view_21 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:154 in rms_norm, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_34: "f32[128, 901, 128]" = torch.ops.prims.convert_element_type.default(add_10, torch.float32);  add_10 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:156 in rms_norm, code: variance = hidden_states.square().mean(-1, keepdim=True)
        pow_3: "f32[128, 901, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_34, 2)
        mean_2: "f32[128, 901, 1]" = torch.ops.aten.mean.dim(pow_3, [-1], True);  pow_3 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:157 in rms_norm, code: hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        add_11: "f32[128, 901, 1]" = torch.ops.aten.add.Tensor(mean_2, 1e-05);  mean_2 = None
        rsqrt_2: "f32[128, 901, 1]" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
        mul_13: "f32[128, 901, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_34, rsqrt_2);  convert_element_type_34 = rsqrt_2 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:158 in rms_norm, code: return hidden_states.to(input_dtype)
        convert_element_type_35: "bf16[128, 901, 128]" = torch.ops.prims.convert_element_type.default(mul_13, torch.bfloat16);  mul_13 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        convert_element_type_36: "bf16[1024, 128]" = torch.ops.prims.convert_element_type.default(primals_25, torch.bfloat16)
        permute_6: "bf16[128, 1024]" = torch.ops.aten.permute.default(convert_element_type_36, [1, 0]);  convert_element_type_36 = None
        view_22: "bf16[115328, 128]" = torch.ops.aten.reshape.default(convert_element_type_35, [115328, 128])
        mm_6: "bf16[115328, 1024]" = torch.ops.aten.mm.default(view_22, permute_6);  view_22 = None
        view_23: "bf16[128, 901, 1024]" = torch.ops.aten.reshape.default(mm_6, [128, 901, 1024]);  mm_6 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:148 in forward, code: gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        split_1 = torch.ops.aten.split.Tensor(view_23, 512, -1);  view_23 = None
        getitem_10: "bf16[128, 901, 512]" = split_1[0]
        getitem_11: "bf16[128, 901, 512]" = split_1[1];  split_1 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:149 in forward, code: return self.down_proj(F.silu(gate) * up)
        convert_element_type_39: "f32[128, 901, 512]" = torch.ops.prims.convert_element_type.default(getitem_10, torch.float32);  getitem_10 = None
        sigmoid_1: "f32[128, 901, 512]" = torch.ops.aten.sigmoid.default(convert_element_type_39)
        mul_14: "f32[128, 901, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_39, sigmoid_1);  convert_element_type_39 = sigmoid_1 = None
        convert_element_type_40: "bf16[128, 901, 512]" = torch.ops.prims.convert_element_type.default(mul_14, torch.bfloat16);  mul_14 = None
        mul_15: "bf16[128, 901, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_40, getitem_11);  convert_element_type_40 = getitem_11 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        convert_element_type_41: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(primals_26, torch.bfloat16)
        permute_7: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_41, [1, 0]);  convert_element_type_41 = None
        view_24: "bf16[115328, 512]" = torch.ops.aten.reshape.default(mul_15, [115328, 512]);  mul_15 = None
        mm_7: "bf16[115328, 128]" = torch.ops.aten.mm.default(view_24, permute_7);  view_24 = None
        view_25: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(mm_7, [128, 901, 128]);  mm_7 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:82 in forward, code: hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)
        add_12: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(convert_element_type_35, view_25);  convert_element_type_35 = view_25 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:154 in rms_norm, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_44: "f32[128, 901, 128]" = torch.ops.prims.convert_element_type.default(add_12, torch.float32);  add_12 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:156 in rms_norm, code: variance = hidden_states.square().mean(-1, keepdim=True)
        pow_4: "f32[128, 901, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_44, 2)
        mean_3: "f32[128, 901, 1]" = torch.ops.aten.mean.dim(pow_4, [-1], True);  pow_4 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:157 in rms_norm, code: hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        add_13: "f32[128, 901, 1]" = torch.ops.aten.add.Tensor(mean_3, 1e-05);  mean_3 = None
        rsqrt_3: "f32[128, 901, 1]" = torch.ops.aten.rsqrt.default(add_13);  add_13 = None
        mul_16: "f32[128, 901, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_44, rsqrt_3);  convert_element_type_44 = rsqrt_3 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:158 in rms_norm, code: return hidden_states.to(input_dtype)
        convert_element_type_45: "bf16[128, 901, 128]" = torch.ops.prims.convert_element_type.default(mul_16, torch.bfloat16);  mul_16 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        convert_element_type_46: "bf16[384, 128]" = torch.ops.prims.convert_element_type.default(primals_27, torch.bfloat16)
        permute_8: "bf16[128, 384]" = torch.ops.aten.permute.default(convert_element_type_46, [1, 0]);  convert_element_type_46 = None
        view_26: "bf16[115328, 128]" = torch.ops.aten.reshape.default(convert_element_type_45, [115328, 128])
        mm_8: "bf16[115328, 384]" = torch.ops.aten.mm.default(view_26, permute_8);  view_26 = None
        view_27: "bf16[128, 901, 384]" = torch.ops.aten.reshape.default(mm_8, [128, 901, 384]);  mm_8 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:119 in forward, code: qkv = qkv.view(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        view_28: "bf16[128, 901, 6, 64]" = torch.ops.aten.reshape.default(view_27, [128, 901, 6, 64]);  view_27 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:120 in forward, code: query = qkv[:, :, :self.num_heads]
        slice_29: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_28, 2, 0, 2)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:121 in forward, code: key = qkv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
        slice_32: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_28, 2, 2, 4)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:122 in forward, code: value = qkv[:, :, self.num_heads + self.num_key_value_heads:]
        slice_35: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_28, 2, 4, 9223372036854775807);  view_28 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:34 in apply_rotary_pos_emb, code: q = q.to(cos.dtype)
        convert_element_type_49: "f32[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(slice_29, torch.float32);  slice_29 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:35 in apply_rotary_pos_emb, code: k = k.to(cos.dtype)
        convert_element_type_50: "f32[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(slice_32, torch.float32);  slice_32 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:37 in apply_rotary_pos_emb, code: q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
        mul_17: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_49, unsqueeze)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:25 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_36: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_49, 3, 0, 32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:26 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_37: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_49, 3, 32, 9223372036854775807);  convert_element_type_49 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:27 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_4: "f32[128, 901, 2, 32]" = torch.ops.aten.neg.default(slice_37);  slice_37 = None
        cat_5: "f32[128, 901, 2, 64]" = torch.ops.aten.cat.default([neg_4, slice_36], -1);  neg_4 = slice_36 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:37 in apply_rotary_pos_emb, code: q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
        mul_18: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(cat_5, unsqueeze_1);  cat_5 = None
        add_14: "f32[128, 901, 2, 64]" = torch.ops.aten.add.Tensor(mul_17, mul_18);  mul_17 = mul_18 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:38 in apply_rotary_pos_emb, code: k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
        mul_19: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_50, unsqueeze)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:25 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_38: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_50, 3, 0, 32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:26 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_39: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_50, 3, 32, 9223372036854775807);  convert_element_type_50 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:27 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_5: "f32[128, 901, 2, 32]" = torch.ops.aten.neg.default(slice_39);  slice_39 = None
        cat_6: "f32[128, 901, 2, 64]" = torch.ops.aten.cat.default([neg_5, slice_38], -1);  neg_5 = slice_38 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:38 in apply_rotary_pos_emb, code: k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
        mul_20: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(cat_6, unsqueeze_1);  cat_6 = None
        add_15: "f32[128, 901, 2, 64]" = torch.ops.aten.add.Tensor(mul_19, mul_20);  mul_19 = mul_20 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:40 in apply_rotary_pos_emb, code: return q_embed.to(orig_dtype), k_embed.to(orig_dtype)
        convert_element_type_51: "bf16[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(add_14, torch.bfloat16);  add_14 = None
        convert_element_type_52: "bf16[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(add_15, torch.bfloat16);  add_15 = None
        
         # File: C:\Users\arroy\anaconda3\envs\hrm_v1\lib\site-packages\flash_attn\flash_attn_interface.py:839 in forward, code: out_padded, softmax_lse, S_dmask, rng_state = _wrapped_flash_attn_forward(
        _flash_attn_forward_2 = torch.ops.flash_attn._flash_attn_forward.default(convert_element_type_51, convert_element_type_52, slice_35, 0.0, 0.125, False, -1, -1, 0.0, None, False);  convert_element_type_51 = convert_element_type_52 = slice_35 = None
        getitem_12: "bf16[128, 901, 2, 64]" = _flash_attn_forward_2[0];  _flash_attn_forward_2 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:135 in forward, code: attn_output = attn_output.view(batch_size, seq_len, self.output_size)  # type: ignore
        view_29: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(getitem_12, [128, 901, 128]);  getitem_12 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        convert_element_type_53: "bf16[128, 128]" = torch.ops.prims.convert_element_type.default(primals_28, torch.bfloat16)
        permute_9: "bf16[128, 128]" = torch.ops.aten.permute.default(convert_element_type_53, [1, 0]);  convert_element_type_53 = None
        view_30: "bf16[115328, 128]" = torch.ops.aten.reshape.default(view_29, [115328, 128]);  view_29 = None
        mm_9: "bf16[115328, 128]" = torch.ops.aten.mm.default(view_30, permute_9);  view_30 = None
        view_31: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(mm_9, [128, 901, 128]);  mm_9 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:80 in forward, code: hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        add_16: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(convert_element_type_45, view_31);  convert_element_type_45 = view_31 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:154 in rms_norm, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_56: "f32[128, 901, 128]" = torch.ops.prims.convert_element_type.default(add_16, torch.float32);  add_16 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:156 in rms_norm, code: variance = hidden_states.square().mean(-1, keepdim=True)
        pow_5: "f32[128, 901, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_56, 2)
        mean_4: "f32[128, 901, 1]" = torch.ops.aten.mean.dim(pow_5, [-1], True);  pow_5 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:157 in rms_norm, code: hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        add_17: "f32[128, 901, 1]" = torch.ops.aten.add.Tensor(mean_4, 1e-05);  mean_4 = None
        rsqrt_4: "f32[128, 901, 1]" = torch.ops.aten.rsqrt.default(add_17);  add_17 = None
        mul_21: "f32[128, 901, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_56, rsqrt_4);  convert_element_type_56 = rsqrt_4 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:158 in rms_norm, code: return hidden_states.to(input_dtype)
        convert_element_type_57: "bf16[128, 901, 128]" = torch.ops.prims.convert_element_type.default(mul_21, torch.bfloat16);  mul_21 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        convert_element_type_58: "bf16[1024, 128]" = torch.ops.prims.convert_element_type.default(primals_29, torch.bfloat16)
        permute_10: "bf16[128, 1024]" = torch.ops.aten.permute.default(convert_element_type_58, [1, 0]);  convert_element_type_58 = None
        view_32: "bf16[115328, 128]" = torch.ops.aten.reshape.default(convert_element_type_57, [115328, 128])
        mm_10: "bf16[115328, 1024]" = torch.ops.aten.mm.default(view_32, permute_10);  view_32 = None
        view_33: "bf16[128, 901, 1024]" = torch.ops.aten.reshape.default(mm_10, [128, 901, 1024]);  mm_10 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:148 in forward, code: gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        split_2 = torch.ops.aten.split.Tensor(view_33, 512, -1);  view_33 = None
        getitem_16: "bf16[128, 901, 512]" = split_2[0]
        getitem_17: "bf16[128, 901, 512]" = split_2[1];  split_2 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:149 in forward, code: return self.down_proj(F.silu(gate) * up)
        convert_element_type_61: "f32[128, 901, 512]" = torch.ops.prims.convert_element_type.default(getitem_16, torch.float32);  getitem_16 = None
        sigmoid_2: "f32[128, 901, 512]" = torch.ops.aten.sigmoid.default(convert_element_type_61)
        mul_22: "f32[128, 901, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_61, sigmoid_2);  convert_element_type_61 = sigmoid_2 = None
        convert_element_type_62: "bf16[128, 901, 512]" = torch.ops.prims.convert_element_type.default(mul_22, torch.bfloat16);  mul_22 = None
        mul_23: "bf16[128, 901, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_62, getitem_17);  convert_element_type_62 = getitem_17 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        convert_element_type_63: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(primals_30, torch.bfloat16)
        permute_11: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_63, [1, 0]);  convert_element_type_63 = None
        view_34: "bf16[115328, 512]" = torch.ops.aten.reshape.default(mul_23, [115328, 512]);  mul_23 = None
        mm_11: "bf16[115328, 128]" = torch.ops.aten.mm.default(view_34, permute_11);  view_34 = None
        view_35: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(mm_11, [128, 901, 128]);  mm_11 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:82 in forward, code: hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)
        add_18: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(convert_element_type_57, view_35);  convert_element_type_57 = view_35 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:154 in rms_norm, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_66: "f32[128, 901, 128]" = torch.ops.prims.convert_element_type.default(add_18, torch.float32);  add_18 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:156 in rms_norm, code: variance = hidden_states.square().mean(-1, keepdim=True)
        pow_6: "f32[128, 901, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_66, 2)
        mean_5: "f32[128, 901, 1]" = torch.ops.aten.mean.dim(pow_6, [-1], True);  pow_6 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:157 in rms_norm, code: hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        add_19: "f32[128, 901, 1]" = torch.ops.aten.add.Tensor(mean_5, 1e-05);  mean_5 = None
        rsqrt_5: "f32[128, 901, 1]" = torch.ops.aten.rsqrt.default(add_19);  add_19 = None
        mul_24: "f32[128, 901, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_66, rsqrt_5);  convert_element_type_66 = rsqrt_5 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:158 in rms_norm, code: return hidden_states.to(input_dtype)
        convert_element_type_67: "bf16[128, 901, 128]" = torch.ops.prims.convert_element_type.default(mul_24, torch.bfloat16);  mul_24 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        convert_element_type_68: "bf16[384, 128]" = torch.ops.prims.convert_element_type.default(primals_31, torch.bfloat16)
        permute_12: "bf16[128, 384]" = torch.ops.aten.permute.default(convert_element_type_68, [1, 0]);  convert_element_type_68 = None
        view_36: "bf16[115328, 128]" = torch.ops.aten.reshape.default(convert_element_type_67, [115328, 128])
        mm_12: "bf16[115328, 384]" = torch.ops.aten.mm.default(view_36, permute_12);  view_36 = None
        view_37: "bf16[128, 901, 384]" = torch.ops.aten.reshape.default(mm_12, [128, 901, 384]);  mm_12 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:119 in forward, code: qkv = qkv.view(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        view_38: "bf16[128, 901, 6, 64]" = torch.ops.aten.reshape.default(view_37, [128, 901, 6, 64]);  view_37 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:120 in forward, code: query = qkv[:, :, :self.num_heads]
        slice_42: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_38, 2, 0, 2)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:121 in forward, code: key = qkv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
        slice_45: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_38, 2, 2, 4)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:122 in forward, code: value = qkv[:, :, self.num_heads + self.num_key_value_heads:]
        slice_48: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_38, 2, 4, 9223372036854775807);  view_38 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:34 in apply_rotary_pos_emb, code: q = q.to(cos.dtype)
        convert_element_type_71: "f32[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(slice_42, torch.float32);  slice_42 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:35 in apply_rotary_pos_emb, code: k = k.to(cos.dtype)
        convert_element_type_72: "f32[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(slice_45, torch.float32);  slice_45 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:37 in apply_rotary_pos_emb, code: q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
        mul_25: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_71, unsqueeze)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:25 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_49: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_71, 3, 0, 32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:26 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_50: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_71, 3, 32, 9223372036854775807);  convert_element_type_71 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:27 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_6: "f32[128, 901, 2, 32]" = torch.ops.aten.neg.default(slice_50);  slice_50 = None
        cat_7: "f32[128, 901, 2, 64]" = torch.ops.aten.cat.default([neg_6, slice_49], -1);  neg_6 = slice_49 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:37 in apply_rotary_pos_emb, code: q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
        mul_26: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(cat_7, unsqueeze_1);  cat_7 = None
        add_20: "f32[128, 901, 2, 64]" = torch.ops.aten.add.Tensor(mul_25, mul_26);  mul_25 = mul_26 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:38 in apply_rotary_pos_emb, code: k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
        mul_27: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_72, unsqueeze)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:25 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_51: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_72, 3, 0, 32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:26 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_52: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_72, 3, 32, 9223372036854775807);  convert_element_type_72 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:27 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_7: "f32[128, 901, 2, 32]" = torch.ops.aten.neg.default(slice_52);  slice_52 = None
        cat_8: "f32[128, 901, 2, 64]" = torch.ops.aten.cat.default([neg_7, slice_51], -1);  neg_7 = slice_51 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:38 in apply_rotary_pos_emb, code: k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
        mul_28: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(cat_8, unsqueeze_1);  cat_8 = None
        add_21: "f32[128, 901, 2, 64]" = torch.ops.aten.add.Tensor(mul_27, mul_28);  mul_27 = mul_28 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:40 in apply_rotary_pos_emb, code: return q_embed.to(orig_dtype), k_embed.to(orig_dtype)
        convert_element_type_73: "bf16[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(add_20, torch.bfloat16);  add_20 = None
        convert_element_type_74: "bf16[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(add_21, torch.bfloat16);  add_21 = None
        
         # File: C:\Users\arroy\anaconda3\envs\hrm_v1\lib\site-packages\flash_attn\flash_attn_interface.py:839 in forward, code: out_padded, softmax_lse, S_dmask, rng_state = _wrapped_flash_attn_forward(
        _flash_attn_forward_3 = torch.ops.flash_attn._flash_attn_forward.default(convert_element_type_73, convert_element_type_74, slice_48, 0.0, 0.125, False, -1, -1, 0.0, None, False);  convert_element_type_73 = convert_element_type_74 = slice_48 = None
        getitem_18: "bf16[128, 901, 2, 64]" = _flash_attn_forward_3[0];  _flash_attn_forward_3 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:135 in forward, code: attn_output = attn_output.view(batch_size, seq_len, self.output_size)  # type: ignore
        view_39: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(getitem_18, [128, 901, 128]);  getitem_18 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        convert_element_type_75: "bf16[128, 128]" = torch.ops.prims.convert_element_type.default(primals_32, torch.bfloat16)
        permute_13: "bf16[128, 128]" = torch.ops.aten.permute.default(convert_element_type_75, [1, 0]);  convert_element_type_75 = None
        view_40: "bf16[115328, 128]" = torch.ops.aten.reshape.default(view_39, [115328, 128]);  view_39 = None
        mm_13: "bf16[115328, 128]" = torch.ops.aten.mm.default(view_40, permute_13);  view_40 = None
        view_41: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(mm_13, [128, 901, 128]);  mm_13 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:80 in forward, code: hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        add_22: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(convert_element_type_67, view_41);  convert_element_type_67 = view_41 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:154 in rms_norm, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_78: "f32[128, 901, 128]" = torch.ops.prims.convert_element_type.default(add_22, torch.float32);  add_22 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:156 in rms_norm, code: variance = hidden_states.square().mean(-1, keepdim=True)
        pow_7: "f32[128, 901, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_78, 2)
        mean_6: "f32[128, 901, 1]" = torch.ops.aten.mean.dim(pow_7, [-1], True);  pow_7 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:157 in rms_norm, code: hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        add_23: "f32[128, 901, 1]" = torch.ops.aten.add.Tensor(mean_6, 1e-05);  mean_6 = None
        rsqrt_6: "f32[128, 901, 1]" = torch.ops.aten.rsqrt.default(add_23);  add_23 = None
        mul_29: "f32[128, 901, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_78, rsqrt_6);  convert_element_type_78 = rsqrt_6 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:158 in rms_norm, code: return hidden_states.to(input_dtype)
        convert_element_type_79: "bf16[128, 901, 128]" = torch.ops.prims.convert_element_type.default(mul_29, torch.bfloat16);  mul_29 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        convert_element_type_80: "bf16[1024, 128]" = torch.ops.prims.convert_element_type.default(primals_33, torch.bfloat16)
        permute_14: "bf16[128, 1024]" = torch.ops.aten.permute.default(convert_element_type_80, [1, 0]);  convert_element_type_80 = None
        view_42: "bf16[115328, 128]" = torch.ops.aten.reshape.default(convert_element_type_79, [115328, 128])
        mm_14: "bf16[115328, 1024]" = torch.ops.aten.mm.default(view_42, permute_14);  view_42 = None
        view_43: "bf16[128, 901, 1024]" = torch.ops.aten.reshape.default(mm_14, [128, 901, 1024]);  mm_14 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:148 in forward, code: gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        split_3 = torch.ops.aten.split.Tensor(view_43, 512, -1);  view_43 = None
        getitem_22: "bf16[128, 901, 512]" = split_3[0]
        getitem_23: "bf16[128, 901, 512]" = split_3[1];  split_3 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:149 in forward, code: return self.down_proj(F.silu(gate) * up)
        convert_element_type_83: "f32[128, 901, 512]" = torch.ops.prims.convert_element_type.default(getitem_22, torch.float32);  getitem_22 = None
        sigmoid_3: "f32[128, 901, 512]" = torch.ops.aten.sigmoid.default(convert_element_type_83)
        mul_30: "f32[128, 901, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_83, sigmoid_3);  convert_element_type_83 = sigmoid_3 = None
        convert_element_type_84: "bf16[128, 901, 512]" = torch.ops.prims.convert_element_type.default(mul_30, torch.bfloat16);  mul_30 = None
        mul_31: "bf16[128, 901, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_84, getitem_23);  convert_element_type_84 = getitem_23 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        convert_element_type_85: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(primals_34, torch.bfloat16)
        permute_15: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_85, [1, 0]);  convert_element_type_85 = None
        view_44: "bf16[115328, 512]" = torch.ops.aten.reshape.default(mul_31, [115328, 512]);  mul_31 = None
        mm_15: "bf16[115328, 128]" = torch.ops.aten.mm.default(view_44, permute_15);  view_44 = None
        view_45: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(mm_15, [128, 901, 128]);  mm_15 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:82 in forward, code: hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)
        add_24: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(convert_element_type_79, view_45);  convert_element_type_79 = view_45 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:154 in rms_norm, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_88: "f32[128, 901, 128]" = torch.ops.prims.convert_element_type.default(add_24, torch.float32);  add_24 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:156 in rms_norm, code: variance = hidden_states.square().mean(-1, keepdim=True)
        pow_8: "f32[128, 901, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_88, 2)
        mean_7: "f32[128, 901, 1]" = torch.ops.aten.mean.dim(pow_8, [-1], True);  pow_8 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:157 in rms_norm, code: hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        add_25: "f32[128, 901, 1]" = torch.ops.aten.add.Tensor(mean_7, 1e-05);  mean_7 = None
        rsqrt_7: "f32[128, 901, 1]" = torch.ops.aten.rsqrt.default(add_25);  add_25 = None
        mul_32: "f32[128, 901, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_88, rsqrt_7);  convert_element_type_88 = rsqrt_7 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:158 in rms_norm, code: return hidden_states.to(input_dtype)
        convert_element_type_89: "bf16[128, 901, 128]" = torch.ops.prims.convert_element_type.default(mul_32, torch.bfloat16);  mul_32 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:94 in forward, code: hidden_states = hidden_states + input_injection
        add_27: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(convert_element_type_89, add);  convert_element_type_89 = add = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_46: "bf16[115328, 128]" = torch.ops.aten.reshape.default(add_27, [115328, 128])
        mm_16: "bf16[115328, 384]" = torch.ops.aten.mm.default(view_46, permute);  view_46 = None
        view_47: "bf16[128, 901, 384]" = torch.ops.aten.reshape.default(mm_16, [128, 901, 384]);  mm_16 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:119 in forward, code: qkv = qkv.view(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        view_48: "bf16[128, 901, 6, 64]" = torch.ops.aten.reshape.default(view_47, [128, 901, 6, 64]);  view_47 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:120 in forward, code: query = qkv[:, :, :self.num_heads]
        slice_55: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_48, 2, 0, 2)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:121 in forward, code: key = qkv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
        slice_58: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_48, 2, 2, 4)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:122 in forward, code: value = qkv[:, :, self.num_heads + self.num_key_value_heads:]
        slice_61: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_48, 2, 4, 9223372036854775807);  view_48 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:34 in apply_rotary_pos_emb, code: q = q.to(cos.dtype)
        convert_element_type_93: "f32[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(slice_55, torch.float32);  slice_55 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:35 in apply_rotary_pos_emb, code: k = k.to(cos.dtype)
        convert_element_type_94: "f32[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(slice_58, torch.float32);  slice_58 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:37 in apply_rotary_pos_emb, code: q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
        mul_33: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_93, unsqueeze)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:25 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_62: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_93, 3, 0, 32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:26 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_63: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_93, 3, 32, 9223372036854775807);  convert_element_type_93 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:27 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_8: "f32[128, 901, 2, 32]" = torch.ops.aten.neg.default(slice_63);  slice_63 = None
        cat_9: "f32[128, 901, 2, 64]" = torch.ops.aten.cat.default([neg_8, slice_62], -1);  neg_8 = slice_62 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:37 in apply_rotary_pos_emb, code: q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
        mul_34: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(cat_9, unsqueeze_1);  cat_9 = None
        add_28: "f32[128, 901, 2, 64]" = torch.ops.aten.add.Tensor(mul_33, mul_34);  mul_33 = mul_34 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:38 in apply_rotary_pos_emb, code: k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
        mul_35: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_94, unsqueeze)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:25 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_64: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_94, 3, 0, 32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:26 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_65: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_94, 3, 32, 9223372036854775807);  convert_element_type_94 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:27 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_9: "f32[128, 901, 2, 32]" = torch.ops.aten.neg.default(slice_65);  slice_65 = None
        cat_10: "f32[128, 901, 2, 64]" = torch.ops.aten.cat.default([neg_9, slice_64], -1);  neg_9 = slice_64 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:38 in apply_rotary_pos_emb, code: k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
        mul_36: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(cat_10, unsqueeze_1);  cat_10 = None
        add_29: "f32[128, 901, 2, 64]" = torch.ops.aten.add.Tensor(mul_35, mul_36);  mul_35 = mul_36 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:40 in apply_rotary_pos_emb, code: return q_embed.to(orig_dtype), k_embed.to(orig_dtype)
        convert_element_type_95: "bf16[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(add_28, torch.bfloat16);  add_28 = None
        convert_element_type_96: "bf16[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(add_29, torch.bfloat16);  add_29 = None
        
         # File: C:\Users\arroy\anaconda3\envs\hrm_v1\lib\site-packages\flash_attn\flash_attn_interface.py:839 in forward, code: out_padded, softmax_lse, S_dmask, rng_state = _wrapped_flash_attn_forward(
        _flash_attn_forward_4 = torch.ops.flash_attn._flash_attn_forward.default(convert_element_type_95, convert_element_type_96, slice_61, 0.0, 0.125, False, -1, -1, 0.0, None, False);  convert_element_type_95 = convert_element_type_96 = slice_61 = None
        getitem_24: "bf16[128, 901, 2, 64]" = _flash_attn_forward_4[0];  _flash_attn_forward_4 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:135 in forward, code: attn_output = attn_output.view(batch_size, seq_len, self.output_size)  # type: ignore
        view_49: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(getitem_24, [128, 901, 128]);  getitem_24 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_50: "bf16[115328, 128]" = torch.ops.aten.reshape.default(view_49, [115328, 128]);  view_49 = None
        mm_17: "bf16[115328, 128]" = torch.ops.aten.mm.default(view_50, permute_1);  view_50 = None
        view_51: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(mm_17, [128, 901, 128]);  mm_17 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:80 in forward, code: hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        add_30: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(add_27, view_51);  add_27 = view_51 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:154 in rms_norm, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_100: "f32[128, 901, 128]" = torch.ops.prims.convert_element_type.default(add_30, torch.float32);  add_30 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:156 in rms_norm, code: variance = hidden_states.square().mean(-1, keepdim=True)
        pow_9: "f32[128, 901, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_100, 2)
        mean_8: "f32[128, 901, 1]" = torch.ops.aten.mean.dim(pow_9, [-1], True);  pow_9 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:157 in rms_norm, code: hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        add_31: "f32[128, 901, 1]" = torch.ops.aten.add.Tensor(mean_8, 1e-05);  mean_8 = None
        rsqrt_8: "f32[128, 901, 1]" = torch.ops.aten.rsqrt.default(add_31);  add_31 = None
        mul_37: "f32[128, 901, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_100, rsqrt_8);  convert_element_type_100 = rsqrt_8 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:158 in rms_norm, code: return hidden_states.to(input_dtype)
        convert_element_type_101: "bf16[128, 901, 128]" = torch.ops.prims.convert_element_type.default(mul_37, torch.bfloat16);  mul_37 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_52: "bf16[115328, 128]" = torch.ops.aten.reshape.default(convert_element_type_101, [115328, 128])
        mm_18: "bf16[115328, 1024]" = torch.ops.aten.mm.default(view_52, permute_2);  view_52 = None
        view_53: "bf16[128, 901, 1024]" = torch.ops.aten.reshape.default(mm_18, [128, 901, 1024]);  mm_18 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:148 in forward, code: gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        split_4 = torch.ops.aten.split.Tensor(view_53, 512, -1);  view_53 = None
        getitem_28: "bf16[128, 901, 512]" = split_4[0]
        getitem_29: "bf16[128, 901, 512]" = split_4[1];  split_4 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:149 in forward, code: return self.down_proj(F.silu(gate) * up)
        convert_element_type_105: "f32[128, 901, 512]" = torch.ops.prims.convert_element_type.default(getitem_28, torch.float32);  getitem_28 = None
        sigmoid_4: "f32[128, 901, 512]" = torch.ops.aten.sigmoid.default(convert_element_type_105)
        mul_38: "f32[128, 901, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_105, sigmoid_4);  convert_element_type_105 = sigmoid_4 = None
        convert_element_type_106: "bf16[128, 901, 512]" = torch.ops.prims.convert_element_type.default(mul_38, torch.bfloat16);  mul_38 = None
        mul_39: "bf16[128, 901, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_106, getitem_29);  convert_element_type_106 = getitem_29 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_54: "bf16[115328, 512]" = torch.ops.aten.reshape.default(mul_39, [115328, 512]);  mul_39 = None
        mm_19: "bf16[115328, 128]" = torch.ops.aten.mm.default(view_54, permute_3);  view_54 = None
        view_55: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(mm_19, [128, 901, 128]);  mm_19 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:82 in forward, code: hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)
        add_32: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(convert_element_type_101, view_55);  convert_element_type_101 = view_55 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:154 in rms_norm, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_110: "f32[128, 901, 128]" = torch.ops.prims.convert_element_type.default(add_32, torch.float32);  add_32 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:156 in rms_norm, code: variance = hidden_states.square().mean(-1, keepdim=True)
        pow_10: "f32[128, 901, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_110, 2)
        mean_9: "f32[128, 901, 1]" = torch.ops.aten.mean.dim(pow_10, [-1], True);  pow_10 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:157 in rms_norm, code: hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        add_33: "f32[128, 901, 1]" = torch.ops.aten.add.Tensor(mean_9, 1e-05);  mean_9 = None
        rsqrt_9: "f32[128, 901, 1]" = torch.ops.aten.rsqrt.default(add_33);  add_33 = None
        mul_40: "f32[128, 901, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_110, rsqrt_9);  convert_element_type_110 = rsqrt_9 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:158 in rms_norm, code: return hidden_states.to(input_dtype)
        convert_element_type_111: "bf16[128, 901, 128]" = torch.ops.prims.convert_element_type.default(mul_40, torch.bfloat16);  mul_40 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_56: "bf16[115328, 128]" = torch.ops.aten.reshape.default(convert_element_type_111, [115328, 128])
        mm_20: "bf16[115328, 384]" = torch.ops.aten.mm.default(view_56, permute_4);  view_56 = None
        view_57: "bf16[128, 901, 384]" = torch.ops.aten.reshape.default(mm_20, [128, 901, 384]);  mm_20 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:119 in forward, code: qkv = qkv.view(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        view_58: "bf16[128, 901, 6, 64]" = torch.ops.aten.reshape.default(view_57, [128, 901, 6, 64]);  view_57 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:120 in forward, code: query = qkv[:, :, :self.num_heads]
        slice_68: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_58, 2, 0, 2)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:121 in forward, code: key = qkv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
        slice_71: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_58, 2, 2, 4)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:122 in forward, code: value = qkv[:, :, self.num_heads + self.num_key_value_heads:]
        slice_74: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_58, 2, 4, 9223372036854775807);  view_58 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:34 in apply_rotary_pos_emb, code: q = q.to(cos.dtype)
        convert_element_type_115: "f32[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(slice_68, torch.float32);  slice_68 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:35 in apply_rotary_pos_emb, code: k = k.to(cos.dtype)
        convert_element_type_116: "f32[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(slice_71, torch.float32);  slice_71 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:37 in apply_rotary_pos_emb, code: q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
        mul_41: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_115, unsqueeze)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:25 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_75: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_115, 3, 0, 32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:26 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_76: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_115, 3, 32, 9223372036854775807);  convert_element_type_115 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:27 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_10: "f32[128, 901, 2, 32]" = torch.ops.aten.neg.default(slice_76);  slice_76 = None
        cat_11: "f32[128, 901, 2, 64]" = torch.ops.aten.cat.default([neg_10, slice_75], -1);  neg_10 = slice_75 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:37 in apply_rotary_pos_emb, code: q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
        mul_42: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(cat_11, unsqueeze_1);  cat_11 = None
        add_34: "f32[128, 901, 2, 64]" = torch.ops.aten.add.Tensor(mul_41, mul_42);  mul_41 = mul_42 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:38 in apply_rotary_pos_emb, code: k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
        mul_43: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_116, unsqueeze)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:25 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_77: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_116, 3, 0, 32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:26 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_78: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_116, 3, 32, 9223372036854775807);  convert_element_type_116 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:27 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_11: "f32[128, 901, 2, 32]" = torch.ops.aten.neg.default(slice_78);  slice_78 = None
        cat_12: "f32[128, 901, 2, 64]" = torch.ops.aten.cat.default([neg_11, slice_77], -1);  neg_11 = slice_77 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:38 in apply_rotary_pos_emb, code: k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
        mul_44: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(cat_12, unsqueeze_1);  cat_12 = None
        add_35: "f32[128, 901, 2, 64]" = torch.ops.aten.add.Tensor(mul_43, mul_44);  mul_43 = mul_44 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:40 in apply_rotary_pos_emb, code: return q_embed.to(orig_dtype), k_embed.to(orig_dtype)
        convert_element_type_117: "bf16[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(add_34, torch.bfloat16);  add_34 = None
        convert_element_type_118: "bf16[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(add_35, torch.bfloat16);  add_35 = None
        
         # File: C:\Users\arroy\anaconda3\envs\hrm_v1\lib\site-packages\flash_attn\flash_attn_interface.py:839 in forward, code: out_padded, softmax_lse, S_dmask, rng_state = _wrapped_flash_attn_forward(
        _flash_attn_forward_5 = torch.ops.flash_attn._flash_attn_forward.default(convert_element_type_117, convert_element_type_118, slice_74, 0.0, 0.125, False, -1, -1, 0.0, None, False);  convert_element_type_117 = convert_element_type_118 = slice_74 = None
        getitem_30: "bf16[128, 901, 2, 64]" = _flash_attn_forward_5[0];  _flash_attn_forward_5 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:135 in forward, code: attn_output = attn_output.view(batch_size, seq_len, self.output_size)  # type: ignore
        view_59: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(getitem_30, [128, 901, 128]);  getitem_30 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_60: "bf16[115328, 128]" = torch.ops.aten.reshape.default(view_59, [115328, 128]);  view_59 = None
        mm_21: "bf16[115328, 128]" = torch.ops.aten.mm.default(view_60, permute_5);  view_60 = None
        view_61: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(mm_21, [128, 901, 128]);  mm_21 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:80 in forward, code: hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        add_36: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(convert_element_type_111, view_61);  convert_element_type_111 = view_61 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:154 in rms_norm, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_122: "f32[128, 901, 128]" = torch.ops.prims.convert_element_type.default(add_36, torch.float32);  add_36 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:156 in rms_norm, code: variance = hidden_states.square().mean(-1, keepdim=True)
        pow_11: "f32[128, 901, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_122, 2)
        mean_10: "f32[128, 901, 1]" = torch.ops.aten.mean.dim(pow_11, [-1], True);  pow_11 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:157 in rms_norm, code: hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        add_37: "f32[128, 901, 1]" = torch.ops.aten.add.Tensor(mean_10, 1e-05);  mean_10 = None
        rsqrt_10: "f32[128, 901, 1]" = torch.ops.aten.rsqrt.default(add_37);  add_37 = None
        mul_45: "f32[128, 901, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_122, rsqrt_10);  convert_element_type_122 = rsqrt_10 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:158 in rms_norm, code: return hidden_states.to(input_dtype)
        convert_element_type_123: "bf16[128, 901, 128]" = torch.ops.prims.convert_element_type.default(mul_45, torch.bfloat16);  mul_45 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_62: "bf16[115328, 128]" = torch.ops.aten.reshape.default(convert_element_type_123, [115328, 128])
        mm_22: "bf16[115328, 1024]" = torch.ops.aten.mm.default(view_62, permute_6);  view_62 = None
        view_63: "bf16[128, 901, 1024]" = torch.ops.aten.reshape.default(mm_22, [128, 901, 1024]);  mm_22 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:148 in forward, code: gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        split_5 = torch.ops.aten.split.Tensor(view_63, 512, -1);  view_63 = None
        getitem_34: "bf16[128, 901, 512]" = split_5[0]
        getitem_35: "bf16[128, 901, 512]" = split_5[1];  split_5 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:149 in forward, code: return self.down_proj(F.silu(gate) * up)
        convert_element_type_127: "f32[128, 901, 512]" = torch.ops.prims.convert_element_type.default(getitem_34, torch.float32);  getitem_34 = None
        sigmoid_5: "f32[128, 901, 512]" = torch.ops.aten.sigmoid.default(convert_element_type_127)
        mul_46: "f32[128, 901, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_127, sigmoid_5);  convert_element_type_127 = sigmoid_5 = None
        convert_element_type_128: "bf16[128, 901, 512]" = torch.ops.prims.convert_element_type.default(mul_46, torch.bfloat16);  mul_46 = None
        mul_47: "bf16[128, 901, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_128, getitem_35);  convert_element_type_128 = getitem_35 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_64: "bf16[115328, 512]" = torch.ops.aten.reshape.default(mul_47, [115328, 512]);  mul_47 = None
        mm_23: "bf16[115328, 128]" = torch.ops.aten.mm.default(view_64, permute_7);  view_64 = None
        view_65: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(mm_23, [128, 901, 128]);  mm_23 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:82 in forward, code: hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)
        add_38: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(convert_element_type_123, view_65);  convert_element_type_123 = view_65 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:154 in rms_norm, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_132: "f32[128, 901, 128]" = torch.ops.prims.convert_element_type.default(add_38, torch.float32);  add_38 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:156 in rms_norm, code: variance = hidden_states.square().mean(-1, keepdim=True)
        pow_12: "f32[128, 901, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_132, 2)
        mean_11: "f32[128, 901, 1]" = torch.ops.aten.mean.dim(pow_12, [-1], True);  pow_12 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:157 in rms_norm, code: hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        add_39: "f32[128, 901, 1]" = torch.ops.aten.add.Tensor(mean_11, 1e-05);  mean_11 = None
        rsqrt_11: "f32[128, 901, 1]" = torch.ops.aten.rsqrt.default(add_39);  add_39 = None
        mul_48: "f32[128, 901, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_132, rsqrt_11);  convert_element_type_132 = rsqrt_11 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:158 in rms_norm, code: return hidden_states.to(input_dtype)
        convert_element_type_133: "bf16[128, 901, 128]" = torch.ops.prims.convert_element_type.default(mul_48, torch.bfloat16);  mul_48 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_66: "bf16[115328, 128]" = torch.ops.aten.reshape.default(convert_element_type_133, [115328, 128])
        mm_24: "bf16[115328, 384]" = torch.ops.aten.mm.default(view_66, permute_8);  view_66 = None
        view_67: "bf16[128, 901, 384]" = torch.ops.aten.reshape.default(mm_24, [128, 901, 384]);  mm_24 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:119 in forward, code: qkv = qkv.view(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        view_68: "bf16[128, 901, 6, 64]" = torch.ops.aten.reshape.default(view_67, [128, 901, 6, 64]);  view_67 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:120 in forward, code: query = qkv[:, :, :self.num_heads]
        slice_81: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_68, 2, 0, 2)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:121 in forward, code: key = qkv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
        slice_84: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_68, 2, 2, 4)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:122 in forward, code: value = qkv[:, :, self.num_heads + self.num_key_value_heads:]
        slice_87: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_68, 2, 4, 9223372036854775807);  view_68 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:34 in apply_rotary_pos_emb, code: q = q.to(cos.dtype)
        convert_element_type_137: "f32[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(slice_81, torch.float32);  slice_81 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:35 in apply_rotary_pos_emb, code: k = k.to(cos.dtype)
        convert_element_type_138: "f32[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(slice_84, torch.float32);  slice_84 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:37 in apply_rotary_pos_emb, code: q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
        mul_49: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_137, unsqueeze)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:25 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_88: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_137, 3, 0, 32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:26 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_89: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_137, 3, 32, 9223372036854775807);  convert_element_type_137 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:27 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_12: "f32[128, 901, 2, 32]" = torch.ops.aten.neg.default(slice_89);  slice_89 = None
        cat_13: "f32[128, 901, 2, 64]" = torch.ops.aten.cat.default([neg_12, slice_88], -1);  neg_12 = slice_88 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:37 in apply_rotary_pos_emb, code: q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
        mul_50: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(cat_13, unsqueeze_1);  cat_13 = None
        add_40: "f32[128, 901, 2, 64]" = torch.ops.aten.add.Tensor(mul_49, mul_50);  mul_49 = mul_50 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:38 in apply_rotary_pos_emb, code: k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
        mul_51: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_138, unsqueeze)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:25 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_90: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_138, 3, 0, 32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:26 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_91: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_138, 3, 32, 9223372036854775807);  convert_element_type_138 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:27 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_13: "f32[128, 901, 2, 32]" = torch.ops.aten.neg.default(slice_91);  slice_91 = None
        cat_14: "f32[128, 901, 2, 64]" = torch.ops.aten.cat.default([neg_13, slice_90], -1);  neg_13 = slice_90 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:38 in apply_rotary_pos_emb, code: k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
        mul_52: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(cat_14, unsqueeze_1);  cat_14 = None
        add_41: "f32[128, 901, 2, 64]" = torch.ops.aten.add.Tensor(mul_51, mul_52);  mul_51 = mul_52 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:40 in apply_rotary_pos_emb, code: return q_embed.to(orig_dtype), k_embed.to(orig_dtype)
        convert_element_type_139: "bf16[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(add_40, torch.bfloat16);  add_40 = None
        convert_element_type_140: "bf16[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(add_41, torch.bfloat16);  add_41 = None
        
         # File: C:\Users\arroy\anaconda3\envs\hrm_v1\lib\site-packages\flash_attn\flash_attn_interface.py:839 in forward, code: out_padded, softmax_lse, S_dmask, rng_state = _wrapped_flash_attn_forward(
        _flash_attn_forward_6 = torch.ops.flash_attn._flash_attn_forward.default(convert_element_type_139, convert_element_type_140, slice_87, 0.0, 0.125, False, -1, -1, 0.0, None, False);  convert_element_type_139 = convert_element_type_140 = slice_87 = None
        getitem_36: "bf16[128, 901, 2, 64]" = _flash_attn_forward_6[0];  _flash_attn_forward_6 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:135 in forward, code: attn_output = attn_output.view(batch_size, seq_len, self.output_size)  # type: ignore
        view_69: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(getitem_36, [128, 901, 128]);  getitem_36 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_70: "bf16[115328, 128]" = torch.ops.aten.reshape.default(view_69, [115328, 128]);  view_69 = None
        mm_25: "bf16[115328, 128]" = torch.ops.aten.mm.default(view_70, permute_9);  view_70 = None
        view_71: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(mm_25, [128, 901, 128]);  mm_25 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:80 in forward, code: hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        add_42: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(convert_element_type_133, view_71);  convert_element_type_133 = view_71 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:154 in rms_norm, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_144: "f32[128, 901, 128]" = torch.ops.prims.convert_element_type.default(add_42, torch.float32);  add_42 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:156 in rms_norm, code: variance = hidden_states.square().mean(-1, keepdim=True)
        pow_13: "f32[128, 901, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_144, 2)
        mean_12: "f32[128, 901, 1]" = torch.ops.aten.mean.dim(pow_13, [-1], True);  pow_13 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:157 in rms_norm, code: hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        add_43: "f32[128, 901, 1]" = torch.ops.aten.add.Tensor(mean_12, 1e-05);  mean_12 = None
        rsqrt_12: "f32[128, 901, 1]" = torch.ops.aten.rsqrt.default(add_43);  add_43 = None
        mul_53: "f32[128, 901, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_144, rsqrt_12);  convert_element_type_144 = rsqrt_12 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:158 in rms_norm, code: return hidden_states.to(input_dtype)
        convert_element_type_145: "bf16[128, 901, 128]" = torch.ops.prims.convert_element_type.default(mul_53, torch.bfloat16);  mul_53 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_72: "bf16[115328, 128]" = torch.ops.aten.reshape.default(convert_element_type_145, [115328, 128])
        mm_26: "bf16[115328, 1024]" = torch.ops.aten.mm.default(view_72, permute_10);  view_72 = None
        view_73: "bf16[128, 901, 1024]" = torch.ops.aten.reshape.default(mm_26, [128, 901, 1024]);  mm_26 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:148 in forward, code: gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        split_6 = torch.ops.aten.split.Tensor(view_73, 512, -1);  view_73 = None
        getitem_40: "bf16[128, 901, 512]" = split_6[0]
        getitem_41: "bf16[128, 901, 512]" = split_6[1];  split_6 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:149 in forward, code: return self.down_proj(F.silu(gate) * up)
        convert_element_type_149: "f32[128, 901, 512]" = torch.ops.prims.convert_element_type.default(getitem_40, torch.float32);  getitem_40 = None
        sigmoid_6: "f32[128, 901, 512]" = torch.ops.aten.sigmoid.default(convert_element_type_149)
        mul_54: "f32[128, 901, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_149, sigmoid_6);  convert_element_type_149 = sigmoid_6 = None
        convert_element_type_150: "bf16[128, 901, 512]" = torch.ops.prims.convert_element_type.default(mul_54, torch.bfloat16);  mul_54 = None
        mul_55: "bf16[128, 901, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_150, getitem_41);  convert_element_type_150 = getitem_41 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_74: "bf16[115328, 512]" = torch.ops.aten.reshape.default(mul_55, [115328, 512]);  mul_55 = None
        mm_27: "bf16[115328, 128]" = torch.ops.aten.mm.default(view_74, permute_11);  view_74 = None
        view_75: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(mm_27, [128, 901, 128]);  mm_27 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:82 in forward, code: hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)
        add_44: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(convert_element_type_145, view_75);  convert_element_type_145 = view_75 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:154 in rms_norm, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_154: "f32[128, 901, 128]" = torch.ops.prims.convert_element_type.default(add_44, torch.float32);  add_44 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:156 in rms_norm, code: variance = hidden_states.square().mean(-1, keepdim=True)
        pow_14: "f32[128, 901, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_154, 2)
        mean_13: "f32[128, 901, 1]" = torch.ops.aten.mean.dim(pow_14, [-1], True);  pow_14 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:157 in rms_norm, code: hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        add_45: "f32[128, 901, 1]" = torch.ops.aten.add.Tensor(mean_13, 1e-05);  mean_13 = None
        rsqrt_13: "f32[128, 901, 1]" = torch.ops.aten.rsqrt.default(add_45);  add_45 = None
        mul_56: "f32[128, 901, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_154, rsqrt_13);  convert_element_type_154 = rsqrt_13 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:158 in rms_norm, code: return hidden_states.to(input_dtype)
        convert_element_type_155: "bf16[128, 901, 128]" = torch.ops.prims.convert_element_type.default(mul_56, torch.bfloat16);  mul_56 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_76: "bf16[115328, 128]" = torch.ops.aten.reshape.default(convert_element_type_155, [115328, 128])
        mm_28: "bf16[115328, 384]" = torch.ops.aten.mm.default(view_76, permute_12);  view_76 = None
        view_77: "bf16[128, 901, 384]" = torch.ops.aten.reshape.default(mm_28, [128, 901, 384]);  mm_28 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:119 in forward, code: qkv = qkv.view(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        view_78: "bf16[128, 901, 6, 64]" = torch.ops.aten.reshape.default(view_77, [128, 901, 6, 64]);  view_77 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:120 in forward, code: query = qkv[:, :, :self.num_heads]
        slice_94: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_78, 2, 0, 2)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:121 in forward, code: key = qkv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
        slice_97: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_78, 2, 2, 4)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:122 in forward, code: value = qkv[:, :, self.num_heads + self.num_key_value_heads:]
        slice_100: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_78, 2, 4, 9223372036854775807);  view_78 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:34 in apply_rotary_pos_emb, code: q = q.to(cos.dtype)
        convert_element_type_159: "f32[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(slice_94, torch.float32);  slice_94 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:35 in apply_rotary_pos_emb, code: k = k.to(cos.dtype)
        convert_element_type_160: "f32[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(slice_97, torch.float32);  slice_97 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:37 in apply_rotary_pos_emb, code: q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
        mul_57: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_159, unsqueeze)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:25 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_101: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_159, 3, 0, 32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:26 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_102: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_159, 3, 32, 9223372036854775807);  convert_element_type_159 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:27 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_14: "f32[128, 901, 2, 32]" = torch.ops.aten.neg.default(slice_102);  slice_102 = None
        cat_15: "f32[128, 901, 2, 64]" = torch.ops.aten.cat.default([neg_14, slice_101], -1);  neg_14 = slice_101 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:37 in apply_rotary_pos_emb, code: q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
        mul_58: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(cat_15, unsqueeze_1);  cat_15 = None
        add_46: "f32[128, 901, 2, 64]" = torch.ops.aten.add.Tensor(mul_57, mul_58);  mul_57 = mul_58 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:38 in apply_rotary_pos_emb, code: k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
        mul_59: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_160, unsqueeze)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:25 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_103: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_160, 3, 0, 32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:26 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_104: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_160, 3, 32, 9223372036854775807);  convert_element_type_160 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:27 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_15: "f32[128, 901, 2, 32]" = torch.ops.aten.neg.default(slice_104);  slice_104 = None
        cat_16: "f32[128, 901, 2, 64]" = torch.ops.aten.cat.default([neg_15, slice_103], -1);  neg_15 = slice_103 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:38 in apply_rotary_pos_emb, code: k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
        mul_60: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(cat_16, unsqueeze_1);  cat_16 = None
        add_47: "f32[128, 901, 2, 64]" = torch.ops.aten.add.Tensor(mul_59, mul_60);  mul_59 = mul_60 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:40 in apply_rotary_pos_emb, code: return q_embed.to(orig_dtype), k_embed.to(orig_dtype)
        convert_element_type_161: "bf16[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(add_46, torch.bfloat16);  add_46 = None
        convert_element_type_162: "bf16[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(add_47, torch.bfloat16);  add_47 = None
        
         # File: C:\Users\arroy\anaconda3\envs\hrm_v1\lib\site-packages\flash_attn\flash_attn_interface.py:839 in forward, code: out_padded, softmax_lse, S_dmask, rng_state = _wrapped_flash_attn_forward(
        _flash_attn_forward_7 = torch.ops.flash_attn._flash_attn_forward.default(convert_element_type_161, convert_element_type_162, slice_100, 0.0, 0.125, False, -1, -1, 0.0, None, False);  convert_element_type_161 = convert_element_type_162 = slice_100 = None
        getitem_42: "bf16[128, 901, 2, 64]" = _flash_attn_forward_7[0];  _flash_attn_forward_7 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:135 in forward, code: attn_output = attn_output.view(batch_size, seq_len, self.output_size)  # type: ignore
        view_79: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(getitem_42, [128, 901, 128]);  getitem_42 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_80: "bf16[115328, 128]" = torch.ops.aten.reshape.default(view_79, [115328, 128]);  view_79 = None
        mm_29: "bf16[115328, 128]" = torch.ops.aten.mm.default(view_80, permute_13);  view_80 = None
        view_81: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(mm_29, [128, 901, 128]);  mm_29 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:80 in forward, code: hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        add_48: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(convert_element_type_155, view_81);  convert_element_type_155 = view_81 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:154 in rms_norm, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_166: "f32[128, 901, 128]" = torch.ops.prims.convert_element_type.default(add_48, torch.float32);  add_48 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:156 in rms_norm, code: variance = hidden_states.square().mean(-1, keepdim=True)
        pow_15: "f32[128, 901, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_166, 2)
        mean_14: "f32[128, 901, 1]" = torch.ops.aten.mean.dim(pow_15, [-1], True);  pow_15 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:157 in rms_norm, code: hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        add_49: "f32[128, 901, 1]" = torch.ops.aten.add.Tensor(mean_14, 1e-05);  mean_14 = None
        rsqrt_14: "f32[128, 901, 1]" = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
        mul_61: "f32[128, 901, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_166, rsqrt_14);  convert_element_type_166 = rsqrt_14 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:158 in rms_norm, code: return hidden_states.to(input_dtype)
        convert_element_type_167: "bf16[128, 901, 128]" = torch.ops.prims.convert_element_type.default(mul_61, torch.bfloat16);  mul_61 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_82: "bf16[115328, 128]" = torch.ops.aten.reshape.default(convert_element_type_167, [115328, 128])
        mm_30: "bf16[115328, 1024]" = torch.ops.aten.mm.default(view_82, permute_14);  view_82 = None
        view_83: "bf16[128, 901, 1024]" = torch.ops.aten.reshape.default(mm_30, [128, 901, 1024]);  mm_30 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:148 in forward, code: gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        split_7 = torch.ops.aten.split.Tensor(view_83, 512, -1);  view_83 = None
        getitem_46: "bf16[128, 901, 512]" = split_7[0]
        getitem_47: "bf16[128, 901, 512]" = split_7[1];  split_7 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:149 in forward, code: return self.down_proj(F.silu(gate) * up)
        convert_element_type_171: "f32[128, 901, 512]" = torch.ops.prims.convert_element_type.default(getitem_46, torch.float32);  getitem_46 = None
        sigmoid_7: "f32[128, 901, 512]" = torch.ops.aten.sigmoid.default(convert_element_type_171)
        mul_62: "f32[128, 901, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_171, sigmoid_7);  convert_element_type_171 = sigmoid_7 = None
        convert_element_type_172: "bf16[128, 901, 512]" = torch.ops.prims.convert_element_type.default(mul_62, torch.bfloat16);  mul_62 = None
        mul_63: "bf16[128, 901, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_172, getitem_47);  convert_element_type_172 = getitem_47 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_84: "bf16[115328, 512]" = torch.ops.aten.reshape.default(mul_63, [115328, 512]);  mul_63 = None
        mm_31: "bf16[115328, 128]" = torch.ops.aten.mm.default(view_84, permute_15);  view_84 = None
        view_85: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(mm_31, [128, 901, 128]);  mm_31 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:82 in forward, code: hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)
        add_50: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(convert_element_type_167, view_85);  convert_element_type_167 = view_85 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:154 in rms_norm, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_176: "f32[128, 901, 128]" = torch.ops.prims.convert_element_type.default(add_50, torch.float32);  add_50 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:156 in rms_norm, code: variance = hidden_states.square().mean(-1, keepdim=True)
        pow_16: "f32[128, 901, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_176, 2)
        mean_15: "f32[128, 901, 1]" = torch.ops.aten.mean.dim(pow_16, [-1], True);  pow_16 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:157 in rms_norm, code: hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        add_51: "f32[128, 901, 1]" = torch.ops.aten.add.Tensor(mean_15, 1e-05);  mean_15 = None
        rsqrt_15: "f32[128, 901, 1]" = torch.ops.aten.rsqrt.default(add_51);  add_51 = None
        mul_64: "f32[128, 901, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_176, rsqrt_15);  convert_element_type_176 = rsqrt_15 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:158 in rms_norm, code: return hidden_states.to(input_dtype)
        convert_element_type_177: "bf16[128, 901, 128]" = torch.ops.prims.convert_element_type.default(mul_64, torch.bfloat16);  mul_64 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:94 in forward, code: hidden_states = hidden_states + input_injection
        add_52: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(where, convert_element_type_177);  where = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        convert_element_type_178: "bf16[384, 128]" = torch.ops.prims.convert_element_type.default(primals_35, torch.bfloat16)
        permute_32: "bf16[128, 384]" = torch.ops.aten.permute.default(convert_element_type_178, [1, 0]);  convert_element_type_178 = None
        view_86: "bf16[115328, 128]" = torch.ops.aten.reshape.default(add_52, [115328, 128])
        mm_32: "bf16[115328, 384]" = torch.ops.aten.mm.default(view_86, permute_32);  view_86 = None
        view_87: "bf16[128, 901, 384]" = torch.ops.aten.reshape.default(mm_32, [128, 901, 384]);  mm_32 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:119 in forward, code: qkv = qkv.view(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        view_88: "bf16[128, 901, 6, 64]" = torch.ops.aten.reshape.default(view_87, [128, 901, 6, 64]);  view_87 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:120 in forward, code: query = qkv[:, :, :self.num_heads]
        slice_107: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_88, 2, 0, 2)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:121 in forward, code: key = qkv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
        slice_110: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_88, 2, 2, 4)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:122 in forward, code: value = qkv[:, :, self.num_heads + self.num_key_value_heads:]
        slice_113: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_88, 2, 4, 9223372036854775807);  view_88 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:34 in apply_rotary_pos_emb, code: q = q.to(cos.dtype)
        convert_element_type_181: "f32[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(slice_107, torch.float32);  slice_107 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:35 in apply_rotary_pos_emb, code: k = k.to(cos.dtype)
        convert_element_type_182: "f32[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(slice_110, torch.float32);  slice_110 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:37 in apply_rotary_pos_emb, code: q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
        mul_65: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_181, unsqueeze)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:25 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_114: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_181, 3, 0, 32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:26 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_115: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_181, 3, 32, 9223372036854775807);  convert_element_type_181 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:27 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_16: "f32[128, 901, 2, 32]" = torch.ops.aten.neg.default(slice_115);  slice_115 = None
        cat_17: "f32[128, 901, 2, 64]" = torch.ops.aten.cat.default([neg_16, slice_114], -1);  neg_16 = slice_114 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:37 in apply_rotary_pos_emb, code: q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
        mul_66: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(cat_17, unsqueeze_1);  cat_17 = None
        add_53: "f32[128, 901, 2, 64]" = torch.ops.aten.add.Tensor(mul_65, mul_66);  mul_65 = mul_66 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:38 in apply_rotary_pos_emb, code: k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
        mul_67: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_182, unsqueeze)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:25 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_116: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_182, 3, 0, 32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:26 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_117: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_182, 3, 32, 9223372036854775807);  convert_element_type_182 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:27 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_17: "f32[128, 901, 2, 32]" = torch.ops.aten.neg.default(slice_117);  slice_117 = None
        cat_18: "f32[128, 901, 2, 64]" = torch.ops.aten.cat.default([neg_17, slice_116], -1);  neg_17 = slice_116 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:38 in apply_rotary_pos_emb, code: k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
        mul_68: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(cat_18, unsqueeze_1);  cat_18 = None
        add_54: "f32[128, 901, 2, 64]" = torch.ops.aten.add.Tensor(mul_67, mul_68);  mul_67 = mul_68 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:40 in apply_rotary_pos_emb, code: return q_embed.to(orig_dtype), k_embed.to(orig_dtype)
        convert_element_type_183: "bf16[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(add_53, torch.bfloat16);  add_53 = None
        convert_element_type_184: "bf16[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(add_54, torch.bfloat16);  add_54 = None
        
         # File: C:\Users\arroy\anaconda3\envs\hrm_v1\lib\site-packages\flash_attn\flash_attn_interface.py:839 in forward, code: out_padded, softmax_lse, S_dmask, rng_state = _wrapped_flash_attn_forward(
        _flash_attn_forward_8 = torch.ops.flash_attn._flash_attn_forward.default(convert_element_type_183, convert_element_type_184, slice_113, 0.0, 0.125, False, -1, -1, 0.0, None, False);  convert_element_type_183 = convert_element_type_184 = slice_113 = None
        getitem_48: "bf16[128, 901, 2, 64]" = _flash_attn_forward_8[0];  _flash_attn_forward_8 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:135 in forward, code: attn_output = attn_output.view(batch_size, seq_len, self.output_size)  # type: ignore
        view_89: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(getitem_48, [128, 901, 128]);  getitem_48 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        convert_element_type_185: "bf16[128, 128]" = torch.ops.prims.convert_element_type.default(primals_36, torch.bfloat16)
        permute_33: "bf16[128, 128]" = torch.ops.aten.permute.default(convert_element_type_185, [1, 0]);  convert_element_type_185 = None
        view_90: "bf16[115328, 128]" = torch.ops.aten.reshape.default(view_89, [115328, 128]);  view_89 = None
        mm_33: "bf16[115328, 128]" = torch.ops.aten.mm.default(view_90, permute_33);  view_90 = None
        view_91: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(mm_33, [128, 901, 128]);  mm_33 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:80 in forward, code: hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        add_55: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(add_52, view_91);  add_52 = view_91 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:154 in rms_norm, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_188: "f32[128, 901, 128]" = torch.ops.prims.convert_element_type.default(add_55, torch.float32);  add_55 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:156 in rms_norm, code: variance = hidden_states.square().mean(-1, keepdim=True)
        pow_17: "f32[128, 901, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_188, 2)
        mean_16: "f32[128, 901, 1]" = torch.ops.aten.mean.dim(pow_17, [-1], True);  pow_17 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:157 in rms_norm, code: hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        add_56: "f32[128, 901, 1]" = torch.ops.aten.add.Tensor(mean_16, 1e-05);  mean_16 = None
        rsqrt_16: "f32[128, 901, 1]" = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
        mul_69: "f32[128, 901, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_188, rsqrt_16);  convert_element_type_188 = rsqrt_16 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:158 in rms_norm, code: return hidden_states.to(input_dtype)
        convert_element_type_189: "bf16[128, 901, 128]" = torch.ops.prims.convert_element_type.default(mul_69, torch.bfloat16);  mul_69 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        convert_element_type_190: "bf16[1024, 128]" = torch.ops.prims.convert_element_type.default(primals_37, torch.bfloat16)
        permute_34: "bf16[128, 1024]" = torch.ops.aten.permute.default(convert_element_type_190, [1, 0]);  convert_element_type_190 = None
        view_92: "bf16[115328, 128]" = torch.ops.aten.reshape.default(convert_element_type_189, [115328, 128])
        mm_34: "bf16[115328, 1024]" = torch.ops.aten.mm.default(view_92, permute_34);  view_92 = None
        view_93: "bf16[128, 901, 1024]" = torch.ops.aten.reshape.default(mm_34, [128, 901, 1024]);  mm_34 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:148 in forward, code: gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        split_8 = torch.ops.aten.split.Tensor(view_93, 512, -1);  view_93 = None
        getitem_52: "bf16[128, 901, 512]" = split_8[0]
        getitem_53: "bf16[128, 901, 512]" = split_8[1];  split_8 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:149 in forward, code: return self.down_proj(F.silu(gate) * up)
        convert_element_type_193: "f32[128, 901, 512]" = torch.ops.prims.convert_element_type.default(getitem_52, torch.float32);  getitem_52 = None
        sigmoid_8: "f32[128, 901, 512]" = torch.ops.aten.sigmoid.default(convert_element_type_193)
        mul_70: "f32[128, 901, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_193, sigmoid_8);  convert_element_type_193 = sigmoid_8 = None
        convert_element_type_194: "bf16[128, 901, 512]" = torch.ops.prims.convert_element_type.default(mul_70, torch.bfloat16);  mul_70 = None
        mul_71: "bf16[128, 901, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_194, getitem_53);  convert_element_type_194 = getitem_53 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        convert_element_type_195: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(primals_38, torch.bfloat16)
        permute_35: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_195, [1, 0]);  convert_element_type_195 = None
        view_94: "bf16[115328, 512]" = torch.ops.aten.reshape.default(mul_71, [115328, 512]);  mul_71 = None
        mm_35: "bf16[115328, 128]" = torch.ops.aten.mm.default(view_94, permute_35);  view_94 = None
        view_95: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(mm_35, [128, 901, 128]);  mm_35 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:82 in forward, code: hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)
        add_57: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(convert_element_type_189, view_95);  convert_element_type_189 = view_95 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:154 in rms_norm, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_198: "f32[128, 901, 128]" = torch.ops.prims.convert_element_type.default(add_57, torch.float32);  add_57 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:156 in rms_norm, code: variance = hidden_states.square().mean(-1, keepdim=True)
        pow_18: "f32[128, 901, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_198, 2)
        mean_17: "f32[128, 901, 1]" = torch.ops.aten.mean.dim(pow_18, [-1], True);  pow_18 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:157 in rms_norm, code: hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        add_58: "f32[128, 901, 1]" = torch.ops.aten.add.Tensor(mean_17, 1e-05);  mean_17 = None
        rsqrt_17: "f32[128, 901, 1]" = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
        mul_72: "f32[128, 901, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_198, rsqrt_17);  convert_element_type_198 = rsqrt_17 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:158 in rms_norm, code: return hidden_states.to(input_dtype)
        convert_element_type_199: "bf16[128, 901, 128]" = torch.ops.prims.convert_element_type.default(mul_72, torch.bfloat16);  mul_72 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        convert_element_type_200: "bf16[384, 128]" = torch.ops.prims.convert_element_type.default(primals_39, torch.bfloat16)
        permute_36: "bf16[128, 384]" = torch.ops.aten.permute.default(convert_element_type_200, [1, 0]);  convert_element_type_200 = None
        view_96: "bf16[115328, 128]" = torch.ops.aten.reshape.default(convert_element_type_199, [115328, 128])
        mm_36: "bf16[115328, 384]" = torch.ops.aten.mm.default(view_96, permute_36);  view_96 = None
        view_97: "bf16[128, 901, 384]" = torch.ops.aten.reshape.default(mm_36, [128, 901, 384]);  mm_36 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:119 in forward, code: qkv = qkv.view(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        view_98: "bf16[128, 901, 6, 64]" = torch.ops.aten.reshape.default(view_97, [128, 901, 6, 64]);  view_97 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:120 in forward, code: query = qkv[:, :, :self.num_heads]
        slice_120: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_98, 2, 0, 2)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:121 in forward, code: key = qkv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
        slice_123: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_98, 2, 2, 4)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:122 in forward, code: value = qkv[:, :, self.num_heads + self.num_key_value_heads:]
        slice_126: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_98, 2, 4, 9223372036854775807);  view_98 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:34 in apply_rotary_pos_emb, code: q = q.to(cos.dtype)
        convert_element_type_203: "f32[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(slice_120, torch.float32);  slice_120 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:35 in apply_rotary_pos_emb, code: k = k.to(cos.dtype)
        convert_element_type_204: "f32[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(slice_123, torch.float32);  slice_123 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:37 in apply_rotary_pos_emb, code: q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
        mul_73: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_203, unsqueeze)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:25 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_127: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_203, 3, 0, 32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:26 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_128: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_203, 3, 32, 9223372036854775807);  convert_element_type_203 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:27 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_18: "f32[128, 901, 2, 32]" = torch.ops.aten.neg.default(slice_128);  slice_128 = None
        cat_19: "f32[128, 901, 2, 64]" = torch.ops.aten.cat.default([neg_18, slice_127], -1);  neg_18 = slice_127 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:37 in apply_rotary_pos_emb, code: q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
        mul_74: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(cat_19, unsqueeze_1);  cat_19 = None
        add_59: "f32[128, 901, 2, 64]" = torch.ops.aten.add.Tensor(mul_73, mul_74);  mul_73 = mul_74 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:38 in apply_rotary_pos_emb, code: k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
        mul_75: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_204, unsqueeze)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:25 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_129: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_204, 3, 0, 32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:26 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_130: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_204, 3, 32, 9223372036854775807);  convert_element_type_204 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:27 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_19: "f32[128, 901, 2, 32]" = torch.ops.aten.neg.default(slice_130);  slice_130 = None
        cat_20: "f32[128, 901, 2, 64]" = torch.ops.aten.cat.default([neg_19, slice_129], -1);  neg_19 = slice_129 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:38 in apply_rotary_pos_emb, code: k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
        mul_76: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(cat_20, unsqueeze_1);  cat_20 = None
        add_60: "f32[128, 901, 2, 64]" = torch.ops.aten.add.Tensor(mul_75, mul_76);  mul_75 = mul_76 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:40 in apply_rotary_pos_emb, code: return q_embed.to(orig_dtype), k_embed.to(orig_dtype)
        convert_element_type_205: "bf16[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(add_59, torch.bfloat16);  add_59 = None
        convert_element_type_206: "bf16[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(add_60, torch.bfloat16);  add_60 = None
        
         # File: C:\Users\arroy\anaconda3\envs\hrm_v1\lib\site-packages\flash_attn\flash_attn_interface.py:839 in forward, code: out_padded, softmax_lse, S_dmask, rng_state = _wrapped_flash_attn_forward(
        _flash_attn_forward_9 = torch.ops.flash_attn._flash_attn_forward.default(convert_element_type_205, convert_element_type_206, slice_126, 0.0, 0.125, False, -1, -1, 0.0, None, False);  convert_element_type_205 = convert_element_type_206 = slice_126 = None
        getitem_54: "bf16[128, 901, 2, 64]" = _flash_attn_forward_9[0];  _flash_attn_forward_9 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:135 in forward, code: attn_output = attn_output.view(batch_size, seq_len, self.output_size)  # type: ignore
        view_99: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(getitem_54, [128, 901, 128]);  getitem_54 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        convert_element_type_207: "bf16[128, 128]" = torch.ops.prims.convert_element_type.default(primals_40, torch.bfloat16)
        permute_37: "bf16[128, 128]" = torch.ops.aten.permute.default(convert_element_type_207, [1, 0]);  convert_element_type_207 = None
        view_100: "bf16[115328, 128]" = torch.ops.aten.reshape.default(view_99, [115328, 128]);  view_99 = None
        mm_37: "bf16[115328, 128]" = torch.ops.aten.mm.default(view_100, permute_37);  view_100 = None
        view_101: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(mm_37, [128, 901, 128]);  mm_37 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:80 in forward, code: hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        add_61: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(convert_element_type_199, view_101);  convert_element_type_199 = view_101 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:154 in rms_norm, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_210: "f32[128, 901, 128]" = torch.ops.prims.convert_element_type.default(add_61, torch.float32);  add_61 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:156 in rms_norm, code: variance = hidden_states.square().mean(-1, keepdim=True)
        pow_19: "f32[128, 901, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_210, 2)
        mean_18: "f32[128, 901, 1]" = torch.ops.aten.mean.dim(pow_19, [-1], True);  pow_19 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:157 in rms_norm, code: hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        add_62: "f32[128, 901, 1]" = torch.ops.aten.add.Tensor(mean_18, 1e-05);  mean_18 = None
        rsqrt_18: "f32[128, 901, 1]" = torch.ops.aten.rsqrt.default(add_62);  add_62 = None
        mul_77: "f32[128, 901, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_210, rsqrt_18);  convert_element_type_210 = rsqrt_18 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:158 in rms_norm, code: return hidden_states.to(input_dtype)
        convert_element_type_211: "bf16[128, 901, 128]" = torch.ops.prims.convert_element_type.default(mul_77, torch.bfloat16);  mul_77 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        convert_element_type_212: "bf16[1024, 128]" = torch.ops.prims.convert_element_type.default(primals_41, torch.bfloat16)
        permute_38: "bf16[128, 1024]" = torch.ops.aten.permute.default(convert_element_type_212, [1, 0]);  convert_element_type_212 = None
        view_102: "bf16[115328, 128]" = torch.ops.aten.reshape.default(convert_element_type_211, [115328, 128])
        mm_38: "bf16[115328, 1024]" = torch.ops.aten.mm.default(view_102, permute_38);  view_102 = None
        view_103: "bf16[128, 901, 1024]" = torch.ops.aten.reshape.default(mm_38, [128, 901, 1024]);  mm_38 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:148 in forward, code: gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        split_9 = torch.ops.aten.split.Tensor(view_103, 512, -1);  view_103 = None
        getitem_58: "bf16[128, 901, 512]" = split_9[0]
        getitem_59: "bf16[128, 901, 512]" = split_9[1];  split_9 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:149 in forward, code: return self.down_proj(F.silu(gate) * up)
        convert_element_type_215: "f32[128, 901, 512]" = torch.ops.prims.convert_element_type.default(getitem_58, torch.float32);  getitem_58 = None
        sigmoid_9: "f32[128, 901, 512]" = torch.ops.aten.sigmoid.default(convert_element_type_215)
        mul_78: "f32[128, 901, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_215, sigmoid_9);  convert_element_type_215 = sigmoid_9 = None
        convert_element_type_216: "bf16[128, 901, 512]" = torch.ops.prims.convert_element_type.default(mul_78, torch.bfloat16);  mul_78 = None
        mul_79: "bf16[128, 901, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_216, getitem_59);  convert_element_type_216 = getitem_59 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        convert_element_type_217: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(primals_42, torch.bfloat16)
        permute_39: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_217, [1, 0]);  convert_element_type_217 = None
        view_104: "bf16[115328, 512]" = torch.ops.aten.reshape.default(mul_79, [115328, 512]);  mul_79 = None
        mm_39: "bf16[115328, 128]" = torch.ops.aten.mm.default(view_104, permute_39);  view_104 = None
        view_105: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(mm_39, [128, 901, 128]);  mm_39 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:82 in forward, code: hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)
        add_63: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(convert_element_type_211, view_105);  convert_element_type_211 = view_105 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:154 in rms_norm, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_220: "f32[128, 901, 128]" = torch.ops.prims.convert_element_type.default(add_63, torch.float32);  add_63 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:156 in rms_norm, code: variance = hidden_states.square().mean(-1, keepdim=True)
        pow_20: "f32[128, 901, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_220, 2)
        mean_19: "f32[128, 901, 1]" = torch.ops.aten.mean.dim(pow_20, [-1], True);  pow_20 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:157 in rms_norm, code: hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        add_64: "f32[128, 901, 1]" = torch.ops.aten.add.Tensor(mean_19, 1e-05);  mean_19 = None
        rsqrt_19: "f32[128, 901, 1]" = torch.ops.aten.rsqrt.default(add_64);  add_64 = None
        mul_80: "f32[128, 901, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_220, rsqrt_19);  convert_element_type_220 = rsqrt_19 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:158 in rms_norm, code: return hidden_states.to(input_dtype)
        convert_element_type_221: "bf16[128, 901, 128]" = torch.ops.prims.convert_element_type.default(mul_80, torch.bfloat16);  mul_80 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        convert_element_type_222: "bf16[384, 128]" = torch.ops.prims.convert_element_type.default(primals_43, torch.bfloat16)
        permute_40: "bf16[128, 384]" = torch.ops.aten.permute.default(convert_element_type_222, [1, 0]);  convert_element_type_222 = None
        view_106: "bf16[115328, 128]" = torch.ops.aten.reshape.default(convert_element_type_221, [115328, 128])
        mm_40: "bf16[115328, 384]" = torch.ops.aten.mm.default(view_106, permute_40);  view_106 = None
        view_107: "bf16[128, 901, 384]" = torch.ops.aten.reshape.default(mm_40, [128, 901, 384]);  mm_40 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:119 in forward, code: qkv = qkv.view(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        view_108: "bf16[128, 901, 6, 64]" = torch.ops.aten.reshape.default(view_107, [128, 901, 6, 64]);  view_107 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:120 in forward, code: query = qkv[:, :, :self.num_heads]
        slice_133: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_108, 2, 0, 2)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:121 in forward, code: key = qkv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
        slice_136: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_108, 2, 2, 4)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:122 in forward, code: value = qkv[:, :, self.num_heads + self.num_key_value_heads:]
        slice_139: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_108, 2, 4, 9223372036854775807);  view_108 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:34 in apply_rotary_pos_emb, code: q = q.to(cos.dtype)
        convert_element_type_225: "f32[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(slice_133, torch.float32);  slice_133 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:35 in apply_rotary_pos_emb, code: k = k.to(cos.dtype)
        convert_element_type_226: "f32[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(slice_136, torch.float32);  slice_136 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:37 in apply_rotary_pos_emb, code: q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
        mul_81: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_225, unsqueeze)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:25 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_140: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_225, 3, 0, 32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:26 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_141: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_225, 3, 32, 9223372036854775807);  convert_element_type_225 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:27 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_20: "f32[128, 901, 2, 32]" = torch.ops.aten.neg.default(slice_141);  slice_141 = None
        cat_21: "f32[128, 901, 2, 64]" = torch.ops.aten.cat.default([neg_20, slice_140], -1);  neg_20 = slice_140 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:37 in apply_rotary_pos_emb, code: q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
        mul_82: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(cat_21, unsqueeze_1);  cat_21 = None
        add_65: "f32[128, 901, 2, 64]" = torch.ops.aten.add.Tensor(mul_81, mul_82);  mul_81 = mul_82 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:38 in apply_rotary_pos_emb, code: k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
        mul_83: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_226, unsqueeze)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:25 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_142: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_226, 3, 0, 32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:26 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_143: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_226, 3, 32, 9223372036854775807);  convert_element_type_226 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:27 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_21: "f32[128, 901, 2, 32]" = torch.ops.aten.neg.default(slice_143);  slice_143 = None
        cat_22: "f32[128, 901, 2, 64]" = torch.ops.aten.cat.default([neg_21, slice_142], -1);  neg_21 = slice_142 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:38 in apply_rotary_pos_emb, code: k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
        mul_84: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(cat_22, unsqueeze_1);  cat_22 = None
        add_66: "f32[128, 901, 2, 64]" = torch.ops.aten.add.Tensor(mul_83, mul_84);  mul_83 = mul_84 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:40 in apply_rotary_pos_emb, code: return q_embed.to(orig_dtype), k_embed.to(orig_dtype)
        convert_element_type_227: "bf16[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(add_65, torch.bfloat16);  add_65 = None
        convert_element_type_228: "bf16[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(add_66, torch.bfloat16);  add_66 = None
        
         # File: C:\Users\arroy\anaconda3\envs\hrm_v1\lib\site-packages\flash_attn\flash_attn_interface.py:839 in forward, code: out_padded, softmax_lse, S_dmask, rng_state = _wrapped_flash_attn_forward(
        _flash_attn_forward_10 = torch.ops.flash_attn._flash_attn_forward.default(convert_element_type_227, convert_element_type_228, slice_139, 0.0, 0.125, False, -1, -1, 0.0, None, False);  convert_element_type_227 = convert_element_type_228 = slice_139 = None
        getitem_60: "bf16[128, 901, 2, 64]" = _flash_attn_forward_10[0];  _flash_attn_forward_10 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:135 in forward, code: attn_output = attn_output.view(batch_size, seq_len, self.output_size)  # type: ignore
        view_109: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(getitem_60, [128, 901, 128]);  getitem_60 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        convert_element_type_229: "bf16[128, 128]" = torch.ops.prims.convert_element_type.default(primals_44, torch.bfloat16)
        permute_41: "bf16[128, 128]" = torch.ops.aten.permute.default(convert_element_type_229, [1, 0]);  convert_element_type_229 = None
        view_110: "bf16[115328, 128]" = torch.ops.aten.reshape.default(view_109, [115328, 128]);  view_109 = None
        mm_41: "bf16[115328, 128]" = torch.ops.aten.mm.default(view_110, permute_41);  view_110 = None
        view_111: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(mm_41, [128, 901, 128]);  mm_41 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:80 in forward, code: hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        add_67: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(convert_element_type_221, view_111);  convert_element_type_221 = view_111 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:154 in rms_norm, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_232: "f32[128, 901, 128]" = torch.ops.prims.convert_element_type.default(add_67, torch.float32);  add_67 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:156 in rms_norm, code: variance = hidden_states.square().mean(-1, keepdim=True)
        pow_21: "f32[128, 901, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_232, 2)
        mean_20: "f32[128, 901, 1]" = torch.ops.aten.mean.dim(pow_21, [-1], True);  pow_21 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:157 in rms_norm, code: hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        add_68: "f32[128, 901, 1]" = torch.ops.aten.add.Tensor(mean_20, 1e-05);  mean_20 = None
        rsqrt_20: "f32[128, 901, 1]" = torch.ops.aten.rsqrt.default(add_68);  add_68 = None
        mul_85: "f32[128, 901, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_232, rsqrt_20);  convert_element_type_232 = rsqrt_20 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:158 in rms_norm, code: return hidden_states.to(input_dtype)
        convert_element_type_233: "bf16[128, 901, 128]" = torch.ops.prims.convert_element_type.default(mul_85, torch.bfloat16);  mul_85 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        convert_element_type_234: "bf16[1024, 128]" = torch.ops.prims.convert_element_type.default(primals_45, torch.bfloat16)
        permute_42: "bf16[128, 1024]" = torch.ops.aten.permute.default(convert_element_type_234, [1, 0]);  convert_element_type_234 = None
        view_112: "bf16[115328, 128]" = torch.ops.aten.reshape.default(convert_element_type_233, [115328, 128])
        mm_42: "bf16[115328, 1024]" = torch.ops.aten.mm.default(view_112, permute_42);  view_112 = None
        view_113: "bf16[128, 901, 1024]" = torch.ops.aten.reshape.default(mm_42, [128, 901, 1024]);  mm_42 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:148 in forward, code: gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        split_10 = torch.ops.aten.split.Tensor(view_113, 512, -1);  view_113 = None
        getitem_64: "bf16[128, 901, 512]" = split_10[0]
        getitem_65: "bf16[128, 901, 512]" = split_10[1];  split_10 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:149 in forward, code: return self.down_proj(F.silu(gate) * up)
        convert_element_type_237: "f32[128, 901, 512]" = torch.ops.prims.convert_element_type.default(getitem_64, torch.float32);  getitem_64 = None
        sigmoid_10: "f32[128, 901, 512]" = torch.ops.aten.sigmoid.default(convert_element_type_237)
        mul_86: "f32[128, 901, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_237, sigmoid_10);  convert_element_type_237 = sigmoid_10 = None
        convert_element_type_238: "bf16[128, 901, 512]" = torch.ops.prims.convert_element_type.default(mul_86, torch.bfloat16);  mul_86 = None
        mul_87: "bf16[128, 901, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_238, getitem_65);  convert_element_type_238 = getitem_65 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        convert_element_type_239: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(primals_46, torch.bfloat16)
        permute_43: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_239, [1, 0]);  convert_element_type_239 = None
        view_114: "bf16[115328, 512]" = torch.ops.aten.reshape.default(mul_87, [115328, 512]);  mul_87 = None
        mm_43: "bf16[115328, 128]" = torch.ops.aten.mm.default(view_114, permute_43);  view_114 = None
        view_115: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(mm_43, [128, 901, 128]);  mm_43 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:82 in forward, code: hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)
        add_69: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(convert_element_type_233, view_115);  convert_element_type_233 = view_115 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:154 in rms_norm, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_242: "f32[128, 901, 128]" = torch.ops.prims.convert_element_type.default(add_69, torch.float32);  add_69 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:156 in rms_norm, code: variance = hidden_states.square().mean(-1, keepdim=True)
        pow_22: "f32[128, 901, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_242, 2)
        mean_21: "f32[128, 901, 1]" = torch.ops.aten.mean.dim(pow_22, [-1], True);  pow_22 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:157 in rms_norm, code: hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        add_70: "f32[128, 901, 1]" = torch.ops.aten.add.Tensor(mean_21, 1e-05);  mean_21 = None
        rsqrt_21: "f32[128, 901, 1]" = torch.ops.aten.rsqrt.default(add_70);  add_70 = None
        mul_88: "f32[128, 901, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_242, rsqrt_21);  convert_element_type_242 = rsqrt_21 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:158 in rms_norm, code: return hidden_states.to(input_dtype)
        convert_element_type_243: "bf16[128, 901, 128]" = torch.ops.prims.convert_element_type.default(mul_88, torch.bfloat16);  mul_88 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        convert_element_type_244: "bf16[384, 128]" = torch.ops.prims.convert_element_type.default(primals_47, torch.bfloat16)
        permute_44: "bf16[128, 384]" = torch.ops.aten.permute.default(convert_element_type_244, [1, 0]);  convert_element_type_244 = None
        view_116: "bf16[115328, 128]" = torch.ops.aten.reshape.default(convert_element_type_243, [115328, 128])
        mm_44: "bf16[115328, 384]" = torch.ops.aten.mm.default(view_116, permute_44);  view_116 = None
        view_117: "bf16[128, 901, 384]" = torch.ops.aten.reshape.default(mm_44, [128, 901, 384]);  mm_44 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:119 in forward, code: qkv = qkv.view(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        view_118: "bf16[128, 901, 6, 64]" = torch.ops.aten.reshape.default(view_117, [128, 901, 6, 64]);  view_117 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:120 in forward, code: query = qkv[:, :, :self.num_heads]
        slice_146: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_118, 2, 0, 2)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:121 in forward, code: key = qkv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
        slice_149: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_118, 2, 2, 4)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:122 in forward, code: value = qkv[:, :, self.num_heads + self.num_key_value_heads:]
        slice_152: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_118, 2, 4, 9223372036854775807);  view_118 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:34 in apply_rotary_pos_emb, code: q = q.to(cos.dtype)
        convert_element_type_247: "f32[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(slice_146, torch.float32);  slice_146 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:35 in apply_rotary_pos_emb, code: k = k.to(cos.dtype)
        convert_element_type_248: "f32[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(slice_149, torch.float32);  slice_149 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:37 in apply_rotary_pos_emb, code: q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
        mul_89: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_247, unsqueeze)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:25 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_153: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_247, 3, 0, 32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:26 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_154: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_247, 3, 32, 9223372036854775807);  convert_element_type_247 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:27 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_22: "f32[128, 901, 2, 32]" = torch.ops.aten.neg.default(slice_154);  slice_154 = None
        cat_23: "f32[128, 901, 2, 64]" = torch.ops.aten.cat.default([neg_22, slice_153], -1);  neg_22 = slice_153 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:37 in apply_rotary_pos_emb, code: q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
        mul_90: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(cat_23, unsqueeze_1);  cat_23 = None
        add_71: "f32[128, 901, 2, 64]" = torch.ops.aten.add.Tensor(mul_89, mul_90);  mul_89 = mul_90 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:38 in apply_rotary_pos_emb, code: k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
        mul_91: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_248, unsqueeze)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:25 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_155: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_248, 3, 0, 32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:26 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_156: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_248, 3, 32, 9223372036854775807);  convert_element_type_248 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:27 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_23: "f32[128, 901, 2, 32]" = torch.ops.aten.neg.default(slice_156);  slice_156 = None
        cat_24: "f32[128, 901, 2, 64]" = torch.ops.aten.cat.default([neg_23, slice_155], -1);  neg_23 = slice_155 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:38 in apply_rotary_pos_emb, code: k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
        mul_92: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(cat_24, unsqueeze_1);  cat_24 = None
        add_72: "f32[128, 901, 2, 64]" = torch.ops.aten.add.Tensor(mul_91, mul_92);  mul_91 = mul_92 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:40 in apply_rotary_pos_emb, code: return q_embed.to(orig_dtype), k_embed.to(orig_dtype)
        convert_element_type_249: "bf16[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(add_71, torch.bfloat16);  add_71 = None
        convert_element_type_250: "bf16[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(add_72, torch.bfloat16);  add_72 = None
        
         # File: C:\Users\arroy\anaconda3\envs\hrm_v1\lib\site-packages\flash_attn\flash_attn_interface.py:839 in forward, code: out_padded, softmax_lse, S_dmask, rng_state = _wrapped_flash_attn_forward(
        _flash_attn_forward_11 = torch.ops.flash_attn._flash_attn_forward.default(convert_element_type_249, convert_element_type_250, slice_152, 0.0, 0.125, False, -1, -1, 0.0, None, False);  convert_element_type_249 = convert_element_type_250 = slice_152 = None
        getitem_66: "bf16[128, 901, 2, 64]" = _flash_attn_forward_11[0];  _flash_attn_forward_11 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:135 in forward, code: attn_output = attn_output.view(batch_size, seq_len, self.output_size)  # type: ignore
        view_119: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(getitem_66, [128, 901, 128]);  getitem_66 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        convert_element_type_251: "bf16[128, 128]" = torch.ops.prims.convert_element_type.default(primals_48, torch.bfloat16)
        permute_45: "bf16[128, 128]" = torch.ops.aten.permute.default(convert_element_type_251, [1, 0]);  convert_element_type_251 = None
        view_120: "bf16[115328, 128]" = torch.ops.aten.reshape.default(view_119, [115328, 128]);  view_119 = None
        mm_45: "bf16[115328, 128]" = torch.ops.aten.mm.default(view_120, permute_45);  view_120 = None
        view_121: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(mm_45, [128, 901, 128]);  mm_45 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:80 in forward, code: hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        add_73: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(convert_element_type_243, view_121);  convert_element_type_243 = view_121 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:154 in rms_norm, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_254: "f32[128, 901, 128]" = torch.ops.prims.convert_element_type.default(add_73, torch.float32);  add_73 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:156 in rms_norm, code: variance = hidden_states.square().mean(-1, keepdim=True)
        pow_23: "f32[128, 901, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_254, 2)
        mean_22: "f32[128, 901, 1]" = torch.ops.aten.mean.dim(pow_23, [-1], True);  pow_23 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:157 in rms_norm, code: hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        add_74: "f32[128, 901, 1]" = torch.ops.aten.add.Tensor(mean_22, 1e-05);  mean_22 = None
        rsqrt_22: "f32[128, 901, 1]" = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
        mul_93: "f32[128, 901, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_254, rsqrt_22);  convert_element_type_254 = rsqrt_22 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:158 in rms_norm, code: return hidden_states.to(input_dtype)
        convert_element_type_255: "bf16[128, 901, 128]" = torch.ops.prims.convert_element_type.default(mul_93, torch.bfloat16);  mul_93 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        convert_element_type_256: "bf16[1024, 128]" = torch.ops.prims.convert_element_type.default(primals_49, torch.bfloat16)
        permute_46: "bf16[128, 1024]" = torch.ops.aten.permute.default(convert_element_type_256, [1, 0]);  convert_element_type_256 = None
        view_122: "bf16[115328, 128]" = torch.ops.aten.reshape.default(convert_element_type_255, [115328, 128])
        mm_46: "bf16[115328, 1024]" = torch.ops.aten.mm.default(view_122, permute_46);  view_122 = None
        view_123: "bf16[128, 901, 1024]" = torch.ops.aten.reshape.default(mm_46, [128, 901, 1024]);  mm_46 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:148 in forward, code: gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        split_11 = torch.ops.aten.split.Tensor(view_123, 512, -1);  view_123 = None
        getitem_70: "bf16[128, 901, 512]" = split_11[0]
        getitem_71: "bf16[128, 901, 512]" = split_11[1];  split_11 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:149 in forward, code: return self.down_proj(F.silu(gate) * up)
        convert_element_type_259: "f32[128, 901, 512]" = torch.ops.prims.convert_element_type.default(getitem_70, torch.float32);  getitem_70 = None
        sigmoid_11: "f32[128, 901, 512]" = torch.ops.aten.sigmoid.default(convert_element_type_259)
        mul_94: "f32[128, 901, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_259, sigmoid_11);  convert_element_type_259 = sigmoid_11 = None
        convert_element_type_260: "bf16[128, 901, 512]" = torch.ops.prims.convert_element_type.default(mul_94, torch.bfloat16);  mul_94 = None
        mul_95: "bf16[128, 901, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_260, getitem_71);  convert_element_type_260 = getitem_71 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        convert_element_type_261: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(primals_50, torch.bfloat16)
        permute_47: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_261, [1, 0]);  convert_element_type_261 = None
        view_124: "bf16[115328, 512]" = torch.ops.aten.reshape.default(mul_95, [115328, 512]);  mul_95 = None
        mm_47: "bf16[115328, 128]" = torch.ops.aten.mm.default(view_124, permute_47);  view_124 = None
        view_125: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(mm_47, [128, 901, 128]);  mm_47 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:82 in forward, code: hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)
        add_75: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(convert_element_type_255, view_125);  convert_element_type_255 = view_125 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:154 in rms_norm, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_264: "f32[128, 901, 128]" = torch.ops.prims.convert_element_type.default(add_75, torch.float32);  add_75 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:156 in rms_norm, code: variance = hidden_states.square().mean(-1, keepdim=True)
        pow_24: "f32[128, 901, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_264, 2)
        mean_23: "f32[128, 901, 1]" = torch.ops.aten.mean.dim(pow_24, [-1], True);  pow_24 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:157 in rms_norm, code: hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        add_76: "f32[128, 901, 1]" = torch.ops.aten.add.Tensor(mean_23, 1e-05);  mean_23 = None
        rsqrt_23: "f32[128, 901, 1]" = torch.ops.aten.rsqrt.default(add_76);  add_76 = None
        mul_96: "f32[128, 901, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_264, rsqrt_23);  convert_element_type_264 = rsqrt_23 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:158 in rms_norm, code: return hidden_states.to(input_dtype)
        convert_element_type_265: "bf16[128, 901, 128]" = torch.ops.prims.convert_element_type.default(mul_96, torch.bfloat16);  mul_96 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:195 in forward, code: z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
        add_77: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(convert_element_type_265, mul);  mul = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:94 in forward, code: hidden_states = hidden_states + input_injection
        add_78: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(convert_element_type_177, add_77);  convert_element_type_177 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_126: "bf16[115328, 128]" = torch.ops.aten.reshape.default(add_78, [115328, 128])
        mm_48: "bf16[115328, 384]" = torch.ops.aten.mm.default(view_126, permute);  view_126 = None
        view_127: "bf16[128, 901, 384]" = torch.ops.aten.reshape.default(mm_48, [128, 901, 384]);  mm_48 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:119 in forward, code: qkv = qkv.view(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        view_128: "bf16[128, 901, 6, 64]" = torch.ops.aten.reshape.default(view_127, [128, 901, 6, 64]);  view_127 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:120 in forward, code: query = qkv[:, :, :self.num_heads]
        slice_159: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_128, 2, 0, 2)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:121 in forward, code: key = qkv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
        slice_162: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_128, 2, 2, 4)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:122 in forward, code: value = qkv[:, :, self.num_heads + self.num_key_value_heads:]
        slice_165: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_128, 2, 4, 9223372036854775807);  view_128 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:34 in apply_rotary_pos_emb, code: q = q.to(cos.dtype)
        convert_element_type_269: "f32[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(slice_159, torch.float32);  slice_159 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:35 in apply_rotary_pos_emb, code: k = k.to(cos.dtype)
        convert_element_type_270: "f32[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(slice_162, torch.float32);  slice_162 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:37 in apply_rotary_pos_emb, code: q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
        mul_97: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_269, unsqueeze)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:25 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_166: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_269, 3, 0, 32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:26 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_167: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_269, 3, 32, 9223372036854775807);  convert_element_type_269 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:27 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_24: "f32[128, 901, 2, 32]" = torch.ops.aten.neg.default(slice_167);  slice_167 = None
        cat_25: "f32[128, 901, 2, 64]" = torch.ops.aten.cat.default([neg_24, slice_166], -1);  neg_24 = slice_166 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:37 in apply_rotary_pos_emb, code: q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
        mul_98: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(cat_25, unsqueeze_1);  cat_25 = None
        add_79: "f32[128, 901, 2, 64]" = torch.ops.aten.add.Tensor(mul_97, mul_98);  mul_97 = mul_98 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:38 in apply_rotary_pos_emb, code: k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
        mul_99: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_270, unsqueeze)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:25 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_168: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_270, 3, 0, 32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:26 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_169: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_270, 3, 32, 9223372036854775807);  convert_element_type_270 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:27 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_25: "f32[128, 901, 2, 32]" = torch.ops.aten.neg.default(slice_169);  slice_169 = None
        cat_26: "f32[128, 901, 2, 64]" = torch.ops.aten.cat.default([neg_25, slice_168], -1);  neg_25 = slice_168 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:38 in apply_rotary_pos_emb, code: k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
        mul_100: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(cat_26, unsqueeze_1);  cat_26 = None
        add_80: "f32[128, 901, 2, 64]" = torch.ops.aten.add.Tensor(mul_99, mul_100);  mul_99 = mul_100 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:40 in apply_rotary_pos_emb, code: return q_embed.to(orig_dtype), k_embed.to(orig_dtype)
        convert_element_type_271: "bf16[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(add_79, torch.bfloat16);  add_79 = None
        convert_element_type_272: "bf16[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(add_80, torch.bfloat16);  add_80 = None
        
         # File: C:\Users\arroy\anaconda3\envs\hrm_v1\lib\site-packages\flash_attn\flash_attn_interface.py:839 in forward, code: out_padded, softmax_lse, S_dmask, rng_state = _wrapped_flash_attn_forward(
        _flash_attn_forward_12 = torch.ops.flash_attn._flash_attn_forward.default(convert_element_type_271, convert_element_type_272, slice_165, 0.0, 0.125, False, -1, -1, 0.0, None, False);  convert_element_type_271 = convert_element_type_272 = slice_165 = None
        getitem_72: "bf16[128, 901, 2, 64]" = _flash_attn_forward_12[0];  _flash_attn_forward_12 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:135 in forward, code: attn_output = attn_output.view(batch_size, seq_len, self.output_size)  # type: ignore
        view_129: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(getitem_72, [128, 901, 128]);  getitem_72 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_130: "bf16[115328, 128]" = torch.ops.aten.reshape.default(view_129, [115328, 128]);  view_129 = None
        mm_49: "bf16[115328, 128]" = torch.ops.aten.mm.default(view_130, permute_1);  view_130 = None
        view_131: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(mm_49, [128, 901, 128]);  mm_49 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:80 in forward, code: hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        add_81: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(add_78, view_131);  add_78 = view_131 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:154 in rms_norm, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_276: "f32[128, 901, 128]" = torch.ops.prims.convert_element_type.default(add_81, torch.float32);  add_81 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:156 in rms_norm, code: variance = hidden_states.square().mean(-1, keepdim=True)
        pow_25: "f32[128, 901, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_276, 2)
        mean_24: "f32[128, 901, 1]" = torch.ops.aten.mean.dim(pow_25, [-1], True);  pow_25 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:157 in rms_norm, code: hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        add_82: "f32[128, 901, 1]" = torch.ops.aten.add.Tensor(mean_24, 1e-05);  mean_24 = None
        rsqrt_24: "f32[128, 901, 1]" = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
        mul_101: "f32[128, 901, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_276, rsqrt_24);  convert_element_type_276 = rsqrt_24 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:158 in rms_norm, code: return hidden_states.to(input_dtype)
        convert_element_type_277: "bf16[128, 901, 128]" = torch.ops.prims.convert_element_type.default(mul_101, torch.bfloat16);  mul_101 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_132: "bf16[115328, 128]" = torch.ops.aten.reshape.default(convert_element_type_277, [115328, 128])
        mm_50: "bf16[115328, 1024]" = torch.ops.aten.mm.default(view_132, permute_2);  view_132 = None
        view_133: "bf16[128, 901, 1024]" = torch.ops.aten.reshape.default(mm_50, [128, 901, 1024]);  mm_50 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:148 in forward, code: gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        split_12 = torch.ops.aten.split.Tensor(view_133, 512, -1);  view_133 = None
        getitem_76: "bf16[128, 901, 512]" = split_12[0]
        getitem_77: "bf16[128, 901, 512]" = split_12[1];  split_12 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:149 in forward, code: return self.down_proj(F.silu(gate) * up)
        convert_element_type_281: "f32[128, 901, 512]" = torch.ops.prims.convert_element_type.default(getitem_76, torch.float32);  getitem_76 = None
        sigmoid_12: "f32[128, 901, 512]" = torch.ops.aten.sigmoid.default(convert_element_type_281)
        mul_102: "f32[128, 901, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_281, sigmoid_12);  convert_element_type_281 = sigmoid_12 = None
        convert_element_type_282: "bf16[128, 901, 512]" = torch.ops.prims.convert_element_type.default(mul_102, torch.bfloat16);  mul_102 = None
        mul_103: "bf16[128, 901, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_282, getitem_77);  convert_element_type_282 = getitem_77 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_134: "bf16[115328, 512]" = torch.ops.aten.reshape.default(mul_103, [115328, 512]);  mul_103 = None
        mm_51: "bf16[115328, 128]" = torch.ops.aten.mm.default(view_134, permute_3);  view_134 = None
        view_135: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(mm_51, [128, 901, 128]);  mm_51 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:82 in forward, code: hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)
        add_83: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(convert_element_type_277, view_135);  convert_element_type_277 = view_135 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:154 in rms_norm, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_286: "f32[128, 901, 128]" = torch.ops.prims.convert_element_type.default(add_83, torch.float32);  add_83 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:156 in rms_norm, code: variance = hidden_states.square().mean(-1, keepdim=True)
        pow_26: "f32[128, 901, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_286, 2)
        mean_25: "f32[128, 901, 1]" = torch.ops.aten.mean.dim(pow_26, [-1], True);  pow_26 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:157 in rms_norm, code: hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        add_84: "f32[128, 901, 1]" = torch.ops.aten.add.Tensor(mean_25, 1e-05);  mean_25 = None
        rsqrt_25: "f32[128, 901, 1]" = torch.ops.aten.rsqrt.default(add_84);  add_84 = None
        mul_104: "f32[128, 901, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_286, rsqrt_25);  convert_element_type_286 = rsqrt_25 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:158 in rms_norm, code: return hidden_states.to(input_dtype)
        convert_element_type_287: "bf16[128, 901, 128]" = torch.ops.prims.convert_element_type.default(mul_104, torch.bfloat16);  mul_104 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_136: "bf16[115328, 128]" = torch.ops.aten.reshape.default(convert_element_type_287, [115328, 128])
        mm_52: "bf16[115328, 384]" = torch.ops.aten.mm.default(view_136, permute_4);  view_136 = None
        view_137: "bf16[128, 901, 384]" = torch.ops.aten.reshape.default(mm_52, [128, 901, 384]);  mm_52 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:119 in forward, code: qkv = qkv.view(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        view_138: "bf16[128, 901, 6, 64]" = torch.ops.aten.reshape.default(view_137, [128, 901, 6, 64]);  view_137 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:120 in forward, code: query = qkv[:, :, :self.num_heads]
        slice_172: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_138, 2, 0, 2)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:121 in forward, code: key = qkv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
        slice_175: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_138, 2, 2, 4)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:122 in forward, code: value = qkv[:, :, self.num_heads + self.num_key_value_heads:]
        slice_178: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_138, 2, 4, 9223372036854775807);  view_138 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:34 in apply_rotary_pos_emb, code: q = q.to(cos.dtype)
        convert_element_type_291: "f32[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(slice_172, torch.float32);  slice_172 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:35 in apply_rotary_pos_emb, code: k = k.to(cos.dtype)
        convert_element_type_292: "f32[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(slice_175, torch.float32);  slice_175 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:37 in apply_rotary_pos_emb, code: q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
        mul_105: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_291, unsqueeze)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:25 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_179: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_291, 3, 0, 32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:26 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_180: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_291, 3, 32, 9223372036854775807);  convert_element_type_291 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:27 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_26: "f32[128, 901, 2, 32]" = torch.ops.aten.neg.default(slice_180);  slice_180 = None
        cat_27: "f32[128, 901, 2, 64]" = torch.ops.aten.cat.default([neg_26, slice_179], -1);  neg_26 = slice_179 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:37 in apply_rotary_pos_emb, code: q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
        mul_106: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(cat_27, unsqueeze_1);  cat_27 = None
        add_85: "f32[128, 901, 2, 64]" = torch.ops.aten.add.Tensor(mul_105, mul_106);  mul_105 = mul_106 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:38 in apply_rotary_pos_emb, code: k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
        mul_107: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_292, unsqueeze)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:25 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_181: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_292, 3, 0, 32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:26 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_182: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_292, 3, 32, 9223372036854775807);  convert_element_type_292 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:27 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_27: "f32[128, 901, 2, 32]" = torch.ops.aten.neg.default(slice_182);  slice_182 = None
        cat_28: "f32[128, 901, 2, 64]" = torch.ops.aten.cat.default([neg_27, slice_181], -1);  neg_27 = slice_181 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:38 in apply_rotary_pos_emb, code: k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
        mul_108: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(cat_28, unsqueeze_1);  cat_28 = None
        add_86: "f32[128, 901, 2, 64]" = torch.ops.aten.add.Tensor(mul_107, mul_108);  mul_107 = mul_108 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:40 in apply_rotary_pos_emb, code: return q_embed.to(orig_dtype), k_embed.to(orig_dtype)
        convert_element_type_293: "bf16[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(add_85, torch.bfloat16);  add_85 = None
        convert_element_type_294: "bf16[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(add_86, torch.bfloat16);  add_86 = None
        
         # File: C:\Users\arroy\anaconda3\envs\hrm_v1\lib\site-packages\flash_attn\flash_attn_interface.py:839 in forward, code: out_padded, softmax_lse, S_dmask, rng_state = _wrapped_flash_attn_forward(
        _flash_attn_forward_13 = torch.ops.flash_attn._flash_attn_forward.default(convert_element_type_293, convert_element_type_294, slice_178, 0.0, 0.125, False, -1, -1, 0.0, None, False);  convert_element_type_293 = convert_element_type_294 = slice_178 = None
        getitem_78: "bf16[128, 901, 2, 64]" = _flash_attn_forward_13[0];  _flash_attn_forward_13 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:135 in forward, code: attn_output = attn_output.view(batch_size, seq_len, self.output_size)  # type: ignore
        view_139: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(getitem_78, [128, 901, 128]);  getitem_78 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_140: "bf16[115328, 128]" = torch.ops.aten.reshape.default(view_139, [115328, 128]);  view_139 = None
        mm_53: "bf16[115328, 128]" = torch.ops.aten.mm.default(view_140, permute_5);  view_140 = None
        view_141: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(mm_53, [128, 901, 128]);  mm_53 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:80 in forward, code: hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        add_87: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(convert_element_type_287, view_141);  convert_element_type_287 = view_141 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:154 in rms_norm, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_298: "f32[128, 901, 128]" = torch.ops.prims.convert_element_type.default(add_87, torch.float32);  add_87 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:156 in rms_norm, code: variance = hidden_states.square().mean(-1, keepdim=True)
        pow_27: "f32[128, 901, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_298, 2)
        mean_26: "f32[128, 901, 1]" = torch.ops.aten.mean.dim(pow_27, [-1], True);  pow_27 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:157 in rms_norm, code: hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        add_88: "f32[128, 901, 1]" = torch.ops.aten.add.Tensor(mean_26, 1e-05);  mean_26 = None
        rsqrt_26: "f32[128, 901, 1]" = torch.ops.aten.rsqrt.default(add_88);  add_88 = None
        mul_109: "f32[128, 901, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_298, rsqrt_26);  convert_element_type_298 = rsqrt_26 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:158 in rms_norm, code: return hidden_states.to(input_dtype)
        convert_element_type_299: "bf16[128, 901, 128]" = torch.ops.prims.convert_element_type.default(mul_109, torch.bfloat16);  mul_109 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_142: "bf16[115328, 128]" = torch.ops.aten.reshape.default(convert_element_type_299, [115328, 128])
        mm_54: "bf16[115328, 1024]" = torch.ops.aten.mm.default(view_142, permute_6);  view_142 = None
        view_143: "bf16[128, 901, 1024]" = torch.ops.aten.reshape.default(mm_54, [128, 901, 1024]);  mm_54 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:148 in forward, code: gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        split_13 = torch.ops.aten.split.Tensor(view_143, 512, -1);  view_143 = None
        getitem_82: "bf16[128, 901, 512]" = split_13[0]
        getitem_83: "bf16[128, 901, 512]" = split_13[1];  split_13 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:149 in forward, code: return self.down_proj(F.silu(gate) * up)
        convert_element_type_303: "f32[128, 901, 512]" = torch.ops.prims.convert_element_type.default(getitem_82, torch.float32);  getitem_82 = None
        sigmoid_13: "f32[128, 901, 512]" = torch.ops.aten.sigmoid.default(convert_element_type_303)
        mul_110: "f32[128, 901, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_303, sigmoid_13);  convert_element_type_303 = sigmoid_13 = None
        convert_element_type_304: "bf16[128, 901, 512]" = torch.ops.prims.convert_element_type.default(mul_110, torch.bfloat16);  mul_110 = None
        mul_111: "bf16[128, 901, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_304, getitem_83);  convert_element_type_304 = getitem_83 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_144: "bf16[115328, 512]" = torch.ops.aten.reshape.default(mul_111, [115328, 512]);  mul_111 = None
        mm_55: "bf16[115328, 128]" = torch.ops.aten.mm.default(view_144, permute_7);  view_144 = None
        view_145: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(mm_55, [128, 901, 128]);  mm_55 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:82 in forward, code: hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)
        add_89: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(convert_element_type_299, view_145);  convert_element_type_299 = view_145 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:154 in rms_norm, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_308: "f32[128, 901, 128]" = torch.ops.prims.convert_element_type.default(add_89, torch.float32);  add_89 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:156 in rms_norm, code: variance = hidden_states.square().mean(-1, keepdim=True)
        pow_28: "f32[128, 901, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_308, 2)
        mean_27: "f32[128, 901, 1]" = torch.ops.aten.mean.dim(pow_28, [-1], True);  pow_28 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:157 in rms_norm, code: hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        add_90: "f32[128, 901, 1]" = torch.ops.aten.add.Tensor(mean_27, 1e-05);  mean_27 = None
        rsqrt_27: "f32[128, 901, 1]" = torch.ops.aten.rsqrt.default(add_90);  add_90 = None
        mul_112: "f32[128, 901, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_308, rsqrt_27);  convert_element_type_308 = rsqrt_27 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:158 in rms_norm, code: return hidden_states.to(input_dtype)
        convert_element_type_309: "bf16[128, 901, 128]" = torch.ops.prims.convert_element_type.default(mul_112, torch.bfloat16);  mul_112 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_146: "bf16[115328, 128]" = torch.ops.aten.reshape.default(convert_element_type_309, [115328, 128])
        mm_56: "bf16[115328, 384]" = torch.ops.aten.mm.default(view_146, permute_8);  view_146 = None
        view_147: "bf16[128, 901, 384]" = torch.ops.aten.reshape.default(mm_56, [128, 901, 384]);  mm_56 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:119 in forward, code: qkv = qkv.view(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        view_148: "bf16[128, 901, 6, 64]" = torch.ops.aten.reshape.default(view_147, [128, 901, 6, 64]);  view_147 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:120 in forward, code: query = qkv[:, :, :self.num_heads]
        slice_185: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_148, 2, 0, 2)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:121 in forward, code: key = qkv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
        slice_188: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_148, 2, 2, 4)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:122 in forward, code: value = qkv[:, :, self.num_heads + self.num_key_value_heads:]
        slice_191: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_148, 2, 4, 9223372036854775807);  view_148 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:34 in apply_rotary_pos_emb, code: q = q.to(cos.dtype)
        convert_element_type_313: "f32[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(slice_185, torch.float32);  slice_185 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:35 in apply_rotary_pos_emb, code: k = k.to(cos.dtype)
        convert_element_type_314: "f32[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(slice_188, torch.float32);  slice_188 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:37 in apply_rotary_pos_emb, code: q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
        mul_113: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_313, unsqueeze)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:25 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_192: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_313, 3, 0, 32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:26 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_193: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_313, 3, 32, 9223372036854775807);  convert_element_type_313 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:27 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_28: "f32[128, 901, 2, 32]" = torch.ops.aten.neg.default(slice_193);  slice_193 = None
        cat_29: "f32[128, 901, 2, 64]" = torch.ops.aten.cat.default([neg_28, slice_192], -1);  neg_28 = slice_192 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:37 in apply_rotary_pos_emb, code: q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
        mul_114: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(cat_29, unsqueeze_1);  cat_29 = None
        add_91: "f32[128, 901, 2, 64]" = torch.ops.aten.add.Tensor(mul_113, mul_114);  mul_113 = mul_114 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:38 in apply_rotary_pos_emb, code: k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
        mul_115: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_314, unsqueeze)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:25 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_194: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_314, 3, 0, 32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:26 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_195: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_314, 3, 32, 9223372036854775807);  convert_element_type_314 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:27 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_29: "f32[128, 901, 2, 32]" = torch.ops.aten.neg.default(slice_195);  slice_195 = None
        cat_30: "f32[128, 901, 2, 64]" = torch.ops.aten.cat.default([neg_29, slice_194], -1);  neg_29 = slice_194 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:38 in apply_rotary_pos_emb, code: k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
        mul_116: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(cat_30, unsqueeze_1);  cat_30 = None
        add_92: "f32[128, 901, 2, 64]" = torch.ops.aten.add.Tensor(mul_115, mul_116);  mul_115 = mul_116 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:40 in apply_rotary_pos_emb, code: return q_embed.to(orig_dtype), k_embed.to(orig_dtype)
        convert_element_type_315: "bf16[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(add_91, torch.bfloat16);  add_91 = None
        convert_element_type_316: "bf16[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(add_92, torch.bfloat16);  add_92 = None
        
         # File: C:\Users\arroy\anaconda3\envs\hrm_v1\lib\site-packages\flash_attn\flash_attn_interface.py:839 in forward, code: out_padded, softmax_lse, S_dmask, rng_state = _wrapped_flash_attn_forward(
        _flash_attn_forward_14 = torch.ops.flash_attn._flash_attn_forward.default(convert_element_type_315, convert_element_type_316, slice_191, 0.0, 0.125, False, -1, -1, 0.0, None, False);  convert_element_type_315 = convert_element_type_316 = slice_191 = None
        getitem_84: "bf16[128, 901, 2, 64]" = _flash_attn_forward_14[0];  _flash_attn_forward_14 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:135 in forward, code: attn_output = attn_output.view(batch_size, seq_len, self.output_size)  # type: ignore
        view_149: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(getitem_84, [128, 901, 128]);  getitem_84 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_150: "bf16[115328, 128]" = torch.ops.aten.reshape.default(view_149, [115328, 128]);  view_149 = None
        mm_57: "bf16[115328, 128]" = torch.ops.aten.mm.default(view_150, permute_9);  view_150 = None
        view_151: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(mm_57, [128, 901, 128]);  mm_57 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:80 in forward, code: hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        add_93: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(convert_element_type_309, view_151);  convert_element_type_309 = view_151 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:154 in rms_norm, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_320: "f32[128, 901, 128]" = torch.ops.prims.convert_element_type.default(add_93, torch.float32);  add_93 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:156 in rms_norm, code: variance = hidden_states.square().mean(-1, keepdim=True)
        pow_29: "f32[128, 901, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_320, 2)
        mean_28: "f32[128, 901, 1]" = torch.ops.aten.mean.dim(pow_29, [-1], True);  pow_29 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:157 in rms_norm, code: hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        add_94: "f32[128, 901, 1]" = torch.ops.aten.add.Tensor(mean_28, 1e-05);  mean_28 = None
        rsqrt_28: "f32[128, 901, 1]" = torch.ops.aten.rsqrt.default(add_94);  add_94 = None
        mul_117: "f32[128, 901, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_320, rsqrt_28);  convert_element_type_320 = rsqrt_28 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:158 in rms_norm, code: return hidden_states.to(input_dtype)
        convert_element_type_321: "bf16[128, 901, 128]" = torch.ops.prims.convert_element_type.default(mul_117, torch.bfloat16);  mul_117 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_152: "bf16[115328, 128]" = torch.ops.aten.reshape.default(convert_element_type_321, [115328, 128])
        mm_58: "bf16[115328, 1024]" = torch.ops.aten.mm.default(view_152, permute_10);  view_152 = None
        view_153: "bf16[128, 901, 1024]" = torch.ops.aten.reshape.default(mm_58, [128, 901, 1024]);  mm_58 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:148 in forward, code: gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        split_14 = torch.ops.aten.split.Tensor(view_153, 512, -1);  view_153 = None
        getitem_88: "bf16[128, 901, 512]" = split_14[0]
        getitem_89: "bf16[128, 901, 512]" = split_14[1];  split_14 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:149 in forward, code: return self.down_proj(F.silu(gate) * up)
        convert_element_type_325: "f32[128, 901, 512]" = torch.ops.prims.convert_element_type.default(getitem_88, torch.float32);  getitem_88 = None
        sigmoid_14: "f32[128, 901, 512]" = torch.ops.aten.sigmoid.default(convert_element_type_325)
        mul_118: "f32[128, 901, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_325, sigmoid_14);  convert_element_type_325 = sigmoid_14 = None
        convert_element_type_326: "bf16[128, 901, 512]" = torch.ops.prims.convert_element_type.default(mul_118, torch.bfloat16);  mul_118 = None
        mul_119: "bf16[128, 901, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_326, getitem_89);  convert_element_type_326 = getitem_89 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_154: "bf16[115328, 512]" = torch.ops.aten.reshape.default(mul_119, [115328, 512]);  mul_119 = None
        mm_59: "bf16[115328, 128]" = torch.ops.aten.mm.default(view_154, permute_11);  view_154 = None
        view_155: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(mm_59, [128, 901, 128]);  mm_59 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:82 in forward, code: hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)
        add_95: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(convert_element_type_321, view_155);  convert_element_type_321 = view_155 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:154 in rms_norm, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_330: "f32[128, 901, 128]" = torch.ops.prims.convert_element_type.default(add_95, torch.float32);  add_95 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:156 in rms_norm, code: variance = hidden_states.square().mean(-1, keepdim=True)
        pow_30: "f32[128, 901, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_330, 2)
        mean_29: "f32[128, 901, 1]" = torch.ops.aten.mean.dim(pow_30, [-1], True);  pow_30 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:157 in rms_norm, code: hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        add_96: "f32[128, 901, 1]" = torch.ops.aten.add.Tensor(mean_29, 1e-05);  mean_29 = None
        rsqrt_29: "f32[128, 901, 1]" = torch.ops.aten.rsqrt.default(add_96);  add_96 = None
        mul_120: "f32[128, 901, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_330, rsqrt_29);  convert_element_type_330 = rsqrt_29 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:158 in rms_norm, code: return hidden_states.to(input_dtype)
        convert_element_type_331: "bf16[128, 901, 128]" = torch.ops.prims.convert_element_type.default(mul_120, torch.bfloat16);  mul_120 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_156: "bf16[115328, 128]" = torch.ops.aten.reshape.default(convert_element_type_331, [115328, 128])
        mm_60: "bf16[115328, 384]" = torch.ops.aten.mm.default(view_156, permute_12);  view_156 = None
        view_157: "bf16[128, 901, 384]" = torch.ops.aten.reshape.default(mm_60, [128, 901, 384]);  mm_60 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:119 in forward, code: qkv = qkv.view(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        view_158: "bf16[128, 901, 6, 64]" = torch.ops.aten.reshape.default(view_157, [128, 901, 6, 64]);  view_157 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:120 in forward, code: query = qkv[:, :, :self.num_heads]
        slice_198: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_158, 2, 0, 2)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:121 in forward, code: key = qkv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
        slice_201: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_158, 2, 2, 4)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:122 in forward, code: value = qkv[:, :, self.num_heads + self.num_key_value_heads:]
        slice_204: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_158, 2, 4, 9223372036854775807);  view_158 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:34 in apply_rotary_pos_emb, code: q = q.to(cos.dtype)
        convert_element_type_335: "f32[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(slice_198, torch.float32);  slice_198 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:35 in apply_rotary_pos_emb, code: k = k.to(cos.dtype)
        convert_element_type_336: "f32[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(slice_201, torch.float32);  slice_201 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:37 in apply_rotary_pos_emb, code: q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
        mul_121: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_335, unsqueeze)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:25 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_205: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_335, 3, 0, 32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:26 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_206: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_335, 3, 32, 9223372036854775807);  convert_element_type_335 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:27 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_30: "f32[128, 901, 2, 32]" = torch.ops.aten.neg.default(slice_206);  slice_206 = None
        cat_31: "f32[128, 901, 2, 64]" = torch.ops.aten.cat.default([neg_30, slice_205], -1);  neg_30 = slice_205 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:37 in apply_rotary_pos_emb, code: q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
        mul_122: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(cat_31, unsqueeze_1);  cat_31 = None
        add_97: "f32[128, 901, 2, 64]" = torch.ops.aten.add.Tensor(mul_121, mul_122);  mul_121 = mul_122 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:38 in apply_rotary_pos_emb, code: k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
        mul_123: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_336, unsqueeze)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:25 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_207: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_336, 3, 0, 32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:26 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_208: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_336, 3, 32, 9223372036854775807);  convert_element_type_336 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:27 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_31: "f32[128, 901, 2, 32]" = torch.ops.aten.neg.default(slice_208);  slice_208 = None
        cat_32: "f32[128, 901, 2, 64]" = torch.ops.aten.cat.default([neg_31, slice_207], -1);  neg_31 = slice_207 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:38 in apply_rotary_pos_emb, code: k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
        mul_124: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(cat_32, unsqueeze_1);  cat_32 = None
        add_98: "f32[128, 901, 2, 64]" = torch.ops.aten.add.Tensor(mul_123, mul_124);  mul_123 = mul_124 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:40 in apply_rotary_pos_emb, code: return q_embed.to(orig_dtype), k_embed.to(orig_dtype)
        convert_element_type_337: "bf16[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(add_97, torch.bfloat16);  add_97 = None
        convert_element_type_338: "bf16[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(add_98, torch.bfloat16);  add_98 = None
        
         # File: C:\Users\arroy\anaconda3\envs\hrm_v1\lib\site-packages\flash_attn\flash_attn_interface.py:839 in forward, code: out_padded, softmax_lse, S_dmask, rng_state = _wrapped_flash_attn_forward(
        _flash_attn_forward_15 = torch.ops.flash_attn._flash_attn_forward.default(convert_element_type_337, convert_element_type_338, slice_204, 0.0, 0.125, False, -1, -1, 0.0, None, False);  convert_element_type_337 = convert_element_type_338 = slice_204 = None
        getitem_90: "bf16[128, 901, 2, 64]" = _flash_attn_forward_15[0];  _flash_attn_forward_15 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:135 in forward, code: attn_output = attn_output.view(batch_size, seq_len, self.output_size)  # type: ignore
        view_159: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(getitem_90, [128, 901, 128]);  getitem_90 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_160: "bf16[115328, 128]" = torch.ops.aten.reshape.default(view_159, [115328, 128]);  view_159 = None
        mm_61: "bf16[115328, 128]" = torch.ops.aten.mm.default(view_160, permute_13);  view_160 = None
        view_161: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(mm_61, [128, 901, 128]);  mm_61 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:80 in forward, code: hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        add_99: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(convert_element_type_331, view_161);  convert_element_type_331 = view_161 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:154 in rms_norm, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_342: "f32[128, 901, 128]" = torch.ops.prims.convert_element_type.default(add_99, torch.float32);  add_99 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:156 in rms_norm, code: variance = hidden_states.square().mean(-1, keepdim=True)
        pow_31: "f32[128, 901, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_342, 2)
        mean_30: "f32[128, 901, 1]" = torch.ops.aten.mean.dim(pow_31, [-1], True);  pow_31 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:157 in rms_norm, code: hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        add_100: "f32[128, 901, 1]" = torch.ops.aten.add.Tensor(mean_30, 1e-05);  mean_30 = None
        rsqrt_30: "f32[128, 901, 1]" = torch.ops.aten.rsqrt.default(add_100);  add_100 = None
        mul_125: "f32[128, 901, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_342, rsqrt_30);  convert_element_type_342 = rsqrt_30 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:158 in rms_norm, code: return hidden_states.to(input_dtype)
        convert_element_type_343: "bf16[128, 901, 128]" = torch.ops.prims.convert_element_type.default(mul_125, torch.bfloat16);  mul_125 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_162: "bf16[115328, 128]" = torch.ops.aten.reshape.default(convert_element_type_343, [115328, 128])
        mm_62: "bf16[115328, 1024]" = torch.ops.aten.mm.default(view_162, permute_14);  view_162 = None
        view_163: "bf16[128, 901, 1024]" = torch.ops.aten.reshape.default(mm_62, [128, 901, 1024]);  mm_62 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:148 in forward, code: gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        split_15 = torch.ops.aten.split.Tensor(view_163, 512, -1);  view_163 = None
        getitem_94: "bf16[128, 901, 512]" = split_15[0]
        getitem_95: "bf16[128, 901, 512]" = split_15[1];  split_15 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:149 in forward, code: return self.down_proj(F.silu(gate) * up)
        convert_element_type_347: "f32[128, 901, 512]" = torch.ops.prims.convert_element_type.default(getitem_94, torch.float32);  getitem_94 = None
        sigmoid_15: "f32[128, 901, 512]" = torch.ops.aten.sigmoid.default(convert_element_type_347)
        mul_126: "f32[128, 901, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_347, sigmoid_15);  convert_element_type_347 = sigmoid_15 = None
        convert_element_type_348: "bf16[128, 901, 512]" = torch.ops.prims.convert_element_type.default(mul_126, torch.bfloat16);  mul_126 = None
        mul_127: "bf16[128, 901, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_348, getitem_95);  convert_element_type_348 = getitem_95 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_164: "bf16[115328, 512]" = torch.ops.aten.reshape.default(mul_127, [115328, 512]);  mul_127 = None
        mm_63: "bf16[115328, 128]" = torch.ops.aten.mm.default(view_164, permute_15);  view_164 = None
        view_165: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(mm_63, [128, 901, 128]);  mm_63 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:82 in forward, code: hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)
        add_101: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(convert_element_type_343, view_165);  convert_element_type_343 = view_165 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:154 in rms_norm, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_352: "f32[128, 901, 128]" = torch.ops.prims.convert_element_type.default(add_101, torch.float32);  add_101 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:156 in rms_norm, code: variance = hidden_states.square().mean(-1, keepdim=True)
        pow_32: "f32[128, 901, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_352, 2)
        mean_31: "f32[128, 901, 1]" = torch.ops.aten.mean.dim(pow_32, [-1], True);  pow_32 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:157 in rms_norm, code: hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        add_102: "f32[128, 901, 1]" = torch.ops.aten.add.Tensor(mean_31, 1e-05);  mean_31 = None
        rsqrt_31: "f32[128, 901, 1]" = torch.ops.aten.rsqrt.default(add_102);  add_102 = None
        mul_128: "f32[128, 901, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_352, rsqrt_31);  convert_element_type_352 = rsqrt_31 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:158 in rms_norm, code: return hidden_states.to(input_dtype)
        convert_element_type_353: "bf16[128, 901, 128]" = torch.ops.prims.convert_element_type.default(mul_128, torch.bfloat16);  mul_128 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:94 in forward, code: hidden_states = hidden_states + input_injection
        add_104: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(convert_element_type_353, add_77);  convert_element_type_353 = add_77 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_166: "bf16[115328, 128]" = torch.ops.aten.reshape.default(add_104, [115328, 128])
        mm_64: "bf16[115328, 384]" = torch.ops.aten.mm.default(view_166, permute)
        view_167: "bf16[128, 901, 384]" = torch.ops.aten.reshape.default(mm_64, [128, 901, 384]);  mm_64 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:119 in forward, code: qkv = qkv.view(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        view_168: "bf16[128, 901, 6, 64]" = torch.ops.aten.reshape.default(view_167, [128, 901, 6, 64]);  view_167 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:120 in forward, code: query = qkv[:, :, :self.num_heads]
        slice_211: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_168, 2, 0, 2)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:121 in forward, code: key = qkv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
        slice_214: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_168, 2, 2, 4)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:122 in forward, code: value = qkv[:, :, self.num_heads + self.num_key_value_heads:]
        slice_217: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_168, 2, 4, 9223372036854775807);  view_168 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:34 in apply_rotary_pos_emb, code: q = q.to(cos.dtype)
        convert_element_type_357: "f32[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(slice_211, torch.float32);  slice_211 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:35 in apply_rotary_pos_emb, code: k = k.to(cos.dtype)
        convert_element_type_358: "f32[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(slice_214, torch.float32);  slice_214 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:37 in apply_rotary_pos_emb, code: q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
        mul_129: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_357, unsqueeze)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:25 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_218: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_357, 3, 0, 32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:26 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_219: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_357, 3, 32, 9223372036854775807);  convert_element_type_357 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:27 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_32: "f32[128, 901, 2, 32]" = torch.ops.aten.neg.default(slice_219);  slice_219 = None
        cat_33: "f32[128, 901, 2, 64]" = torch.ops.aten.cat.default([neg_32, slice_218], -1);  neg_32 = slice_218 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:37 in apply_rotary_pos_emb, code: q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
        mul_130: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(cat_33, unsqueeze_1);  cat_33 = None
        add_105: "f32[128, 901, 2, 64]" = torch.ops.aten.add.Tensor(mul_129, mul_130);  mul_129 = mul_130 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:38 in apply_rotary_pos_emb, code: k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
        mul_131: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_358, unsqueeze)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:25 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_220: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_358, 3, 0, 32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:26 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_221: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_358, 3, 32, 9223372036854775807);  convert_element_type_358 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:27 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_33: "f32[128, 901, 2, 32]" = torch.ops.aten.neg.default(slice_221);  slice_221 = None
        cat_34: "f32[128, 901, 2, 64]" = torch.ops.aten.cat.default([neg_33, slice_220], -1);  neg_33 = slice_220 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:38 in apply_rotary_pos_emb, code: k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
        mul_132: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(cat_34, unsqueeze_1);  cat_34 = None
        add_106: "f32[128, 901, 2, 64]" = torch.ops.aten.add.Tensor(mul_131, mul_132);  mul_131 = mul_132 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:40 in apply_rotary_pos_emb, code: return q_embed.to(orig_dtype), k_embed.to(orig_dtype)
        convert_element_type_359: "bf16[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(add_105, torch.bfloat16);  add_105 = None
        convert_element_type_360: "bf16[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(add_106, torch.bfloat16);  add_106 = None
        
         # File: C:\Users\arroy\anaconda3\envs\hrm_v1\lib\site-packages\flash_attn\flash_attn_interface.py:1201 in flash_attn_func, code: return FlashAttnFunc.apply(
        _flash_attn_forward_16 = torch.ops.flash_attn._flash_attn_forward.default(convert_element_type_359, convert_element_type_360, slice_217, 0.0, 0.125, False, -1, -1, 0.0, None, False)
        getitem_96: "bf16[128, 901, 2, 64]" = _flash_attn_forward_16[0]
        getitem_97: "f32[128, 2, 901]" = _flash_attn_forward_16[1]
        getitem_99: "i64[2]" = _flash_attn_forward_16[3];  _flash_attn_forward_16 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:135 in forward, code: attn_output = attn_output.view(batch_size, seq_len, self.output_size)  # type: ignore
        view_169: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(getitem_96, [128, 901, 128])
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_170: "bf16[115328, 128]" = torch.ops.aten.reshape.default(view_169, [115328, 128]);  view_169 = None
        mm_65: "bf16[115328, 128]" = torch.ops.aten.mm.default(view_170, permute_1);  view_170 = None
        view_171: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(mm_65, [128, 901, 128]);  mm_65 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:80 in forward, code: hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        add_107: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(add_104, view_171);  add_104 = view_171 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:154 in rms_norm, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_364: "f32[128, 901, 128]" = torch.ops.prims.convert_element_type.default(add_107, torch.float32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:156 in rms_norm, code: variance = hidden_states.square().mean(-1, keepdim=True)
        pow_33: "f32[128, 901, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_364, 2)
        mean_32: "f32[128, 901, 1]" = torch.ops.aten.mean.dim(pow_33, [-1], True);  pow_33 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:157 in rms_norm, code: hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        add_108: "f32[128, 901, 1]" = torch.ops.aten.add.Tensor(mean_32, 1e-05);  mean_32 = None
        rsqrt_32: "f32[128, 901, 1]" = torch.ops.aten.rsqrt.default(add_108);  add_108 = None
        mul_133: "f32[128, 901, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_364, rsqrt_32);  convert_element_type_364 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:158 in rms_norm, code: return hidden_states.to(input_dtype)
        convert_element_type_365: "bf16[128, 901, 128]" = torch.ops.prims.convert_element_type.default(mul_133, torch.bfloat16);  mul_133 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_172: "bf16[115328, 128]" = torch.ops.aten.reshape.default(convert_element_type_365, [115328, 128])
        mm_66: "bf16[115328, 1024]" = torch.ops.aten.mm.default(view_172, permute_2)
        view_173: "bf16[128, 901, 1024]" = torch.ops.aten.reshape.default(mm_66, [128, 901, 1024]);  mm_66 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:148 in forward, code: gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        split_16 = torch.ops.aten.split.Tensor(view_173, 512, -1);  view_173 = None
        getitem_100: "bf16[128, 901, 512]" = split_16[0]
        getitem_101: "bf16[128, 901, 512]" = split_16[1];  split_16 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:149 in forward, code: return self.down_proj(F.silu(gate) * up)
        convert_element_type_369: "f32[128, 901, 512]" = torch.ops.prims.convert_element_type.default(getitem_100, torch.float32)
        sigmoid_16: "f32[128, 901, 512]" = torch.ops.aten.sigmoid.default(convert_element_type_369)
        mul_134: "f32[128, 901, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_369, sigmoid_16);  convert_element_type_369 = sigmoid_16 = None
        convert_element_type_370: "bf16[128, 901, 512]" = torch.ops.prims.convert_element_type.default(mul_134, torch.bfloat16);  mul_134 = None
        mul_135: "bf16[128, 901, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_370, getitem_101);  convert_element_type_370 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_174: "bf16[115328, 512]" = torch.ops.aten.reshape.default(mul_135, [115328, 512]);  mul_135 = None
        mm_67: "bf16[115328, 128]" = torch.ops.aten.mm.default(view_174, permute_3)
        view_175: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(mm_67, [128, 901, 128]);  mm_67 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:82 in forward, code: hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)
        add_109: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(convert_element_type_365, view_175);  convert_element_type_365 = view_175 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:154 in rms_norm, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_374: "f32[128, 901, 128]" = torch.ops.prims.convert_element_type.default(add_109, torch.float32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:156 in rms_norm, code: variance = hidden_states.square().mean(-1, keepdim=True)
        pow_34: "f32[128, 901, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_374, 2)
        mean_33: "f32[128, 901, 1]" = torch.ops.aten.mean.dim(pow_34, [-1], True);  pow_34 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:157 in rms_norm, code: hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        add_110: "f32[128, 901, 1]" = torch.ops.aten.add.Tensor(mean_33, 1e-05);  mean_33 = None
        rsqrt_33: "f32[128, 901, 1]" = torch.ops.aten.rsqrt.default(add_110);  add_110 = None
        mul_136: "f32[128, 901, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_374, rsqrt_33);  convert_element_type_374 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:158 in rms_norm, code: return hidden_states.to(input_dtype)
        convert_element_type_375: "bf16[128, 901, 128]" = torch.ops.prims.convert_element_type.default(mul_136, torch.bfloat16);  mul_136 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_176: "bf16[115328, 128]" = torch.ops.aten.reshape.default(convert_element_type_375, [115328, 128])
        mm_68: "bf16[115328, 384]" = torch.ops.aten.mm.default(view_176, permute_4)
        view_177: "bf16[128, 901, 384]" = torch.ops.aten.reshape.default(mm_68, [128, 901, 384]);  mm_68 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:119 in forward, code: qkv = qkv.view(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        view_178: "bf16[128, 901, 6, 64]" = torch.ops.aten.reshape.default(view_177, [128, 901, 6, 64]);  view_177 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:120 in forward, code: query = qkv[:, :, :self.num_heads]
        slice_224: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_178, 2, 0, 2)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:121 in forward, code: key = qkv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
        slice_227: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_178, 2, 2, 4)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:122 in forward, code: value = qkv[:, :, self.num_heads + self.num_key_value_heads:]
        slice_230: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_178, 2, 4, 9223372036854775807);  view_178 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:34 in apply_rotary_pos_emb, code: q = q.to(cos.dtype)
        convert_element_type_379: "f32[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(slice_224, torch.float32);  slice_224 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:35 in apply_rotary_pos_emb, code: k = k.to(cos.dtype)
        convert_element_type_380: "f32[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(slice_227, torch.float32);  slice_227 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:37 in apply_rotary_pos_emb, code: q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
        mul_137: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_379, unsqueeze)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:25 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_231: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_379, 3, 0, 32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:26 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_232: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_379, 3, 32, 9223372036854775807);  convert_element_type_379 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:27 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_34: "f32[128, 901, 2, 32]" = torch.ops.aten.neg.default(slice_232);  slice_232 = None
        cat_35: "f32[128, 901, 2, 64]" = torch.ops.aten.cat.default([neg_34, slice_231], -1);  neg_34 = slice_231 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:37 in apply_rotary_pos_emb, code: q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
        mul_138: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(cat_35, unsqueeze_1);  cat_35 = None
        add_111: "f32[128, 901, 2, 64]" = torch.ops.aten.add.Tensor(mul_137, mul_138);  mul_137 = mul_138 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:38 in apply_rotary_pos_emb, code: k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
        mul_139: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_380, unsqueeze)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:25 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_233: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_380, 3, 0, 32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:26 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_234: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_380, 3, 32, 9223372036854775807);  convert_element_type_380 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:27 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_35: "f32[128, 901, 2, 32]" = torch.ops.aten.neg.default(slice_234);  slice_234 = None
        cat_36: "f32[128, 901, 2, 64]" = torch.ops.aten.cat.default([neg_35, slice_233], -1);  neg_35 = slice_233 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:38 in apply_rotary_pos_emb, code: k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
        mul_140: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(cat_36, unsqueeze_1);  cat_36 = None
        add_112: "f32[128, 901, 2, 64]" = torch.ops.aten.add.Tensor(mul_139, mul_140);  mul_139 = mul_140 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:40 in apply_rotary_pos_emb, code: return q_embed.to(orig_dtype), k_embed.to(orig_dtype)
        convert_element_type_381: "bf16[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(add_111, torch.bfloat16);  add_111 = None
        convert_element_type_382: "bf16[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(add_112, torch.bfloat16);  add_112 = None
        
         # File: C:\Users\arroy\anaconda3\envs\hrm_v1\lib\site-packages\flash_attn\flash_attn_interface.py:1201 in flash_attn_func, code: return FlashAttnFunc.apply(
        _flash_attn_forward_17 = torch.ops.flash_attn._flash_attn_forward.default(convert_element_type_381, convert_element_type_382, slice_230, 0.0, 0.125, False, -1, -1, 0.0, None, False)
        getitem_102: "bf16[128, 901, 2, 64]" = _flash_attn_forward_17[0]
        getitem_103: "f32[128, 2, 901]" = _flash_attn_forward_17[1]
        getitem_105: "i64[2]" = _flash_attn_forward_17[3];  _flash_attn_forward_17 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:135 in forward, code: attn_output = attn_output.view(batch_size, seq_len, self.output_size)  # type: ignore
        view_179: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(getitem_102, [128, 901, 128])
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_180: "bf16[115328, 128]" = torch.ops.aten.reshape.default(view_179, [115328, 128]);  view_179 = None
        mm_69: "bf16[115328, 128]" = torch.ops.aten.mm.default(view_180, permute_5);  view_180 = None
        view_181: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(mm_69, [128, 901, 128]);  mm_69 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:80 in forward, code: hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        add_113: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(convert_element_type_375, view_181);  convert_element_type_375 = view_181 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:154 in rms_norm, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_386: "f32[128, 901, 128]" = torch.ops.prims.convert_element_type.default(add_113, torch.float32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:156 in rms_norm, code: variance = hidden_states.square().mean(-1, keepdim=True)
        pow_35: "f32[128, 901, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_386, 2)
        mean_34: "f32[128, 901, 1]" = torch.ops.aten.mean.dim(pow_35, [-1], True);  pow_35 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:157 in rms_norm, code: hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        add_114: "f32[128, 901, 1]" = torch.ops.aten.add.Tensor(mean_34, 1e-05);  mean_34 = None
        rsqrt_34: "f32[128, 901, 1]" = torch.ops.aten.rsqrt.default(add_114);  add_114 = None
        mul_141: "f32[128, 901, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_386, rsqrt_34);  convert_element_type_386 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:158 in rms_norm, code: return hidden_states.to(input_dtype)
        convert_element_type_387: "bf16[128, 901, 128]" = torch.ops.prims.convert_element_type.default(mul_141, torch.bfloat16);  mul_141 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_182: "bf16[115328, 128]" = torch.ops.aten.reshape.default(convert_element_type_387, [115328, 128])
        mm_70: "bf16[115328, 1024]" = torch.ops.aten.mm.default(view_182, permute_6)
        view_183: "bf16[128, 901, 1024]" = torch.ops.aten.reshape.default(mm_70, [128, 901, 1024]);  mm_70 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:148 in forward, code: gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        split_17 = torch.ops.aten.split.Tensor(view_183, 512, -1);  view_183 = None
        getitem_106: "bf16[128, 901, 512]" = split_17[0]
        getitem_107: "bf16[128, 901, 512]" = split_17[1];  split_17 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:149 in forward, code: return self.down_proj(F.silu(gate) * up)
        convert_element_type_391: "f32[128, 901, 512]" = torch.ops.prims.convert_element_type.default(getitem_106, torch.float32)
        sigmoid_17: "f32[128, 901, 512]" = torch.ops.aten.sigmoid.default(convert_element_type_391)
        mul_142: "f32[128, 901, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_391, sigmoid_17);  convert_element_type_391 = sigmoid_17 = None
        convert_element_type_392: "bf16[128, 901, 512]" = torch.ops.prims.convert_element_type.default(mul_142, torch.bfloat16);  mul_142 = None
        mul_143: "bf16[128, 901, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_392, getitem_107);  convert_element_type_392 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_184: "bf16[115328, 512]" = torch.ops.aten.reshape.default(mul_143, [115328, 512]);  mul_143 = None
        mm_71: "bf16[115328, 128]" = torch.ops.aten.mm.default(view_184, permute_7)
        view_185: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(mm_71, [128, 901, 128]);  mm_71 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:82 in forward, code: hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)
        add_115: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(convert_element_type_387, view_185);  convert_element_type_387 = view_185 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:154 in rms_norm, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_396: "f32[128, 901, 128]" = torch.ops.prims.convert_element_type.default(add_115, torch.float32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:156 in rms_norm, code: variance = hidden_states.square().mean(-1, keepdim=True)
        pow_36: "f32[128, 901, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_396, 2)
        mean_35: "f32[128, 901, 1]" = torch.ops.aten.mean.dim(pow_36, [-1], True);  pow_36 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:157 in rms_norm, code: hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        add_116: "f32[128, 901, 1]" = torch.ops.aten.add.Tensor(mean_35, 1e-05);  mean_35 = None
        rsqrt_35: "f32[128, 901, 1]" = torch.ops.aten.rsqrt.default(add_116);  add_116 = None
        mul_144: "f32[128, 901, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_396, rsqrt_35);  convert_element_type_396 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:158 in rms_norm, code: return hidden_states.to(input_dtype)
        convert_element_type_397: "bf16[128, 901, 128]" = torch.ops.prims.convert_element_type.default(mul_144, torch.bfloat16);  mul_144 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_186: "bf16[115328, 128]" = torch.ops.aten.reshape.default(convert_element_type_397, [115328, 128])
        mm_72: "bf16[115328, 384]" = torch.ops.aten.mm.default(view_186, permute_8)
        view_187: "bf16[128, 901, 384]" = torch.ops.aten.reshape.default(mm_72, [128, 901, 384]);  mm_72 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:119 in forward, code: qkv = qkv.view(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        view_188: "bf16[128, 901, 6, 64]" = torch.ops.aten.reshape.default(view_187, [128, 901, 6, 64]);  view_187 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:120 in forward, code: query = qkv[:, :, :self.num_heads]
        slice_237: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_188, 2, 0, 2)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:121 in forward, code: key = qkv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
        slice_240: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_188, 2, 2, 4)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:122 in forward, code: value = qkv[:, :, self.num_heads + self.num_key_value_heads:]
        slice_243: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_188, 2, 4, 9223372036854775807);  view_188 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:34 in apply_rotary_pos_emb, code: q = q.to(cos.dtype)
        convert_element_type_401: "f32[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(slice_237, torch.float32);  slice_237 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:35 in apply_rotary_pos_emb, code: k = k.to(cos.dtype)
        convert_element_type_402: "f32[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(slice_240, torch.float32);  slice_240 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:37 in apply_rotary_pos_emb, code: q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
        mul_145: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_401, unsqueeze)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:25 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_244: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_401, 3, 0, 32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:26 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_245: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_401, 3, 32, 9223372036854775807);  convert_element_type_401 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:27 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_36: "f32[128, 901, 2, 32]" = torch.ops.aten.neg.default(slice_245);  slice_245 = None
        cat_37: "f32[128, 901, 2, 64]" = torch.ops.aten.cat.default([neg_36, slice_244], -1);  neg_36 = slice_244 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:37 in apply_rotary_pos_emb, code: q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
        mul_146: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(cat_37, unsqueeze_1);  cat_37 = None
        add_117: "f32[128, 901, 2, 64]" = torch.ops.aten.add.Tensor(mul_145, mul_146);  mul_145 = mul_146 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:38 in apply_rotary_pos_emb, code: k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
        mul_147: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_402, unsqueeze)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:25 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_246: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_402, 3, 0, 32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:26 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_247: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_402, 3, 32, 9223372036854775807);  convert_element_type_402 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:27 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_37: "f32[128, 901, 2, 32]" = torch.ops.aten.neg.default(slice_247);  slice_247 = None
        cat_38: "f32[128, 901, 2, 64]" = torch.ops.aten.cat.default([neg_37, slice_246], -1);  neg_37 = slice_246 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:38 in apply_rotary_pos_emb, code: k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
        mul_148: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(cat_38, unsqueeze_1);  cat_38 = None
        add_118: "f32[128, 901, 2, 64]" = torch.ops.aten.add.Tensor(mul_147, mul_148);  mul_147 = mul_148 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:40 in apply_rotary_pos_emb, code: return q_embed.to(orig_dtype), k_embed.to(orig_dtype)
        convert_element_type_403: "bf16[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(add_117, torch.bfloat16);  add_117 = None
        convert_element_type_404: "bf16[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(add_118, torch.bfloat16);  add_118 = None
        
         # File: C:\Users\arroy\anaconda3\envs\hrm_v1\lib\site-packages\flash_attn\flash_attn_interface.py:1201 in flash_attn_func, code: return FlashAttnFunc.apply(
        _flash_attn_forward_18 = torch.ops.flash_attn._flash_attn_forward.default(convert_element_type_403, convert_element_type_404, slice_243, 0.0, 0.125, False, -1, -1, 0.0, None, False)
        getitem_108: "bf16[128, 901, 2, 64]" = _flash_attn_forward_18[0]
        getitem_109: "f32[128, 2, 901]" = _flash_attn_forward_18[1]
        getitem_111: "i64[2]" = _flash_attn_forward_18[3];  _flash_attn_forward_18 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:135 in forward, code: attn_output = attn_output.view(batch_size, seq_len, self.output_size)  # type: ignore
        view_189: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(getitem_108, [128, 901, 128])
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_190: "bf16[115328, 128]" = torch.ops.aten.reshape.default(view_189, [115328, 128]);  view_189 = None
        mm_73: "bf16[115328, 128]" = torch.ops.aten.mm.default(view_190, permute_9);  view_190 = None
        view_191: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(mm_73, [128, 901, 128]);  mm_73 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:80 in forward, code: hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        add_119: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(convert_element_type_397, view_191);  convert_element_type_397 = view_191 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:154 in rms_norm, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_408: "f32[128, 901, 128]" = torch.ops.prims.convert_element_type.default(add_119, torch.float32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:156 in rms_norm, code: variance = hidden_states.square().mean(-1, keepdim=True)
        pow_37: "f32[128, 901, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_408, 2)
        mean_36: "f32[128, 901, 1]" = torch.ops.aten.mean.dim(pow_37, [-1], True);  pow_37 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:157 in rms_norm, code: hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        add_120: "f32[128, 901, 1]" = torch.ops.aten.add.Tensor(mean_36, 1e-05);  mean_36 = None
        rsqrt_36: "f32[128, 901, 1]" = torch.ops.aten.rsqrt.default(add_120);  add_120 = None
        mul_149: "f32[128, 901, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_408, rsqrt_36);  convert_element_type_408 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:158 in rms_norm, code: return hidden_states.to(input_dtype)
        convert_element_type_409: "bf16[128, 901, 128]" = torch.ops.prims.convert_element_type.default(mul_149, torch.bfloat16);  mul_149 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_192: "bf16[115328, 128]" = torch.ops.aten.reshape.default(convert_element_type_409, [115328, 128])
        mm_74: "bf16[115328, 1024]" = torch.ops.aten.mm.default(view_192, permute_10)
        view_193: "bf16[128, 901, 1024]" = torch.ops.aten.reshape.default(mm_74, [128, 901, 1024]);  mm_74 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:148 in forward, code: gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        split_18 = torch.ops.aten.split.Tensor(view_193, 512, -1);  view_193 = None
        getitem_112: "bf16[128, 901, 512]" = split_18[0]
        getitem_113: "bf16[128, 901, 512]" = split_18[1];  split_18 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:149 in forward, code: return self.down_proj(F.silu(gate) * up)
        convert_element_type_413: "f32[128, 901, 512]" = torch.ops.prims.convert_element_type.default(getitem_112, torch.float32)
        sigmoid_18: "f32[128, 901, 512]" = torch.ops.aten.sigmoid.default(convert_element_type_413)
        mul_150: "f32[128, 901, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_413, sigmoid_18);  convert_element_type_413 = sigmoid_18 = None
        convert_element_type_414: "bf16[128, 901, 512]" = torch.ops.prims.convert_element_type.default(mul_150, torch.bfloat16);  mul_150 = None
        mul_151: "bf16[128, 901, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_414, getitem_113);  convert_element_type_414 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_194: "bf16[115328, 512]" = torch.ops.aten.reshape.default(mul_151, [115328, 512]);  mul_151 = None
        mm_75: "bf16[115328, 128]" = torch.ops.aten.mm.default(view_194, permute_11)
        view_195: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(mm_75, [128, 901, 128]);  mm_75 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:82 in forward, code: hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)
        add_121: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(convert_element_type_409, view_195);  convert_element_type_409 = view_195 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:154 in rms_norm, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_418: "f32[128, 901, 128]" = torch.ops.prims.convert_element_type.default(add_121, torch.float32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:156 in rms_norm, code: variance = hidden_states.square().mean(-1, keepdim=True)
        pow_38: "f32[128, 901, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_418, 2)
        mean_37: "f32[128, 901, 1]" = torch.ops.aten.mean.dim(pow_38, [-1], True);  pow_38 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:157 in rms_norm, code: hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        add_122: "f32[128, 901, 1]" = torch.ops.aten.add.Tensor(mean_37, 1e-05);  mean_37 = None
        rsqrt_37: "f32[128, 901, 1]" = torch.ops.aten.rsqrt.default(add_122);  add_122 = None
        mul_152: "f32[128, 901, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_418, rsqrt_37);  convert_element_type_418 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:158 in rms_norm, code: return hidden_states.to(input_dtype)
        convert_element_type_419: "bf16[128, 901, 128]" = torch.ops.prims.convert_element_type.default(mul_152, torch.bfloat16);  mul_152 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_196: "bf16[115328, 128]" = torch.ops.aten.reshape.default(convert_element_type_419, [115328, 128])
        mm_76: "bf16[115328, 384]" = torch.ops.aten.mm.default(view_196, permute_12)
        view_197: "bf16[128, 901, 384]" = torch.ops.aten.reshape.default(mm_76, [128, 901, 384]);  mm_76 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:119 in forward, code: qkv = qkv.view(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        view_198: "bf16[128, 901, 6, 64]" = torch.ops.aten.reshape.default(view_197, [128, 901, 6, 64]);  view_197 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:120 in forward, code: query = qkv[:, :, :self.num_heads]
        slice_250: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_198, 2, 0, 2)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:121 in forward, code: key = qkv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
        slice_253: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_198, 2, 2, 4)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:122 in forward, code: value = qkv[:, :, self.num_heads + self.num_key_value_heads:]
        slice_256: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_198, 2, 4, 9223372036854775807);  view_198 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:34 in apply_rotary_pos_emb, code: q = q.to(cos.dtype)
        convert_element_type_423: "f32[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(slice_250, torch.float32);  slice_250 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:35 in apply_rotary_pos_emb, code: k = k.to(cos.dtype)
        convert_element_type_424: "f32[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(slice_253, torch.float32);  slice_253 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:37 in apply_rotary_pos_emb, code: q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
        mul_153: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_423, unsqueeze)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:25 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_257: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_423, 3, 0, 32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:26 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_258: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_423, 3, 32, 9223372036854775807);  convert_element_type_423 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:27 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_38: "f32[128, 901, 2, 32]" = torch.ops.aten.neg.default(slice_258);  slice_258 = None
        cat_39: "f32[128, 901, 2, 64]" = torch.ops.aten.cat.default([neg_38, slice_257], -1);  neg_38 = slice_257 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:37 in apply_rotary_pos_emb, code: q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
        mul_154: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(cat_39, unsqueeze_1);  cat_39 = None
        add_123: "f32[128, 901, 2, 64]" = torch.ops.aten.add.Tensor(mul_153, mul_154);  mul_153 = mul_154 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:38 in apply_rotary_pos_emb, code: k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
        mul_155: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_424, unsqueeze)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:25 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_259: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_424, 3, 0, 32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:26 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_260: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_424, 3, 32, 9223372036854775807);  convert_element_type_424 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:27 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_39: "f32[128, 901, 2, 32]" = torch.ops.aten.neg.default(slice_260);  slice_260 = None
        cat_40: "f32[128, 901, 2, 64]" = torch.ops.aten.cat.default([neg_39, slice_259], -1);  neg_39 = slice_259 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:38 in apply_rotary_pos_emb, code: k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
        mul_156: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(cat_40, unsqueeze_1);  cat_40 = None
        add_124: "f32[128, 901, 2, 64]" = torch.ops.aten.add.Tensor(mul_155, mul_156);  mul_155 = mul_156 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:40 in apply_rotary_pos_emb, code: return q_embed.to(orig_dtype), k_embed.to(orig_dtype)
        convert_element_type_425: "bf16[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(add_123, torch.bfloat16);  add_123 = None
        convert_element_type_426: "bf16[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(add_124, torch.bfloat16);  add_124 = None
        
         # File: C:\Users\arroy\anaconda3\envs\hrm_v1\lib\site-packages\flash_attn\flash_attn_interface.py:1201 in flash_attn_func, code: return FlashAttnFunc.apply(
        _flash_attn_forward_19 = torch.ops.flash_attn._flash_attn_forward.default(convert_element_type_425, convert_element_type_426, slice_256, 0.0, 0.125, False, -1, -1, 0.0, None, False)
        getitem_114: "bf16[128, 901, 2, 64]" = _flash_attn_forward_19[0]
        getitem_115: "f32[128, 2, 901]" = _flash_attn_forward_19[1]
        getitem_117: "i64[2]" = _flash_attn_forward_19[3];  _flash_attn_forward_19 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:135 in forward, code: attn_output = attn_output.view(batch_size, seq_len, self.output_size)  # type: ignore
        view_199: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(getitem_114, [128, 901, 128])
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_200: "bf16[115328, 128]" = torch.ops.aten.reshape.default(view_199, [115328, 128]);  view_199 = None
        mm_77: "bf16[115328, 128]" = torch.ops.aten.mm.default(view_200, permute_13);  view_200 = None
        view_201: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(mm_77, [128, 901, 128]);  mm_77 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:80 in forward, code: hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        add_125: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(convert_element_type_419, view_201);  convert_element_type_419 = view_201 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:154 in rms_norm, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_430: "f32[128, 901, 128]" = torch.ops.prims.convert_element_type.default(add_125, torch.float32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:156 in rms_norm, code: variance = hidden_states.square().mean(-1, keepdim=True)
        pow_39: "f32[128, 901, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_430, 2)
        mean_38: "f32[128, 901, 1]" = torch.ops.aten.mean.dim(pow_39, [-1], True);  pow_39 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:157 in rms_norm, code: hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        add_126: "f32[128, 901, 1]" = torch.ops.aten.add.Tensor(mean_38, 1e-05);  mean_38 = None
        rsqrt_38: "f32[128, 901, 1]" = torch.ops.aten.rsqrt.default(add_126);  add_126 = None
        mul_157: "f32[128, 901, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_430, rsqrt_38);  convert_element_type_430 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:158 in rms_norm, code: return hidden_states.to(input_dtype)
        convert_element_type_431: "bf16[128, 901, 128]" = torch.ops.prims.convert_element_type.default(mul_157, torch.bfloat16);  mul_157 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_202: "bf16[115328, 128]" = torch.ops.aten.reshape.default(convert_element_type_431, [115328, 128])
        mm_78: "bf16[115328, 1024]" = torch.ops.aten.mm.default(view_202, permute_14)
        view_203: "bf16[128, 901, 1024]" = torch.ops.aten.reshape.default(mm_78, [128, 901, 1024]);  mm_78 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:148 in forward, code: gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        split_19 = torch.ops.aten.split.Tensor(view_203, 512, -1);  view_203 = None
        getitem_118: "bf16[128, 901, 512]" = split_19[0]
        getitem_119: "bf16[128, 901, 512]" = split_19[1];  split_19 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:149 in forward, code: return self.down_proj(F.silu(gate) * up)
        convert_element_type_435: "f32[128, 901, 512]" = torch.ops.prims.convert_element_type.default(getitem_118, torch.float32)
        sigmoid_19: "f32[128, 901, 512]" = torch.ops.aten.sigmoid.default(convert_element_type_435)
        mul_158: "f32[128, 901, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_435, sigmoid_19);  convert_element_type_435 = sigmoid_19 = None
        convert_element_type_436: "bf16[128, 901, 512]" = torch.ops.prims.convert_element_type.default(mul_158, torch.bfloat16);  mul_158 = None
        mul_159: "bf16[128, 901, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_436, getitem_119);  convert_element_type_436 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_204: "bf16[115328, 512]" = torch.ops.aten.reshape.default(mul_159, [115328, 512]);  mul_159 = None
        mm_79: "bf16[115328, 128]" = torch.ops.aten.mm.default(view_204, permute_15)
        view_205: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(mm_79, [128, 901, 128]);  mm_79 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:82 in forward, code: hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)
        add_127: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(convert_element_type_431, view_205);  convert_element_type_431 = view_205 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:154 in rms_norm, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_440: "f32[128, 901, 128]" = torch.ops.prims.convert_element_type.default(add_127, torch.float32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:156 in rms_norm, code: variance = hidden_states.square().mean(-1, keepdim=True)
        pow_40: "f32[128, 901, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_440, 2)
        mean_39: "f32[128, 901, 1]" = torch.ops.aten.mean.dim(pow_40, [-1], True);  pow_40 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:157 in rms_norm, code: hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        add_128: "f32[128, 901, 1]" = torch.ops.aten.add.Tensor(mean_39, 1e-05);  mean_39 = None
        rsqrt_39: "f32[128, 901, 1]" = torch.ops.aten.rsqrt.default(add_128);  add_128 = None
        mul_160: "f32[128, 901, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_440, rsqrt_39);  convert_element_type_440 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:158 in rms_norm, code: return hidden_states.to(input_dtype)
        convert_element_type_441: "bf16[128, 901, 128]" = torch.ops.prims.convert_element_type.default(mul_160, torch.bfloat16);  mul_160 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:94 in forward, code: hidden_states = hidden_states + input_injection
        add_129: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(convert_element_type_265, convert_element_type_441);  convert_element_type_265 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_206: "bf16[115328, 128]" = torch.ops.aten.reshape.default(add_129, [115328, 128])
        mm_80: "bf16[115328, 384]" = torch.ops.aten.mm.default(view_206, permute_32)
        view_207: "bf16[128, 901, 384]" = torch.ops.aten.reshape.default(mm_80, [128, 901, 384]);  mm_80 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:119 in forward, code: qkv = qkv.view(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        view_208: "bf16[128, 901, 6, 64]" = torch.ops.aten.reshape.default(view_207, [128, 901, 6, 64]);  view_207 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:120 in forward, code: query = qkv[:, :, :self.num_heads]
        slice_263: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_208, 2, 0, 2)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:121 in forward, code: key = qkv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
        slice_266: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_208, 2, 2, 4)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:122 in forward, code: value = qkv[:, :, self.num_heads + self.num_key_value_heads:]
        slice_269: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_208, 2, 4, 9223372036854775807);  view_208 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:34 in apply_rotary_pos_emb, code: q = q.to(cos.dtype)
        convert_element_type_445: "f32[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(slice_263, torch.float32);  slice_263 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:35 in apply_rotary_pos_emb, code: k = k.to(cos.dtype)
        convert_element_type_446: "f32[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(slice_266, torch.float32);  slice_266 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:37 in apply_rotary_pos_emb, code: q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
        mul_161: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_445, unsqueeze)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:25 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_270: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_445, 3, 0, 32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:26 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_271: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_445, 3, 32, 9223372036854775807);  convert_element_type_445 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:27 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_40: "f32[128, 901, 2, 32]" = torch.ops.aten.neg.default(slice_271);  slice_271 = None
        cat_41: "f32[128, 901, 2, 64]" = torch.ops.aten.cat.default([neg_40, slice_270], -1);  neg_40 = slice_270 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:37 in apply_rotary_pos_emb, code: q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
        mul_162: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(cat_41, unsqueeze_1);  cat_41 = None
        add_130: "f32[128, 901, 2, 64]" = torch.ops.aten.add.Tensor(mul_161, mul_162);  mul_161 = mul_162 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:38 in apply_rotary_pos_emb, code: k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
        mul_163: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_446, unsqueeze)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:25 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_272: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_446, 3, 0, 32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:26 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_273: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_446, 3, 32, 9223372036854775807);  convert_element_type_446 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:27 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_41: "f32[128, 901, 2, 32]" = torch.ops.aten.neg.default(slice_273);  slice_273 = None
        cat_42: "f32[128, 901, 2, 64]" = torch.ops.aten.cat.default([neg_41, slice_272], -1);  neg_41 = slice_272 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:38 in apply_rotary_pos_emb, code: k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
        mul_164: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(cat_42, unsqueeze_1);  cat_42 = None
        add_131: "f32[128, 901, 2, 64]" = torch.ops.aten.add.Tensor(mul_163, mul_164);  mul_163 = mul_164 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:40 in apply_rotary_pos_emb, code: return q_embed.to(orig_dtype), k_embed.to(orig_dtype)
        convert_element_type_447: "bf16[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(add_130, torch.bfloat16);  add_130 = None
        convert_element_type_448: "bf16[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(add_131, torch.bfloat16);  add_131 = None
        
         # File: C:\Users\arroy\anaconda3\envs\hrm_v1\lib\site-packages\flash_attn\flash_attn_interface.py:1201 in flash_attn_func, code: return FlashAttnFunc.apply(
        _flash_attn_forward_20 = torch.ops.flash_attn._flash_attn_forward.default(convert_element_type_447, convert_element_type_448, slice_269, 0.0, 0.125, False, -1, -1, 0.0, None, False)
        getitem_120: "bf16[128, 901, 2, 64]" = _flash_attn_forward_20[0]
        getitem_121: "f32[128, 2, 901]" = _flash_attn_forward_20[1]
        getitem_123: "i64[2]" = _flash_attn_forward_20[3];  _flash_attn_forward_20 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:135 in forward, code: attn_output = attn_output.view(batch_size, seq_len, self.output_size)  # type: ignore
        view_209: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(getitem_120, [128, 901, 128])
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_210: "bf16[115328, 128]" = torch.ops.aten.reshape.default(view_209, [115328, 128]);  view_209 = None
        mm_81: "bf16[115328, 128]" = torch.ops.aten.mm.default(view_210, permute_33);  view_210 = None
        view_211: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(mm_81, [128, 901, 128]);  mm_81 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:80 in forward, code: hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        add_132: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(add_129, view_211);  add_129 = view_211 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:154 in rms_norm, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_452: "f32[128, 901, 128]" = torch.ops.prims.convert_element_type.default(add_132, torch.float32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:156 in rms_norm, code: variance = hidden_states.square().mean(-1, keepdim=True)
        pow_41: "f32[128, 901, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_452, 2)
        mean_40: "f32[128, 901, 1]" = torch.ops.aten.mean.dim(pow_41, [-1], True);  pow_41 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:157 in rms_norm, code: hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        add_133: "f32[128, 901, 1]" = torch.ops.aten.add.Tensor(mean_40, 1e-05);  mean_40 = None
        rsqrt_40: "f32[128, 901, 1]" = torch.ops.aten.rsqrt.default(add_133);  add_133 = None
        mul_165: "f32[128, 901, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_452, rsqrt_40);  convert_element_type_452 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:158 in rms_norm, code: return hidden_states.to(input_dtype)
        convert_element_type_453: "bf16[128, 901, 128]" = torch.ops.prims.convert_element_type.default(mul_165, torch.bfloat16);  mul_165 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_212: "bf16[115328, 128]" = torch.ops.aten.reshape.default(convert_element_type_453, [115328, 128])
        mm_82: "bf16[115328, 1024]" = torch.ops.aten.mm.default(view_212, permute_34)
        view_213: "bf16[128, 901, 1024]" = torch.ops.aten.reshape.default(mm_82, [128, 901, 1024]);  mm_82 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:148 in forward, code: gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        split_20 = torch.ops.aten.split.Tensor(view_213, 512, -1);  view_213 = None
        getitem_124: "bf16[128, 901, 512]" = split_20[0]
        getitem_125: "bf16[128, 901, 512]" = split_20[1];  split_20 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:149 in forward, code: return self.down_proj(F.silu(gate) * up)
        convert_element_type_457: "f32[128, 901, 512]" = torch.ops.prims.convert_element_type.default(getitem_124, torch.float32)
        sigmoid_20: "f32[128, 901, 512]" = torch.ops.aten.sigmoid.default(convert_element_type_457)
        mul_166: "f32[128, 901, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_457, sigmoid_20);  convert_element_type_457 = sigmoid_20 = None
        convert_element_type_458: "bf16[128, 901, 512]" = torch.ops.prims.convert_element_type.default(mul_166, torch.bfloat16);  mul_166 = None
        mul_167: "bf16[128, 901, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_458, getitem_125);  convert_element_type_458 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_214: "bf16[115328, 512]" = torch.ops.aten.reshape.default(mul_167, [115328, 512]);  mul_167 = None
        mm_83: "bf16[115328, 128]" = torch.ops.aten.mm.default(view_214, permute_35)
        view_215: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(mm_83, [128, 901, 128]);  mm_83 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:82 in forward, code: hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)
        add_134: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(convert_element_type_453, view_215);  convert_element_type_453 = view_215 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:154 in rms_norm, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_462: "f32[128, 901, 128]" = torch.ops.prims.convert_element_type.default(add_134, torch.float32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:156 in rms_norm, code: variance = hidden_states.square().mean(-1, keepdim=True)
        pow_42: "f32[128, 901, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_462, 2)
        mean_41: "f32[128, 901, 1]" = torch.ops.aten.mean.dim(pow_42, [-1], True);  pow_42 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:157 in rms_norm, code: hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        add_135: "f32[128, 901, 1]" = torch.ops.aten.add.Tensor(mean_41, 1e-05);  mean_41 = None
        rsqrt_41: "f32[128, 901, 1]" = torch.ops.aten.rsqrt.default(add_135);  add_135 = None
        mul_168: "f32[128, 901, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_462, rsqrt_41);  convert_element_type_462 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:158 in rms_norm, code: return hidden_states.to(input_dtype)
        convert_element_type_463: "bf16[128, 901, 128]" = torch.ops.prims.convert_element_type.default(mul_168, torch.bfloat16);  mul_168 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_216: "bf16[115328, 128]" = torch.ops.aten.reshape.default(convert_element_type_463, [115328, 128])
        mm_84: "bf16[115328, 384]" = torch.ops.aten.mm.default(view_216, permute_36)
        view_217: "bf16[128, 901, 384]" = torch.ops.aten.reshape.default(mm_84, [128, 901, 384]);  mm_84 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:119 in forward, code: qkv = qkv.view(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        view_218: "bf16[128, 901, 6, 64]" = torch.ops.aten.reshape.default(view_217, [128, 901, 6, 64]);  view_217 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:120 in forward, code: query = qkv[:, :, :self.num_heads]
        slice_276: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_218, 2, 0, 2)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:121 in forward, code: key = qkv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
        slice_279: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_218, 2, 2, 4)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:122 in forward, code: value = qkv[:, :, self.num_heads + self.num_key_value_heads:]
        slice_282: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_218, 2, 4, 9223372036854775807);  view_218 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:34 in apply_rotary_pos_emb, code: q = q.to(cos.dtype)
        convert_element_type_467: "f32[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(slice_276, torch.float32);  slice_276 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:35 in apply_rotary_pos_emb, code: k = k.to(cos.dtype)
        convert_element_type_468: "f32[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(slice_279, torch.float32);  slice_279 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:37 in apply_rotary_pos_emb, code: q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
        mul_169: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_467, unsqueeze)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:25 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_283: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_467, 3, 0, 32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:26 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_284: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_467, 3, 32, 9223372036854775807);  convert_element_type_467 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:27 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_42: "f32[128, 901, 2, 32]" = torch.ops.aten.neg.default(slice_284);  slice_284 = None
        cat_43: "f32[128, 901, 2, 64]" = torch.ops.aten.cat.default([neg_42, slice_283], -1);  neg_42 = slice_283 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:37 in apply_rotary_pos_emb, code: q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
        mul_170: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(cat_43, unsqueeze_1);  cat_43 = None
        add_136: "f32[128, 901, 2, 64]" = torch.ops.aten.add.Tensor(mul_169, mul_170);  mul_169 = mul_170 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:38 in apply_rotary_pos_emb, code: k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
        mul_171: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_468, unsqueeze)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:25 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_285: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_468, 3, 0, 32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:26 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_286: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_468, 3, 32, 9223372036854775807);  convert_element_type_468 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:27 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_43: "f32[128, 901, 2, 32]" = torch.ops.aten.neg.default(slice_286);  slice_286 = None
        cat_44: "f32[128, 901, 2, 64]" = torch.ops.aten.cat.default([neg_43, slice_285], -1);  neg_43 = slice_285 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:38 in apply_rotary_pos_emb, code: k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
        mul_172: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(cat_44, unsqueeze_1);  cat_44 = None
        add_137: "f32[128, 901, 2, 64]" = torch.ops.aten.add.Tensor(mul_171, mul_172);  mul_171 = mul_172 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:40 in apply_rotary_pos_emb, code: return q_embed.to(orig_dtype), k_embed.to(orig_dtype)
        convert_element_type_469: "bf16[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(add_136, torch.bfloat16);  add_136 = None
        convert_element_type_470: "bf16[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(add_137, torch.bfloat16);  add_137 = None
        
         # File: C:\Users\arroy\anaconda3\envs\hrm_v1\lib\site-packages\flash_attn\flash_attn_interface.py:1201 in flash_attn_func, code: return FlashAttnFunc.apply(
        _flash_attn_forward_21 = torch.ops.flash_attn._flash_attn_forward.default(convert_element_type_469, convert_element_type_470, slice_282, 0.0, 0.125, False, -1, -1, 0.0, None, False)
        getitem_126: "bf16[128, 901, 2, 64]" = _flash_attn_forward_21[0]
        getitem_127: "f32[128, 2, 901]" = _flash_attn_forward_21[1]
        getitem_129: "i64[2]" = _flash_attn_forward_21[3];  _flash_attn_forward_21 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:135 in forward, code: attn_output = attn_output.view(batch_size, seq_len, self.output_size)  # type: ignore
        view_219: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(getitem_126, [128, 901, 128])
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_220: "bf16[115328, 128]" = torch.ops.aten.reshape.default(view_219, [115328, 128]);  view_219 = None
        mm_85: "bf16[115328, 128]" = torch.ops.aten.mm.default(view_220, permute_37);  view_220 = None
        view_221: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(mm_85, [128, 901, 128]);  mm_85 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:80 in forward, code: hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        add_138: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(convert_element_type_463, view_221);  convert_element_type_463 = view_221 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:154 in rms_norm, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_474: "f32[128, 901, 128]" = torch.ops.prims.convert_element_type.default(add_138, torch.float32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:156 in rms_norm, code: variance = hidden_states.square().mean(-1, keepdim=True)
        pow_43: "f32[128, 901, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_474, 2)
        mean_42: "f32[128, 901, 1]" = torch.ops.aten.mean.dim(pow_43, [-1], True);  pow_43 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:157 in rms_norm, code: hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        add_139: "f32[128, 901, 1]" = torch.ops.aten.add.Tensor(mean_42, 1e-05);  mean_42 = None
        rsqrt_42: "f32[128, 901, 1]" = torch.ops.aten.rsqrt.default(add_139);  add_139 = None
        mul_173: "f32[128, 901, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_474, rsqrt_42);  convert_element_type_474 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:158 in rms_norm, code: return hidden_states.to(input_dtype)
        convert_element_type_475: "bf16[128, 901, 128]" = torch.ops.prims.convert_element_type.default(mul_173, torch.bfloat16);  mul_173 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_222: "bf16[115328, 128]" = torch.ops.aten.reshape.default(convert_element_type_475, [115328, 128])
        mm_86: "bf16[115328, 1024]" = torch.ops.aten.mm.default(view_222, permute_38)
        view_223: "bf16[128, 901, 1024]" = torch.ops.aten.reshape.default(mm_86, [128, 901, 1024]);  mm_86 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:148 in forward, code: gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        split_21 = torch.ops.aten.split.Tensor(view_223, 512, -1);  view_223 = None
        getitem_130: "bf16[128, 901, 512]" = split_21[0]
        getitem_131: "bf16[128, 901, 512]" = split_21[1];  split_21 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:149 in forward, code: return self.down_proj(F.silu(gate) * up)
        convert_element_type_479: "f32[128, 901, 512]" = torch.ops.prims.convert_element_type.default(getitem_130, torch.float32)
        sigmoid_21: "f32[128, 901, 512]" = torch.ops.aten.sigmoid.default(convert_element_type_479)
        mul_174: "f32[128, 901, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_479, sigmoid_21);  convert_element_type_479 = sigmoid_21 = None
        convert_element_type_480: "bf16[128, 901, 512]" = torch.ops.prims.convert_element_type.default(mul_174, torch.bfloat16);  mul_174 = None
        mul_175: "bf16[128, 901, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_480, getitem_131);  convert_element_type_480 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_224: "bf16[115328, 512]" = torch.ops.aten.reshape.default(mul_175, [115328, 512]);  mul_175 = None
        mm_87: "bf16[115328, 128]" = torch.ops.aten.mm.default(view_224, permute_39)
        view_225: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(mm_87, [128, 901, 128]);  mm_87 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:82 in forward, code: hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)
        add_140: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(convert_element_type_475, view_225);  convert_element_type_475 = view_225 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:154 in rms_norm, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_484: "f32[128, 901, 128]" = torch.ops.prims.convert_element_type.default(add_140, torch.float32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:156 in rms_norm, code: variance = hidden_states.square().mean(-1, keepdim=True)
        pow_44: "f32[128, 901, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_484, 2)
        mean_43: "f32[128, 901, 1]" = torch.ops.aten.mean.dim(pow_44, [-1], True);  pow_44 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:157 in rms_norm, code: hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        add_141: "f32[128, 901, 1]" = torch.ops.aten.add.Tensor(mean_43, 1e-05);  mean_43 = None
        rsqrt_43: "f32[128, 901, 1]" = torch.ops.aten.rsqrt.default(add_141);  add_141 = None
        mul_176: "f32[128, 901, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_484, rsqrt_43);  convert_element_type_484 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:158 in rms_norm, code: return hidden_states.to(input_dtype)
        convert_element_type_485: "bf16[128, 901, 128]" = torch.ops.prims.convert_element_type.default(mul_176, torch.bfloat16);  mul_176 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_226: "bf16[115328, 128]" = torch.ops.aten.reshape.default(convert_element_type_485, [115328, 128])
        mm_88: "bf16[115328, 384]" = torch.ops.aten.mm.default(view_226, permute_40)
        view_227: "bf16[128, 901, 384]" = torch.ops.aten.reshape.default(mm_88, [128, 901, 384]);  mm_88 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:119 in forward, code: qkv = qkv.view(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        view_228: "bf16[128, 901, 6, 64]" = torch.ops.aten.reshape.default(view_227, [128, 901, 6, 64]);  view_227 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:120 in forward, code: query = qkv[:, :, :self.num_heads]
        slice_289: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_228, 2, 0, 2)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:121 in forward, code: key = qkv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
        slice_292: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_228, 2, 2, 4)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:122 in forward, code: value = qkv[:, :, self.num_heads + self.num_key_value_heads:]
        slice_295: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_228, 2, 4, 9223372036854775807);  view_228 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:34 in apply_rotary_pos_emb, code: q = q.to(cos.dtype)
        convert_element_type_489: "f32[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(slice_289, torch.float32);  slice_289 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:35 in apply_rotary_pos_emb, code: k = k.to(cos.dtype)
        convert_element_type_490: "f32[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(slice_292, torch.float32);  slice_292 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:37 in apply_rotary_pos_emb, code: q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
        mul_177: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_489, unsqueeze)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:25 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_296: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_489, 3, 0, 32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:26 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_297: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_489, 3, 32, 9223372036854775807);  convert_element_type_489 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:27 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_44: "f32[128, 901, 2, 32]" = torch.ops.aten.neg.default(slice_297);  slice_297 = None
        cat_45: "f32[128, 901, 2, 64]" = torch.ops.aten.cat.default([neg_44, slice_296], -1);  neg_44 = slice_296 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:37 in apply_rotary_pos_emb, code: q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
        mul_178: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(cat_45, unsqueeze_1);  cat_45 = None
        add_142: "f32[128, 901, 2, 64]" = torch.ops.aten.add.Tensor(mul_177, mul_178);  mul_177 = mul_178 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:38 in apply_rotary_pos_emb, code: k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
        mul_179: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_490, unsqueeze)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:25 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_298: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_490, 3, 0, 32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:26 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_299: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_490, 3, 32, 9223372036854775807);  convert_element_type_490 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:27 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_45: "f32[128, 901, 2, 32]" = torch.ops.aten.neg.default(slice_299);  slice_299 = None
        cat_46: "f32[128, 901, 2, 64]" = torch.ops.aten.cat.default([neg_45, slice_298], -1);  neg_45 = slice_298 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:38 in apply_rotary_pos_emb, code: k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
        mul_180: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(cat_46, unsqueeze_1);  cat_46 = None
        add_143: "f32[128, 901, 2, 64]" = torch.ops.aten.add.Tensor(mul_179, mul_180);  mul_179 = mul_180 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:40 in apply_rotary_pos_emb, code: return q_embed.to(orig_dtype), k_embed.to(orig_dtype)
        convert_element_type_491: "bf16[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(add_142, torch.bfloat16);  add_142 = None
        convert_element_type_492: "bf16[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(add_143, torch.bfloat16);  add_143 = None
        
         # File: C:\Users\arroy\anaconda3\envs\hrm_v1\lib\site-packages\flash_attn\flash_attn_interface.py:1201 in flash_attn_func, code: return FlashAttnFunc.apply(
        _flash_attn_forward_22 = torch.ops.flash_attn._flash_attn_forward.default(convert_element_type_491, convert_element_type_492, slice_295, 0.0, 0.125, False, -1, -1, 0.0, None, False)
        getitem_132: "bf16[128, 901, 2, 64]" = _flash_attn_forward_22[0]
        getitem_133: "f32[128, 2, 901]" = _flash_attn_forward_22[1]
        getitem_135: "i64[2]" = _flash_attn_forward_22[3];  _flash_attn_forward_22 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:135 in forward, code: attn_output = attn_output.view(batch_size, seq_len, self.output_size)  # type: ignore
        view_229: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(getitem_132, [128, 901, 128])
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_230: "bf16[115328, 128]" = torch.ops.aten.reshape.default(view_229, [115328, 128]);  view_229 = None
        mm_89: "bf16[115328, 128]" = torch.ops.aten.mm.default(view_230, permute_41);  view_230 = None
        view_231: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(mm_89, [128, 901, 128]);  mm_89 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:80 in forward, code: hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        add_144: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(convert_element_type_485, view_231);  convert_element_type_485 = view_231 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:154 in rms_norm, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_496: "f32[128, 901, 128]" = torch.ops.prims.convert_element_type.default(add_144, torch.float32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:156 in rms_norm, code: variance = hidden_states.square().mean(-1, keepdim=True)
        pow_45: "f32[128, 901, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_496, 2)
        mean_44: "f32[128, 901, 1]" = torch.ops.aten.mean.dim(pow_45, [-1], True);  pow_45 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:157 in rms_norm, code: hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        add_145: "f32[128, 901, 1]" = torch.ops.aten.add.Tensor(mean_44, 1e-05);  mean_44 = None
        rsqrt_44: "f32[128, 901, 1]" = torch.ops.aten.rsqrt.default(add_145);  add_145 = None
        mul_181: "f32[128, 901, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_496, rsqrt_44);  convert_element_type_496 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:158 in rms_norm, code: return hidden_states.to(input_dtype)
        convert_element_type_497: "bf16[128, 901, 128]" = torch.ops.prims.convert_element_type.default(mul_181, torch.bfloat16);  mul_181 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_232: "bf16[115328, 128]" = torch.ops.aten.reshape.default(convert_element_type_497, [115328, 128])
        mm_90: "bf16[115328, 1024]" = torch.ops.aten.mm.default(view_232, permute_42)
        view_233: "bf16[128, 901, 1024]" = torch.ops.aten.reshape.default(mm_90, [128, 901, 1024]);  mm_90 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:148 in forward, code: gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        split_22 = torch.ops.aten.split.Tensor(view_233, 512, -1);  view_233 = None
        getitem_136: "bf16[128, 901, 512]" = split_22[0]
        getitem_137: "bf16[128, 901, 512]" = split_22[1];  split_22 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:149 in forward, code: return self.down_proj(F.silu(gate) * up)
        convert_element_type_501: "f32[128, 901, 512]" = torch.ops.prims.convert_element_type.default(getitem_136, torch.float32)
        sigmoid_22: "f32[128, 901, 512]" = torch.ops.aten.sigmoid.default(convert_element_type_501)
        mul_182: "f32[128, 901, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_501, sigmoid_22);  convert_element_type_501 = sigmoid_22 = None
        convert_element_type_502: "bf16[128, 901, 512]" = torch.ops.prims.convert_element_type.default(mul_182, torch.bfloat16);  mul_182 = None
        mul_183: "bf16[128, 901, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_502, getitem_137);  convert_element_type_502 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_234: "bf16[115328, 512]" = torch.ops.aten.reshape.default(mul_183, [115328, 512]);  mul_183 = None
        mm_91: "bf16[115328, 128]" = torch.ops.aten.mm.default(view_234, permute_43)
        view_235: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(mm_91, [128, 901, 128]);  mm_91 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:82 in forward, code: hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)
        add_146: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(convert_element_type_497, view_235);  convert_element_type_497 = view_235 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:154 in rms_norm, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_506: "f32[128, 901, 128]" = torch.ops.prims.convert_element_type.default(add_146, torch.float32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:156 in rms_norm, code: variance = hidden_states.square().mean(-1, keepdim=True)
        pow_46: "f32[128, 901, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_506, 2)
        mean_45: "f32[128, 901, 1]" = torch.ops.aten.mean.dim(pow_46, [-1], True);  pow_46 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:157 in rms_norm, code: hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        add_147: "f32[128, 901, 1]" = torch.ops.aten.add.Tensor(mean_45, 1e-05);  mean_45 = None
        rsqrt_45: "f32[128, 901, 1]" = torch.ops.aten.rsqrt.default(add_147);  add_147 = None
        mul_184: "f32[128, 901, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_506, rsqrt_45);  convert_element_type_506 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:158 in rms_norm, code: return hidden_states.to(input_dtype)
        convert_element_type_507: "bf16[128, 901, 128]" = torch.ops.prims.convert_element_type.default(mul_184, torch.bfloat16);  mul_184 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_236: "bf16[115328, 128]" = torch.ops.aten.reshape.default(convert_element_type_507, [115328, 128])
        mm_92: "bf16[115328, 384]" = torch.ops.aten.mm.default(view_236, permute_44)
        view_237: "bf16[128, 901, 384]" = torch.ops.aten.reshape.default(mm_92, [128, 901, 384]);  mm_92 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:119 in forward, code: qkv = qkv.view(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        view_238: "bf16[128, 901, 6, 64]" = torch.ops.aten.reshape.default(view_237, [128, 901, 6, 64]);  view_237 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:120 in forward, code: query = qkv[:, :, :self.num_heads]
        slice_302: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_238, 2, 0, 2)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:121 in forward, code: key = qkv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
        slice_305: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_238, 2, 2, 4)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:122 in forward, code: value = qkv[:, :, self.num_heads + self.num_key_value_heads:]
        slice_308: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_238, 2, 4, 9223372036854775807);  view_238 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:34 in apply_rotary_pos_emb, code: q = q.to(cos.dtype)
        convert_element_type_511: "f32[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(slice_302, torch.float32);  slice_302 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:35 in apply_rotary_pos_emb, code: k = k.to(cos.dtype)
        convert_element_type_512: "f32[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(slice_305, torch.float32);  slice_305 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:37 in apply_rotary_pos_emb, code: q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
        mul_185: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_511, unsqueeze)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:25 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_309: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_511, 3, 0, 32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:26 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_310: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_511, 3, 32, 9223372036854775807);  convert_element_type_511 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:27 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_46: "f32[128, 901, 2, 32]" = torch.ops.aten.neg.default(slice_310);  slice_310 = None
        cat_47: "f32[128, 901, 2, 64]" = torch.ops.aten.cat.default([neg_46, slice_309], -1);  neg_46 = slice_309 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:37 in apply_rotary_pos_emb, code: q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
        mul_186: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(cat_47, unsqueeze_1);  cat_47 = None
        add_148: "f32[128, 901, 2, 64]" = torch.ops.aten.add.Tensor(mul_185, mul_186);  mul_185 = mul_186 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:38 in apply_rotary_pos_emb, code: k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
        mul_187: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_512, unsqueeze);  unsqueeze = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:25 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_311: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_512, 3, 0, 32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:26 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_312: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_512, 3, 32, 9223372036854775807);  convert_element_type_512 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:27 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_47: "f32[128, 901, 2, 32]" = torch.ops.aten.neg.default(slice_312);  slice_312 = None
        cat_48: "f32[128, 901, 2, 64]" = torch.ops.aten.cat.default([neg_47, slice_311], -1);  neg_47 = slice_311 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:38 in apply_rotary_pos_emb, code: k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
        mul_188: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(cat_48, unsqueeze_1);  cat_48 = unsqueeze_1 = None
        add_149: "f32[128, 901, 2, 64]" = torch.ops.aten.add.Tensor(mul_187, mul_188);  mul_187 = mul_188 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:40 in apply_rotary_pos_emb, code: return q_embed.to(orig_dtype), k_embed.to(orig_dtype)
        convert_element_type_513: "bf16[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(add_148, torch.bfloat16);  add_148 = None
        convert_element_type_514: "bf16[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(add_149, torch.bfloat16);  add_149 = None
        
         # File: C:\Users\arroy\anaconda3\envs\hrm_v1\lib\site-packages\flash_attn\flash_attn_interface.py:1201 in flash_attn_func, code: return FlashAttnFunc.apply(
        _flash_attn_forward_23 = torch.ops.flash_attn._flash_attn_forward.default(convert_element_type_513, convert_element_type_514, slice_308, 0.0, 0.125, False, -1, -1, 0.0, None, False)
        getitem_138: "bf16[128, 901, 2, 64]" = _flash_attn_forward_23[0]
        getitem_139: "f32[128, 2, 901]" = _flash_attn_forward_23[1]
        getitem_141: "i64[2]" = _flash_attn_forward_23[3];  _flash_attn_forward_23 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:135 in forward, code: attn_output = attn_output.view(batch_size, seq_len, self.output_size)  # type: ignore
        view_239: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(getitem_138, [128, 901, 128])
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_240: "bf16[115328, 128]" = torch.ops.aten.reshape.default(view_239, [115328, 128]);  view_239 = None
        mm_93: "bf16[115328, 128]" = torch.ops.aten.mm.default(view_240, permute_45);  view_240 = None
        view_241: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(mm_93, [128, 901, 128]);  mm_93 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:80 in forward, code: hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        add_150: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(convert_element_type_507, view_241);  convert_element_type_507 = view_241 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:154 in rms_norm, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_518: "f32[128, 901, 128]" = torch.ops.prims.convert_element_type.default(add_150, torch.float32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:156 in rms_norm, code: variance = hidden_states.square().mean(-1, keepdim=True)
        pow_47: "f32[128, 901, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_518, 2)
        mean_46: "f32[128, 901, 1]" = torch.ops.aten.mean.dim(pow_47, [-1], True);  pow_47 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:157 in rms_norm, code: hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        add_151: "f32[128, 901, 1]" = torch.ops.aten.add.Tensor(mean_46, 1e-05);  mean_46 = None
        rsqrt_46: "f32[128, 901, 1]" = torch.ops.aten.rsqrt.default(add_151);  add_151 = None
        mul_189: "f32[128, 901, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_518, rsqrt_46);  convert_element_type_518 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:158 in rms_norm, code: return hidden_states.to(input_dtype)
        convert_element_type_519: "bf16[128, 901, 128]" = torch.ops.prims.convert_element_type.default(mul_189, torch.bfloat16);  mul_189 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_242: "bf16[115328, 128]" = torch.ops.aten.reshape.default(convert_element_type_519, [115328, 128])
        mm_94: "bf16[115328, 1024]" = torch.ops.aten.mm.default(view_242, permute_46)
        view_243: "bf16[128, 901, 1024]" = torch.ops.aten.reshape.default(mm_94, [128, 901, 1024]);  mm_94 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:148 in forward, code: gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        split_23 = torch.ops.aten.split.Tensor(view_243, 512, -1);  view_243 = None
        getitem_142: "bf16[128, 901, 512]" = split_23[0]
        getitem_143: "bf16[128, 901, 512]" = split_23[1];  split_23 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:149 in forward, code: return self.down_proj(F.silu(gate) * up)
        convert_element_type_523: "f32[128, 901, 512]" = torch.ops.prims.convert_element_type.default(getitem_142, torch.float32)
        sigmoid_23: "f32[128, 901, 512]" = torch.ops.aten.sigmoid.default(convert_element_type_523)
        mul_190: "f32[128, 901, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_523, sigmoid_23);  convert_element_type_523 = sigmoid_23 = None
        convert_element_type_524: "bf16[128, 901, 512]" = torch.ops.prims.convert_element_type.default(mul_190, torch.bfloat16);  mul_190 = None
        mul_191: "bf16[128, 901, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_524, getitem_143);  convert_element_type_524 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_244: "bf16[115328, 512]" = torch.ops.aten.reshape.default(mul_191, [115328, 512]);  mul_191 = None
        mm_95: "bf16[115328, 128]" = torch.ops.aten.mm.default(view_244, permute_47)
        view_245: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(mm_95, [128, 901, 128]);  mm_95 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:82 in forward, code: hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)
        add_152: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(convert_element_type_519, view_245);  convert_element_type_519 = view_245 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:154 in rms_norm, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_528: "f32[128, 901, 128]" = torch.ops.prims.convert_element_type.default(add_152, torch.float32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:156 in rms_norm, code: variance = hidden_states.square().mean(-1, keepdim=True)
        pow_48: "f32[128, 901, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_528, 2)
        mean_47: "f32[128, 901, 1]" = torch.ops.aten.mean.dim(pow_48, [-1], True);  pow_48 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:157 in rms_norm, code: hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        add_153: "f32[128, 901, 1]" = torch.ops.aten.add.Tensor(mean_47, 1e-05);  mean_47 = None
        rsqrt_47: "f32[128, 901, 1]" = torch.ops.aten.rsqrt.default(add_153);  add_153 = None
        mul_192: "f32[128, 901, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_528, rsqrt_47);  convert_element_type_528 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:158 in rms_norm, code: return hidden_states.to(input_dtype)
        convert_element_type_529: "bf16[128, 901, 128]" = torch.ops.prims.convert_element_type.default(mul_192, torch.bfloat16);  mul_192 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        convert_element_type_530: "bf16[10, 128]" = torch.ops.prims.convert_element_type.default(primals_51, torch.bfloat16);  primals_51 = None
        permute_96: "bf16[128, 10]" = torch.ops.aten.permute.default(convert_element_type_530, [1, 0]);  convert_element_type_530 = None
        view_246: "bf16[115328, 128]" = torch.ops.aten.reshape.default(convert_element_type_529, [115328, 128])
        mm_96: "bf16[115328, 10]" = torch.ops.aten.mm.default(view_246, permute_96)
        view_247: "bf16[128, 901, 10]" = torch.ops.aten.reshape.default(mm_96, [128, 901, 10]);  mm_96 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:208 in forward, code: output = self.lm_head(z_H)[:, self.puzzle_emb_len:]
        slice_314: "bf16[128, 900, 10]" = torch.ops.aten.slice.Tensor(view_247, 1, 1, 9223372036854775807);  view_247 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:211 in forward, code: q_logits = self.q_head(z_H[:, 0]).to(torch.float32)
        select: "bf16[128, 128]" = torch.ops.aten.select.int(convert_element_type_529, 1, 0)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        convert_element_type_533: "bf16[2, 128]" = torch.ops.prims.convert_element_type.default(primals_52, torch.bfloat16)
        convert_element_type_534: "bf16[2]" = torch.ops.prims.convert_element_type.default(primals_53, torch.bfloat16)
        permute_97: "bf16[128, 2]" = torch.ops.aten.permute.default(convert_element_type_533, [1, 0]);  convert_element_type_533 = None
        
        # No stacktrace found for following nodes
        mm_default_1: "bf16[128, 2]" = torch.ops.aten.mm.default(select, permute_97)
        add_tensor_1: "bf16[128, 2]" = torch.ops.aten.add.Tensor(mm_default_1, convert_element_type_534);  mm_default_1 = convert_element_type_534 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:211 in forward, code: q_logits = self.q_head(z_H[:, 0]).to(torch.float32)
        convert_element_type_538: "f32[128, 2]" = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.float32);  add_tensor_1 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:213 in forward, code: return new_carry, output, (q_logits[..., 0], q_logits[..., 1])
        select_1: "f32[128]" = torch.ops.aten.select.int(convert_element_type_538, 1, 0)
        select_2: "f32[128]" = torch.ops.aten.select.int(convert_element_type_538, 1, 1)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:259 in forward, code: new_steps = new_steps + 1
        add_154: "i32[128]" = torch.ops.aten.add.Tensor(where_2, 1);  where_2 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:260 in forward, code: is_last_step = new_steps >= self.config.halt_max_steps
        ge: "b8[128]" = torch.ops.aten.ge.Scalar(add_154, 16)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:268 in forward, code: halted = halted | (q_halt_logits > q_continue_logits)
        gt: "b8[128]" = torch.ops.aten.gt.Tensor(select_1, select_2)
        bitwise_or: "b8[128]" = torch.ops.aten.bitwise_or.Tensor(ge, gt);  gt = None
        
        # No stacktrace found for following nodes
        inductor_seeds_default: "i64[2]" = torch.ops.prims.inductor_seeds.default(2, device(type='cuda', index=0))
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0)
        inductor_random_default: "f32[128]" = torch.ops.prims.inductor_random.default([128], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:271 in forward, code: min_halt_steps = (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob) * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)
        lt: "b8[128]" = torch.ops.aten.lt.Scalar(inductor_random_default, 0.1);  inductor_random_default = None
        
        # No stacktrace found for following nodes
        inductor_lookup_seed_default_1: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 1);  inductor_seeds_default = None
        inductor_randint_default: "i64[128]" = torch.ops.prims.inductor_randint.default(2, 17, [128], inductor_lookup_seed_default_1);  inductor_lookup_seed_default_1 = None
        convert_element_type_default_1: "i32[128]" = torch.ops.prims.convert_element_type.default(inductor_randint_default, torch.int32);  inductor_randint_default = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:271 in forward, code: min_halt_steps = (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob) * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)
        mul_193: "i32[128]" = torch.ops.aten.mul.Tensor(lt, convert_element_type_default_1);  lt = convert_element_type_default_1 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:273 in forward, code: halted = halted & (new_steps >= min_halt_steps)
        ge_1: "b8[128]" = torch.ops.aten.ge.Tensor(add_154, mul_193);  mul_193 = None
        bitwise_and: "b8[128]" = torch.ops.aten.bitwise_and.Tensor(bitwise_or, ge_1);  bitwise_or = ge_1 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:77 in forward, code: return F.embedding(input, self.embedding_weight.to(self.cast_to))
        convert_element_type_539: "bf16[10, 128]" = torch.ops.prims.convert_element_type.default(primals_15, torch.bfloat16);  primals_15 = None
        embedding_1: "bf16[128, 900, 128]" = torch.ops.aten.embedding.default(convert_element_type_539, where_3);  convert_element_type_539 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\sparse_embedding.py:35 in forward, code: self.local_weights.copy_(self.weights[inputs])
        index_1: "f32[128, 128]" = torch.ops.aten.index.Tensor(primals_17, [where_5]);  primals_17 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\sparse_embedding.py:38 in forward, code: return self.local_weights.to(self.cast_to)
        convert_element_type_540: "bf16[128, 128]" = torch.ops.prims.convert_element_type.default(index_1, torch.bfloat16)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:158 in _input_embeddings, code: embedding = torch.cat((puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding), dim=-2)
        view_248: "bf16[128, 1, 128]" = torch.ops.aten.reshape.default(convert_element_type_540, [-1, 1, 128]);  convert_element_type_540 = None
        cat_49: "bf16[128, 901, 128]" = torch.ops.aten.cat.default([view_248, embedding_1], -2);  view_248 = embedding_1 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:166 in _input_embeddings, code: return self.embed_scale * embedding
        mul_194: "bf16[128, 901, 128]" = torch.ops.aten.mul.Tensor(cat_49, 11.313708498984761);  cat_49 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:195 in forward, code: z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
        add_155: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(convert_element_type_529, mul_194)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:94 in forward, code: hidden_states = hidden_states + input_injection
        add_156: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(convert_element_type_441, add_155)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        convert_element_type_541: "bf16[384, 128]" = torch.ops.prims.convert_element_type.default(primals_19, torch.bfloat16);  primals_19 = None
        permute_98: "bf16[128, 384]" = torch.ops.aten.permute.default(convert_element_type_541, [1, 0]);  convert_element_type_541 = None
        view_249: "bf16[115328, 128]" = torch.ops.aten.reshape.default(add_156, [115328, 128])
        mm_97: "bf16[115328, 384]" = torch.ops.aten.mm.default(view_249, permute_98);  view_249 = None
        view_250: "bf16[128, 901, 384]" = torch.ops.aten.reshape.default(mm_97, [128, 901, 384]);  mm_97 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:119 in forward, code: qkv = qkv.view(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        view_251: "bf16[128, 901, 6, 64]" = torch.ops.aten.reshape.default(view_250, [128, 901, 6, 64]);  view_250 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:120 in forward, code: query = qkv[:, :, :self.num_heads]
        slice_318: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_251, 2, 0, 2)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:121 in forward, code: key = qkv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
        slice_321: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_251, 2, 2, 4)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:122 in forward, code: value = qkv[:, :, self.num_heads + self.num_key_value_heads:]
        slice_324: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_251, 2, 4, 9223372036854775807);  view_251 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:34 in apply_rotary_pos_emb, code: q = q.to(cos.dtype)
        convert_element_type_544: "f32[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(slice_318, torch.float32);  slice_318 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:35 in apply_rotary_pos_emb, code: k = k.to(cos.dtype)
        convert_element_type_545: "f32[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(slice_321, torch.float32);  slice_321 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:37 in apply_rotary_pos_emb, code: q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
        unsqueeze_96: "f32[901, 1, 64]" = torch.ops.aten.unsqueeze.default(primals_13, -2)
        mul_195: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_544, unsqueeze_96)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:25 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_325: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_544, 3, 0, 32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:26 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_326: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_544, 3, 32, 9223372036854775807);  convert_element_type_544 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:27 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_48: "f32[128, 901, 2, 32]" = torch.ops.aten.neg.default(slice_326);  slice_326 = None
        cat_50: "f32[128, 901, 2, 64]" = torch.ops.aten.cat.default([neg_48, slice_325], -1);  neg_48 = slice_325 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:37 in apply_rotary_pos_emb, code: q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
        unsqueeze_97: "f32[901, 1, 64]" = torch.ops.aten.unsqueeze.default(primals_14, -2)
        mul_196: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(cat_50, unsqueeze_97);  cat_50 = None
        add_157: "f32[128, 901, 2, 64]" = torch.ops.aten.add.Tensor(mul_195, mul_196);  mul_195 = mul_196 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:38 in apply_rotary_pos_emb, code: k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
        mul_197: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_545, unsqueeze_96)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:25 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_327: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_545, 3, 0, 32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:26 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_328: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_545, 3, 32, 9223372036854775807);  convert_element_type_545 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:27 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_49: "f32[128, 901, 2, 32]" = torch.ops.aten.neg.default(slice_328);  slice_328 = None
        cat_51: "f32[128, 901, 2, 64]" = torch.ops.aten.cat.default([neg_49, slice_327], -1);  neg_49 = slice_327 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:38 in apply_rotary_pos_emb, code: k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
        mul_198: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(cat_51, unsqueeze_97);  cat_51 = None
        add_158: "f32[128, 901, 2, 64]" = torch.ops.aten.add.Tensor(mul_197, mul_198);  mul_197 = mul_198 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:40 in apply_rotary_pos_emb, code: return q_embed.to(orig_dtype), k_embed.to(orig_dtype)
        convert_element_type_546: "bf16[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(add_157, torch.bfloat16);  add_157 = None
        convert_element_type_547: "bf16[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(add_158, torch.bfloat16);  add_158 = None
        
         # File: C:\Users\arroy\anaconda3\envs\hrm_v1\lib\site-packages\flash_attn\flash_attn_interface.py:839 in forward, code: out_padded, softmax_lse, S_dmask, rng_state = _wrapped_flash_attn_forward(
        _flash_attn_forward_24 = torch.ops.flash_attn._flash_attn_forward.default(convert_element_type_546, convert_element_type_547, slice_324, 0.0, 0.125, False, -1, -1, 0.0, None, False);  convert_element_type_546 = convert_element_type_547 = slice_324 = None
        getitem_144: "bf16[128, 901, 2, 64]" = _flash_attn_forward_24[0];  _flash_attn_forward_24 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:135 in forward, code: attn_output = attn_output.view(batch_size, seq_len, self.output_size)  # type: ignore
        view_252: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(getitem_144, [128, 901, 128]);  getitem_144 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        convert_element_type_548: "bf16[128, 128]" = torch.ops.prims.convert_element_type.default(primals_20, torch.bfloat16);  primals_20 = None
        permute_99: "bf16[128, 128]" = torch.ops.aten.permute.default(convert_element_type_548, [1, 0]);  convert_element_type_548 = None
        view_253: "bf16[115328, 128]" = torch.ops.aten.reshape.default(view_252, [115328, 128]);  view_252 = None
        mm_98: "bf16[115328, 128]" = torch.ops.aten.mm.default(view_253, permute_99);  view_253 = None
        view_254: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(mm_98, [128, 901, 128]);  mm_98 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:80 in forward, code: hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        add_159: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(add_156, view_254);  add_156 = view_254 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:154 in rms_norm, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_551: "f32[128, 901, 128]" = torch.ops.prims.convert_element_type.default(add_159, torch.float32);  add_159 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:156 in rms_norm, code: variance = hidden_states.square().mean(-1, keepdim=True)
        pow_49: "f32[128, 901, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_551, 2)
        mean_48: "f32[128, 901, 1]" = torch.ops.aten.mean.dim(pow_49, [-1], True);  pow_49 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:157 in rms_norm, code: hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        add_160: "f32[128, 901, 1]" = torch.ops.aten.add.Tensor(mean_48, 1e-05);  mean_48 = None
        rsqrt_48: "f32[128, 901, 1]" = torch.ops.aten.rsqrt.default(add_160);  add_160 = None
        mul_199: "f32[128, 901, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_551, rsqrt_48);  convert_element_type_551 = rsqrt_48 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:158 in rms_norm, code: return hidden_states.to(input_dtype)
        convert_element_type_552: "bf16[128, 901, 128]" = torch.ops.prims.convert_element_type.default(mul_199, torch.bfloat16);  mul_199 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        convert_element_type_553: "bf16[1024, 128]" = torch.ops.prims.convert_element_type.default(primals_21, torch.bfloat16);  primals_21 = None
        permute_100: "bf16[128, 1024]" = torch.ops.aten.permute.default(convert_element_type_553, [1, 0]);  convert_element_type_553 = None
        view_255: "bf16[115328, 128]" = torch.ops.aten.reshape.default(convert_element_type_552, [115328, 128])
        mm_99: "bf16[115328, 1024]" = torch.ops.aten.mm.default(view_255, permute_100);  view_255 = None
        view_256: "bf16[128, 901, 1024]" = torch.ops.aten.reshape.default(mm_99, [128, 901, 1024]);  mm_99 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:148 in forward, code: gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        split_24 = torch.ops.aten.split.Tensor(view_256, 512, -1);  view_256 = None
        getitem_148: "bf16[128, 901, 512]" = split_24[0]
        getitem_149: "bf16[128, 901, 512]" = split_24[1];  split_24 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:149 in forward, code: return self.down_proj(F.silu(gate) * up)
        convert_element_type_556: "f32[128, 901, 512]" = torch.ops.prims.convert_element_type.default(getitem_148, torch.float32);  getitem_148 = None
        sigmoid_24: "f32[128, 901, 512]" = torch.ops.aten.sigmoid.default(convert_element_type_556)
        mul_200: "f32[128, 901, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_556, sigmoid_24);  convert_element_type_556 = sigmoid_24 = None
        convert_element_type_557: "bf16[128, 901, 512]" = torch.ops.prims.convert_element_type.default(mul_200, torch.bfloat16);  mul_200 = None
        mul_201: "bf16[128, 901, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_557, getitem_149);  convert_element_type_557 = getitem_149 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        convert_element_type_558: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(primals_22, torch.bfloat16);  primals_22 = None
        permute_101: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_558, [1, 0]);  convert_element_type_558 = None
        view_257: "bf16[115328, 512]" = torch.ops.aten.reshape.default(mul_201, [115328, 512]);  mul_201 = None
        mm_100: "bf16[115328, 128]" = torch.ops.aten.mm.default(view_257, permute_101);  view_257 = None
        view_258: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(mm_100, [128, 901, 128]);  mm_100 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:82 in forward, code: hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)
        add_161: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(convert_element_type_552, view_258);  convert_element_type_552 = view_258 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:154 in rms_norm, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_561: "f32[128, 901, 128]" = torch.ops.prims.convert_element_type.default(add_161, torch.float32);  add_161 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:156 in rms_norm, code: variance = hidden_states.square().mean(-1, keepdim=True)
        pow_50: "f32[128, 901, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_561, 2)
        mean_49: "f32[128, 901, 1]" = torch.ops.aten.mean.dim(pow_50, [-1], True);  pow_50 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:157 in rms_norm, code: hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        add_162: "f32[128, 901, 1]" = torch.ops.aten.add.Tensor(mean_49, 1e-05);  mean_49 = None
        rsqrt_49: "f32[128, 901, 1]" = torch.ops.aten.rsqrt.default(add_162);  add_162 = None
        mul_202: "f32[128, 901, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_561, rsqrt_49);  convert_element_type_561 = rsqrt_49 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:158 in rms_norm, code: return hidden_states.to(input_dtype)
        convert_element_type_562: "bf16[128, 901, 128]" = torch.ops.prims.convert_element_type.default(mul_202, torch.bfloat16);  mul_202 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        convert_element_type_563: "bf16[384, 128]" = torch.ops.prims.convert_element_type.default(primals_23, torch.bfloat16);  primals_23 = None
        permute_102: "bf16[128, 384]" = torch.ops.aten.permute.default(convert_element_type_563, [1, 0]);  convert_element_type_563 = None
        view_259: "bf16[115328, 128]" = torch.ops.aten.reshape.default(convert_element_type_562, [115328, 128])
        mm_101: "bf16[115328, 384]" = torch.ops.aten.mm.default(view_259, permute_102);  view_259 = None
        view_260: "bf16[128, 901, 384]" = torch.ops.aten.reshape.default(mm_101, [128, 901, 384]);  mm_101 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:119 in forward, code: qkv = qkv.view(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        view_261: "bf16[128, 901, 6, 64]" = torch.ops.aten.reshape.default(view_260, [128, 901, 6, 64]);  view_260 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:120 in forward, code: query = qkv[:, :, :self.num_heads]
        slice_331: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_261, 2, 0, 2)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:121 in forward, code: key = qkv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
        slice_334: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_261, 2, 2, 4)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:122 in forward, code: value = qkv[:, :, self.num_heads + self.num_key_value_heads:]
        slice_337: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_261, 2, 4, 9223372036854775807);  view_261 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:34 in apply_rotary_pos_emb, code: q = q.to(cos.dtype)
        convert_element_type_566: "f32[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(slice_331, torch.float32);  slice_331 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:35 in apply_rotary_pos_emb, code: k = k.to(cos.dtype)
        convert_element_type_567: "f32[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(slice_334, torch.float32);  slice_334 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:37 in apply_rotary_pos_emb, code: q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
        mul_203: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_566, unsqueeze_96)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:25 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_338: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_566, 3, 0, 32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:26 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_339: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_566, 3, 32, 9223372036854775807);  convert_element_type_566 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:27 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_50: "f32[128, 901, 2, 32]" = torch.ops.aten.neg.default(slice_339);  slice_339 = None
        cat_52: "f32[128, 901, 2, 64]" = torch.ops.aten.cat.default([neg_50, slice_338], -1);  neg_50 = slice_338 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:37 in apply_rotary_pos_emb, code: q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
        mul_204: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(cat_52, unsqueeze_97);  cat_52 = None
        add_163: "f32[128, 901, 2, 64]" = torch.ops.aten.add.Tensor(mul_203, mul_204);  mul_203 = mul_204 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:38 in apply_rotary_pos_emb, code: k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
        mul_205: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_567, unsqueeze_96)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:25 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_340: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_567, 3, 0, 32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:26 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_341: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_567, 3, 32, 9223372036854775807);  convert_element_type_567 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:27 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_51: "f32[128, 901, 2, 32]" = torch.ops.aten.neg.default(slice_341);  slice_341 = None
        cat_53: "f32[128, 901, 2, 64]" = torch.ops.aten.cat.default([neg_51, slice_340], -1);  neg_51 = slice_340 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:38 in apply_rotary_pos_emb, code: k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
        mul_206: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(cat_53, unsqueeze_97);  cat_53 = None
        add_164: "f32[128, 901, 2, 64]" = torch.ops.aten.add.Tensor(mul_205, mul_206);  mul_205 = mul_206 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:40 in apply_rotary_pos_emb, code: return q_embed.to(orig_dtype), k_embed.to(orig_dtype)
        convert_element_type_568: "bf16[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(add_163, torch.bfloat16);  add_163 = None
        convert_element_type_569: "bf16[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(add_164, torch.bfloat16);  add_164 = None
        
         # File: C:\Users\arroy\anaconda3\envs\hrm_v1\lib\site-packages\flash_attn\flash_attn_interface.py:839 in forward, code: out_padded, softmax_lse, S_dmask, rng_state = _wrapped_flash_attn_forward(
        _flash_attn_forward_25 = torch.ops.flash_attn._flash_attn_forward.default(convert_element_type_568, convert_element_type_569, slice_337, 0.0, 0.125, False, -1, -1, 0.0, None, False);  convert_element_type_568 = convert_element_type_569 = slice_337 = None
        getitem_150: "bf16[128, 901, 2, 64]" = _flash_attn_forward_25[0];  _flash_attn_forward_25 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:135 in forward, code: attn_output = attn_output.view(batch_size, seq_len, self.output_size)  # type: ignore
        view_262: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(getitem_150, [128, 901, 128]);  getitem_150 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        convert_element_type_570: "bf16[128, 128]" = torch.ops.prims.convert_element_type.default(primals_24, torch.bfloat16);  primals_24 = None
        permute_103: "bf16[128, 128]" = torch.ops.aten.permute.default(convert_element_type_570, [1, 0]);  convert_element_type_570 = None
        view_263: "bf16[115328, 128]" = torch.ops.aten.reshape.default(view_262, [115328, 128]);  view_262 = None
        mm_102: "bf16[115328, 128]" = torch.ops.aten.mm.default(view_263, permute_103);  view_263 = None
        view_264: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(mm_102, [128, 901, 128]);  mm_102 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:80 in forward, code: hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        add_165: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(convert_element_type_562, view_264);  convert_element_type_562 = view_264 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:154 in rms_norm, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_573: "f32[128, 901, 128]" = torch.ops.prims.convert_element_type.default(add_165, torch.float32);  add_165 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:156 in rms_norm, code: variance = hidden_states.square().mean(-1, keepdim=True)
        pow_51: "f32[128, 901, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_573, 2)
        mean_50: "f32[128, 901, 1]" = torch.ops.aten.mean.dim(pow_51, [-1], True);  pow_51 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:157 in rms_norm, code: hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        add_166: "f32[128, 901, 1]" = torch.ops.aten.add.Tensor(mean_50, 1e-05);  mean_50 = None
        rsqrt_50: "f32[128, 901, 1]" = torch.ops.aten.rsqrt.default(add_166);  add_166 = None
        mul_207: "f32[128, 901, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_573, rsqrt_50);  convert_element_type_573 = rsqrt_50 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:158 in rms_norm, code: return hidden_states.to(input_dtype)
        convert_element_type_574: "bf16[128, 901, 128]" = torch.ops.prims.convert_element_type.default(mul_207, torch.bfloat16);  mul_207 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        convert_element_type_575: "bf16[1024, 128]" = torch.ops.prims.convert_element_type.default(primals_25, torch.bfloat16);  primals_25 = None
        permute_104: "bf16[128, 1024]" = torch.ops.aten.permute.default(convert_element_type_575, [1, 0]);  convert_element_type_575 = None
        view_265: "bf16[115328, 128]" = torch.ops.aten.reshape.default(convert_element_type_574, [115328, 128])
        mm_103: "bf16[115328, 1024]" = torch.ops.aten.mm.default(view_265, permute_104);  view_265 = None
        view_266: "bf16[128, 901, 1024]" = torch.ops.aten.reshape.default(mm_103, [128, 901, 1024]);  mm_103 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:148 in forward, code: gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        split_25 = torch.ops.aten.split.Tensor(view_266, 512, -1);  view_266 = None
        getitem_154: "bf16[128, 901, 512]" = split_25[0]
        getitem_155: "bf16[128, 901, 512]" = split_25[1];  split_25 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:149 in forward, code: return self.down_proj(F.silu(gate) * up)
        convert_element_type_578: "f32[128, 901, 512]" = torch.ops.prims.convert_element_type.default(getitem_154, torch.float32);  getitem_154 = None
        sigmoid_25: "f32[128, 901, 512]" = torch.ops.aten.sigmoid.default(convert_element_type_578)
        mul_208: "f32[128, 901, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_578, sigmoid_25);  convert_element_type_578 = sigmoid_25 = None
        convert_element_type_579: "bf16[128, 901, 512]" = torch.ops.prims.convert_element_type.default(mul_208, torch.bfloat16);  mul_208 = None
        mul_209: "bf16[128, 901, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_579, getitem_155);  convert_element_type_579 = getitem_155 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        convert_element_type_580: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(primals_26, torch.bfloat16);  primals_26 = None
        permute_105: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_580, [1, 0]);  convert_element_type_580 = None
        view_267: "bf16[115328, 512]" = torch.ops.aten.reshape.default(mul_209, [115328, 512]);  mul_209 = None
        mm_104: "bf16[115328, 128]" = torch.ops.aten.mm.default(view_267, permute_105);  view_267 = None
        view_268: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(mm_104, [128, 901, 128]);  mm_104 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:82 in forward, code: hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)
        add_167: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(convert_element_type_574, view_268);  convert_element_type_574 = view_268 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:154 in rms_norm, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_583: "f32[128, 901, 128]" = torch.ops.prims.convert_element_type.default(add_167, torch.float32);  add_167 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:156 in rms_norm, code: variance = hidden_states.square().mean(-1, keepdim=True)
        pow_52: "f32[128, 901, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_583, 2)
        mean_51: "f32[128, 901, 1]" = torch.ops.aten.mean.dim(pow_52, [-1], True);  pow_52 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:157 in rms_norm, code: hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        add_168: "f32[128, 901, 1]" = torch.ops.aten.add.Tensor(mean_51, 1e-05);  mean_51 = None
        rsqrt_51: "f32[128, 901, 1]" = torch.ops.aten.rsqrt.default(add_168);  add_168 = None
        mul_210: "f32[128, 901, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_583, rsqrt_51);  convert_element_type_583 = rsqrt_51 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:158 in rms_norm, code: return hidden_states.to(input_dtype)
        convert_element_type_584: "bf16[128, 901, 128]" = torch.ops.prims.convert_element_type.default(mul_210, torch.bfloat16);  mul_210 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        convert_element_type_585: "bf16[384, 128]" = torch.ops.prims.convert_element_type.default(primals_27, torch.bfloat16);  primals_27 = None
        permute_106: "bf16[128, 384]" = torch.ops.aten.permute.default(convert_element_type_585, [1, 0]);  convert_element_type_585 = None
        view_269: "bf16[115328, 128]" = torch.ops.aten.reshape.default(convert_element_type_584, [115328, 128])
        mm_105: "bf16[115328, 384]" = torch.ops.aten.mm.default(view_269, permute_106);  view_269 = None
        view_270: "bf16[128, 901, 384]" = torch.ops.aten.reshape.default(mm_105, [128, 901, 384]);  mm_105 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:119 in forward, code: qkv = qkv.view(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        view_271: "bf16[128, 901, 6, 64]" = torch.ops.aten.reshape.default(view_270, [128, 901, 6, 64]);  view_270 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:120 in forward, code: query = qkv[:, :, :self.num_heads]
        slice_344: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_271, 2, 0, 2)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:121 in forward, code: key = qkv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
        slice_347: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_271, 2, 2, 4)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:122 in forward, code: value = qkv[:, :, self.num_heads + self.num_key_value_heads:]
        slice_350: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_271, 2, 4, 9223372036854775807);  view_271 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:34 in apply_rotary_pos_emb, code: q = q.to(cos.dtype)
        convert_element_type_588: "f32[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(slice_344, torch.float32);  slice_344 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:35 in apply_rotary_pos_emb, code: k = k.to(cos.dtype)
        convert_element_type_589: "f32[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(slice_347, torch.float32);  slice_347 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:37 in apply_rotary_pos_emb, code: q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
        mul_211: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_588, unsqueeze_96)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:25 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_351: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_588, 3, 0, 32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:26 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_352: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_588, 3, 32, 9223372036854775807);  convert_element_type_588 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:27 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_52: "f32[128, 901, 2, 32]" = torch.ops.aten.neg.default(slice_352);  slice_352 = None
        cat_54: "f32[128, 901, 2, 64]" = torch.ops.aten.cat.default([neg_52, slice_351], -1);  neg_52 = slice_351 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:37 in apply_rotary_pos_emb, code: q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
        mul_212: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(cat_54, unsqueeze_97);  cat_54 = None
        add_169: "f32[128, 901, 2, 64]" = torch.ops.aten.add.Tensor(mul_211, mul_212);  mul_211 = mul_212 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:38 in apply_rotary_pos_emb, code: k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
        mul_213: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_589, unsqueeze_96)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:25 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_353: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_589, 3, 0, 32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:26 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_354: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_589, 3, 32, 9223372036854775807);  convert_element_type_589 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:27 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_53: "f32[128, 901, 2, 32]" = torch.ops.aten.neg.default(slice_354);  slice_354 = None
        cat_55: "f32[128, 901, 2, 64]" = torch.ops.aten.cat.default([neg_53, slice_353], -1);  neg_53 = slice_353 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:38 in apply_rotary_pos_emb, code: k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
        mul_214: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(cat_55, unsqueeze_97);  cat_55 = None
        add_170: "f32[128, 901, 2, 64]" = torch.ops.aten.add.Tensor(mul_213, mul_214);  mul_213 = mul_214 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:40 in apply_rotary_pos_emb, code: return q_embed.to(orig_dtype), k_embed.to(orig_dtype)
        convert_element_type_590: "bf16[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(add_169, torch.bfloat16);  add_169 = None
        convert_element_type_591: "bf16[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(add_170, torch.bfloat16);  add_170 = None
        
         # File: C:\Users\arroy\anaconda3\envs\hrm_v1\lib\site-packages\flash_attn\flash_attn_interface.py:839 in forward, code: out_padded, softmax_lse, S_dmask, rng_state = _wrapped_flash_attn_forward(
        _flash_attn_forward_26 = torch.ops.flash_attn._flash_attn_forward.default(convert_element_type_590, convert_element_type_591, slice_350, 0.0, 0.125, False, -1, -1, 0.0, None, False);  convert_element_type_590 = convert_element_type_591 = slice_350 = None
        getitem_156: "bf16[128, 901, 2, 64]" = _flash_attn_forward_26[0];  _flash_attn_forward_26 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:135 in forward, code: attn_output = attn_output.view(batch_size, seq_len, self.output_size)  # type: ignore
        view_272: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(getitem_156, [128, 901, 128]);  getitem_156 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        convert_element_type_592: "bf16[128, 128]" = torch.ops.prims.convert_element_type.default(primals_28, torch.bfloat16);  primals_28 = None
        permute_107: "bf16[128, 128]" = torch.ops.aten.permute.default(convert_element_type_592, [1, 0]);  convert_element_type_592 = None
        view_273: "bf16[115328, 128]" = torch.ops.aten.reshape.default(view_272, [115328, 128]);  view_272 = None
        mm_106: "bf16[115328, 128]" = torch.ops.aten.mm.default(view_273, permute_107);  view_273 = None
        view_274: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(mm_106, [128, 901, 128]);  mm_106 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:80 in forward, code: hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        add_171: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(convert_element_type_584, view_274);  convert_element_type_584 = view_274 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:154 in rms_norm, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_595: "f32[128, 901, 128]" = torch.ops.prims.convert_element_type.default(add_171, torch.float32);  add_171 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:156 in rms_norm, code: variance = hidden_states.square().mean(-1, keepdim=True)
        pow_53: "f32[128, 901, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_595, 2)
        mean_52: "f32[128, 901, 1]" = torch.ops.aten.mean.dim(pow_53, [-1], True);  pow_53 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:157 in rms_norm, code: hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        add_172: "f32[128, 901, 1]" = torch.ops.aten.add.Tensor(mean_52, 1e-05);  mean_52 = None
        rsqrt_52: "f32[128, 901, 1]" = torch.ops.aten.rsqrt.default(add_172);  add_172 = None
        mul_215: "f32[128, 901, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_595, rsqrt_52);  convert_element_type_595 = rsqrt_52 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:158 in rms_norm, code: return hidden_states.to(input_dtype)
        convert_element_type_596: "bf16[128, 901, 128]" = torch.ops.prims.convert_element_type.default(mul_215, torch.bfloat16);  mul_215 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        convert_element_type_597: "bf16[1024, 128]" = torch.ops.prims.convert_element_type.default(primals_29, torch.bfloat16);  primals_29 = None
        permute_108: "bf16[128, 1024]" = torch.ops.aten.permute.default(convert_element_type_597, [1, 0]);  convert_element_type_597 = None
        view_275: "bf16[115328, 128]" = torch.ops.aten.reshape.default(convert_element_type_596, [115328, 128])
        mm_107: "bf16[115328, 1024]" = torch.ops.aten.mm.default(view_275, permute_108);  view_275 = None
        view_276: "bf16[128, 901, 1024]" = torch.ops.aten.reshape.default(mm_107, [128, 901, 1024]);  mm_107 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:148 in forward, code: gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        split_26 = torch.ops.aten.split.Tensor(view_276, 512, -1);  view_276 = None
        getitem_160: "bf16[128, 901, 512]" = split_26[0]
        getitem_161: "bf16[128, 901, 512]" = split_26[1];  split_26 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:149 in forward, code: return self.down_proj(F.silu(gate) * up)
        convert_element_type_600: "f32[128, 901, 512]" = torch.ops.prims.convert_element_type.default(getitem_160, torch.float32);  getitem_160 = None
        sigmoid_26: "f32[128, 901, 512]" = torch.ops.aten.sigmoid.default(convert_element_type_600)
        mul_216: "f32[128, 901, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_600, sigmoid_26);  convert_element_type_600 = sigmoid_26 = None
        convert_element_type_601: "bf16[128, 901, 512]" = torch.ops.prims.convert_element_type.default(mul_216, torch.bfloat16);  mul_216 = None
        mul_217: "bf16[128, 901, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_601, getitem_161);  convert_element_type_601 = getitem_161 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        convert_element_type_602: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(primals_30, torch.bfloat16);  primals_30 = None
        permute_109: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_602, [1, 0]);  convert_element_type_602 = None
        view_277: "bf16[115328, 512]" = torch.ops.aten.reshape.default(mul_217, [115328, 512]);  mul_217 = None
        mm_108: "bf16[115328, 128]" = torch.ops.aten.mm.default(view_277, permute_109);  view_277 = None
        view_278: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(mm_108, [128, 901, 128]);  mm_108 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:82 in forward, code: hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)
        add_173: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(convert_element_type_596, view_278);  convert_element_type_596 = view_278 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:154 in rms_norm, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_605: "f32[128, 901, 128]" = torch.ops.prims.convert_element_type.default(add_173, torch.float32);  add_173 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:156 in rms_norm, code: variance = hidden_states.square().mean(-1, keepdim=True)
        pow_54: "f32[128, 901, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_605, 2)
        mean_53: "f32[128, 901, 1]" = torch.ops.aten.mean.dim(pow_54, [-1], True);  pow_54 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:157 in rms_norm, code: hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        add_174: "f32[128, 901, 1]" = torch.ops.aten.add.Tensor(mean_53, 1e-05);  mean_53 = None
        rsqrt_53: "f32[128, 901, 1]" = torch.ops.aten.rsqrt.default(add_174);  add_174 = None
        mul_218: "f32[128, 901, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_605, rsqrt_53);  convert_element_type_605 = rsqrt_53 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:158 in rms_norm, code: return hidden_states.to(input_dtype)
        convert_element_type_606: "bf16[128, 901, 128]" = torch.ops.prims.convert_element_type.default(mul_218, torch.bfloat16);  mul_218 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        convert_element_type_607: "bf16[384, 128]" = torch.ops.prims.convert_element_type.default(primals_31, torch.bfloat16);  primals_31 = None
        permute_110: "bf16[128, 384]" = torch.ops.aten.permute.default(convert_element_type_607, [1, 0]);  convert_element_type_607 = None
        view_279: "bf16[115328, 128]" = torch.ops.aten.reshape.default(convert_element_type_606, [115328, 128])
        mm_109: "bf16[115328, 384]" = torch.ops.aten.mm.default(view_279, permute_110);  view_279 = None
        view_280: "bf16[128, 901, 384]" = torch.ops.aten.reshape.default(mm_109, [128, 901, 384]);  mm_109 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:119 in forward, code: qkv = qkv.view(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        view_281: "bf16[128, 901, 6, 64]" = torch.ops.aten.reshape.default(view_280, [128, 901, 6, 64]);  view_280 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:120 in forward, code: query = qkv[:, :, :self.num_heads]
        slice_357: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_281, 2, 0, 2)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:121 in forward, code: key = qkv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
        slice_360: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_281, 2, 2, 4)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:122 in forward, code: value = qkv[:, :, self.num_heads + self.num_key_value_heads:]
        slice_363: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_281, 2, 4, 9223372036854775807);  view_281 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:34 in apply_rotary_pos_emb, code: q = q.to(cos.dtype)
        convert_element_type_610: "f32[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(slice_357, torch.float32);  slice_357 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:35 in apply_rotary_pos_emb, code: k = k.to(cos.dtype)
        convert_element_type_611: "f32[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(slice_360, torch.float32);  slice_360 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:37 in apply_rotary_pos_emb, code: q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
        mul_219: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_610, unsqueeze_96)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:25 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_364: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_610, 3, 0, 32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:26 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_365: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_610, 3, 32, 9223372036854775807);  convert_element_type_610 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:27 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_54: "f32[128, 901, 2, 32]" = torch.ops.aten.neg.default(slice_365);  slice_365 = None
        cat_56: "f32[128, 901, 2, 64]" = torch.ops.aten.cat.default([neg_54, slice_364], -1);  neg_54 = slice_364 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:37 in apply_rotary_pos_emb, code: q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
        mul_220: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(cat_56, unsqueeze_97);  cat_56 = None
        add_175: "f32[128, 901, 2, 64]" = torch.ops.aten.add.Tensor(mul_219, mul_220);  mul_219 = mul_220 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:38 in apply_rotary_pos_emb, code: k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
        mul_221: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_611, unsqueeze_96)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:25 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_366: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_611, 3, 0, 32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:26 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_367: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_611, 3, 32, 9223372036854775807);  convert_element_type_611 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:27 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_55: "f32[128, 901, 2, 32]" = torch.ops.aten.neg.default(slice_367);  slice_367 = None
        cat_57: "f32[128, 901, 2, 64]" = torch.ops.aten.cat.default([neg_55, slice_366], -1);  neg_55 = slice_366 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:38 in apply_rotary_pos_emb, code: k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
        mul_222: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(cat_57, unsqueeze_97);  cat_57 = None
        add_176: "f32[128, 901, 2, 64]" = torch.ops.aten.add.Tensor(mul_221, mul_222);  mul_221 = mul_222 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:40 in apply_rotary_pos_emb, code: return q_embed.to(orig_dtype), k_embed.to(orig_dtype)
        convert_element_type_612: "bf16[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(add_175, torch.bfloat16);  add_175 = None
        convert_element_type_613: "bf16[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(add_176, torch.bfloat16);  add_176 = None
        
         # File: C:\Users\arroy\anaconda3\envs\hrm_v1\lib\site-packages\flash_attn\flash_attn_interface.py:839 in forward, code: out_padded, softmax_lse, S_dmask, rng_state = _wrapped_flash_attn_forward(
        _flash_attn_forward_27 = torch.ops.flash_attn._flash_attn_forward.default(convert_element_type_612, convert_element_type_613, slice_363, 0.0, 0.125, False, -1, -1, 0.0, None, False);  convert_element_type_612 = convert_element_type_613 = slice_363 = None
        getitem_162: "bf16[128, 901, 2, 64]" = _flash_attn_forward_27[0];  _flash_attn_forward_27 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:135 in forward, code: attn_output = attn_output.view(batch_size, seq_len, self.output_size)  # type: ignore
        view_282: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(getitem_162, [128, 901, 128]);  getitem_162 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        convert_element_type_614: "bf16[128, 128]" = torch.ops.prims.convert_element_type.default(primals_32, torch.bfloat16);  primals_32 = None
        permute_111: "bf16[128, 128]" = torch.ops.aten.permute.default(convert_element_type_614, [1, 0]);  convert_element_type_614 = None
        view_283: "bf16[115328, 128]" = torch.ops.aten.reshape.default(view_282, [115328, 128]);  view_282 = None
        mm_110: "bf16[115328, 128]" = torch.ops.aten.mm.default(view_283, permute_111);  view_283 = None
        view_284: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(mm_110, [128, 901, 128]);  mm_110 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:80 in forward, code: hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        add_177: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(convert_element_type_606, view_284);  convert_element_type_606 = view_284 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:154 in rms_norm, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_617: "f32[128, 901, 128]" = torch.ops.prims.convert_element_type.default(add_177, torch.float32);  add_177 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:156 in rms_norm, code: variance = hidden_states.square().mean(-1, keepdim=True)
        pow_55: "f32[128, 901, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_617, 2)
        mean_54: "f32[128, 901, 1]" = torch.ops.aten.mean.dim(pow_55, [-1], True);  pow_55 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:157 in rms_norm, code: hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        add_178: "f32[128, 901, 1]" = torch.ops.aten.add.Tensor(mean_54, 1e-05);  mean_54 = None
        rsqrt_54: "f32[128, 901, 1]" = torch.ops.aten.rsqrt.default(add_178);  add_178 = None
        mul_223: "f32[128, 901, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_617, rsqrt_54);  convert_element_type_617 = rsqrt_54 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:158 in rms_norm, code: return hidden_states.to(input_dtype)
        convert_element_type_618: "bf16[128, 901, 128]" = torch.ops.prims.convert_element_type.default(mul_223, torch.bfloat16);  mul_223 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        convert_element_type_619: "bf16[1024, 128]" = torch.ops.prims.convert_element_type.default(primals_33, torch.bfloat16);  primals_33 = None
        permute_112: "bf16[128, 1024]" = torch.ops.aten.permute.default(convert_element_type_619, [1, 0]);  convert_element_type_619 = None
        view_285: "bf16[115328, 128]" = torch.ops.aten.reshape.default(convert_element_type_618, [115328, 128])
        mm_111: "bf16[115328, 1024]" = torch.ops.aten.mm.default(view_285, permute_112);  view_285 = None
        view_286: "bf16[128, 901, 1024]" = torch.ops.aten.reshape.default(mm_111, [128, 901, 1024]);  mm_111 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:148 in forward, code: gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        split_27 = torch.ops.aten.split.Tensor(view_286, 512, -1);  view_286 = None
        getitem_166: "bf16[128, 901, 512]" = split_27[0]
        getitem_167: "bf16[128, 901, 512]" = split_27[1];  split_27 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:149 in forward, code: return self.down_proj(F.silu(gate) * up)
        convert_element_type_622: "f32[128, 901, 512]" = torch.ops.prims.convert_element_type.default(getitem_166, torch.float32);  getitem_166 = None
        sigmoid_27: "f32[128, 901, 512]" = torch.ops.aten.sigmoid.default(convert_element_type_622)
        mul_224: "f32[128, 901, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_622, sigmoid_27);  convert_element_type_622 = sigmoid_27 = None
        convert_element_type_623: "bf16[128, 901, 512]" = torch.ops.prims.convert_element_type.default(mul_224, torch.bfloat16);  mul_224 = None
        mul_225: "bf16[128, 901, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_623, getitem_167);  convert_element_type_623 = getitem_167 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        convert_element_type_624: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(primals_34, torch.bfloat16);  primals_34 = None
        permute_113: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_624, [1, 0]);  convert_element_type_624 = None
        view_287: "bf16[115328, 512]" = torch.ops.aten.reshape.default(mul_225, [115328, 512]);  mul_225 = None
        mm_112: "bf16[115328, 128]" = torch.ops.aten.mm.default(view_287, permute_113);  view_287 = None
        view_288: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(mm_112, [128, 901, 128]);  mm_112 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:82 in forward, code: hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)
        add_179: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(convert_element_type_618, view_288);  convert_element_type_618 = view_288 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:154 in rms_norm, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_627: "f32[128, 901, 128]" = torch.ops.prims.convert_element_type.default(add_179, torch.float32);  add_179 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:156 in rms_norm, code: variance = hidden_states.square().mean(-1, keepdim=True)
        pow_56: "f32[128, 901, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_627, 2)
        mean_55: "f32[128, 901, 1]" = torch.ops.aten.mean.dim(pow_56, [-1], True);  pow_56 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:157 in rms_norm, code: hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        add_180: "f32[128, 901, 1]" = torch.ops.aten.add.Tensor(mean_55, 1e-05);  mean_55 = None
        rsqrt_55: "f32[128, 901, 1]" = torch.ops.aten.rsqrt.default(add_180);  add_180 = None
        mul_226: "f32[128, 901, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_627, rsqrt_55);  convert_element_type_627 = rsqrt_55 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:158 in rms_norm, code: return hidden_states.to(input_dtype)
        convert_element_type_628: "bf16[128, 901, 128]" = torch.ops.prims.convert_element_type.default(mul_226, torch.bfloat16);  mul_226 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:94 in forward, code: hidden_states = hidden_states + input_injection
        add_182: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(convert_element_type_628, add_155);  convert_element_type_628 = add_155 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_289: "bf16[115328, 128]" = torch.ops.aten.reshape.default(add_182, [115328, 128])
        mm_113: "bf16[115328, 384]" = torch.ops.aten.mm.default(view_289, permute_98);  view_289 = None
        view_290: "bf16[128, 901, 384]" = torch.ops.aten.reshape.default(mm_113, [128, 901, 384]);  mm_113 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:119 in forward, code: qkv = qkv.view(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        view_291: "bf16[128, 901, 6, 64]" = torch.ops.aten.reshape.default(view_290, [128, 901, 6, 64]);  view_290 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:120 in forward, code: query = qkv[:, :, :self.num_heads]
        slice_370: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_291, 2, 0, 2)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:121 in forward, code: key = qkv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
        slice_373: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_291, 2, 2, 4)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:122 in forward, code: value = qkv[:, :, self.num_heads + self.num_key_value_heads:]
        slice_376: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_291, 2, 4, 9223372036854775807);  view_291 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:34 in apply_rotary_pos_emb, code: q = q.to(cos.dtype)
        convert_element_type_632: "f32[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(slice_370, torch.float32);  slice_370 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:35 in apply_rotary_pos_emb, code: k = k.to(cos.dtype)
        convert_element_type_633: "f32[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(slice_373, torch.float32);  slice_373 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:37 in apply_rotary_pos_emb, code: q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
        mul_227: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_632, unsqueeze_96)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:25 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_377: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_632, 3, 0, 32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:26 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_378: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_632, 3, 32, 9223372036854775807);  convert_element_type_632 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:27 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_56: "f32[128, 901, 2, 32]" = torch.ops.aten.neg.default(slice_378);  slice_378 = None
        cat_58: "f32[128, 901, 2, 64]" = torch.ops.aten.cat.default([neg_56, slice_377], -1);  neg_56 = slice_377 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:37 in apply_rotary_pos_emb, code: q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
        mul_228: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(cat_58, unsqueeze_97);  cat_58 = None
        add_183: "f32[128, 901, 2, 64]" = torch.ops.aten.add.Tensor(mul_227, mul_228);  mul_227 = mul_228 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:38 in apply_rotary_pos_emb, code: k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
        mul_229: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_633, unsqueeze_96)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:25 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_379: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_633, 3, 0, 32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:26 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_380: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_633, 3, 32, 9223372036854775807);  convert_element_type_633 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:27 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_57: "f32[128, 901, 2, 32]" = torch.ops.aten.neg.default(slice_380);  slice_380 = None
        cat_59: "f32[128, 901, 2, 64]" = torch.ops.aten.cat.default([neg_57, slice_379], -1);  neg_57 = slice_379 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:38 in apply_rotary_pos_emb, code: k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
        mul_230: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(cat_59, unsqueeze_97);  cat_59 = None
        add_184: "f32[128, 901, 2, 64]" = torch.ops.aten.add.Tensor(mul_229, mul_230);  mul_229 = mul_230 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:40 in apply_rotary_pos_emb, code: return q_embed.to(orig_dtype), k_embed.to(orig_dtype)
        convert_element_type_634: "bf16[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(add_183, torch.bfloat16);  add_183 = None
        convert_element_type_635: "bf16[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(add_184, torch.bfloat16);  add_184 = None
        
         # File: C:\Users\arroy\anaconda3\envs\hrm_v1\lib\site-packages\flash_attn\flash_attn_interface.py:839 in forward, code: out_padded, softmax_lse, S_dmask, rng_state = _wrapped_flash_attn_forward(
        _flash_attn_forward_28 = torch.ops.flash_attn._flash_attn_forward.default(convert_element_type_634, convert_element_type_635, slice_376, 0.0, 0.125, False, -1, -1, 0.0, None, False);  convert_element_type_634 = convert_element_type_635 = slice_376 = None
        getitem_168: "bf16[128, 901, 2, 64]" = _flash_attn_forward_28[0];  _flash_attn_forward_28 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:135 in forward, code: attn_output = attn_output.view(batch_size, seq_len, self.output_size)  # type: ignore
        view_292: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(getitem_168, [128, 901, 128]);  getitem_168 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_293: "bf16[115328, 128]" = torch.ops.aten.reshape.default(view_292, [115328, 128]);  view_292 = None
        mm_114: "bf16[115328, 128]" = torch.ops.aten.mm.default(view_293, permute_99);  view_293 = None
        view_294: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(mm_114, [128, 901, 128]);  mm_114 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:80 in forward, code: hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        add_185: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(add_182, view_294);  add_182 = view_294 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:154 in rms_norm, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_639: "f32[128, 901, 128]" = torch.ops.prims.convert_element_type.default(add_185, torch.float32);  add_185 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:156 in rms_norm, code: variance = hidden_states.square().mean(-1, keepdim=True)
        pow_57: "f32[128, 901, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_639, 2)
        mean_56: "f32[128, 901, 1]" = torch.ops.aten.mean.dim(pow_57, [-1], True);  pow_57 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:157 in rms_norm, code: hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        add_186: "f32[128, 901, 1]" = torch.ops.aten.add.Tensor(mean_56, 1e-05);  mean_56 = None
        rsqrt_56: "f32[128, 901, 1]" = torch.ops.aten.rsqrt.default(add_186);  add_186 = None
        mul_231: "f32[128, 901, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_639, rsqrt_56);  convert_element_type_639 = rsqrt_56 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:158 in rms_norm, code: return hidden_states.to(input_dtype)
        convert_element_type_640: "bf16[128, 901, 128]" = torch.ops.prims.convert_element_type.default(mul_231, torch.bfloat16);  mul_231 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_295: "bf16[115328, 128]" = torch.ops.aten.reshape.default(convert_element_type_640, [115328, 128])
        mm_115: "bf16[115328, 1024]" = torch.ops.aten.mm.default(view_295, permute_100);  view_295 = None
        view_296: "bf16[128, 901, 1024]" = torch.ops.aten.reshape.default(mm_115, [128, 901, 1024]);  mm_115 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:148 in forward, code: gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        split_28 = torch.ops.aten.split.Tensor(view_296, 512, -1);  view_296 = None
        getitem_172: "bf16[128, 901, 512]" = split_28[0]
        getitem_173: "bf16[128, 901, 512]" = split_28[1];  split_28 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:149 in forward, code: return self.down_proj(F.silu(gate) * up)
        convert_element_type_644: "f32[128, 901, 512]" = torch.ops.prims.convert_element_type.default(getitem_172, torch.float32);  getitem_172 = None
        sigmoid_28: "f32[128, 901, 512]" = torch.ops.aten.sigmoid.default(convert_element_type_644)
        mul_232: "f32[128, 901, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_644, sigmoid_28);  convert_element_type_644 = sigmoid_28 = None
        convert_element_type_645: "bf16[128, 901, 512]" = torch.ops.prims.convert_element_type.default(mul_232, torch.bfloat16);  mul_232 = None
        mul_233: "bf16[128, 901, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_645, getitem_173);  convert_element_type_645 = getitem_173 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_297: "bf16[115328, 512]" = torch.ops.aten.reshape.default(mul_233, [115328, 512]);  mul_233 = None
        mm_116: "bf16[115328, 128]" = torch.ops.aten.mm.default(view_297, permute_101);  view_297 = None
        view_298: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(mm_116, [128, 901, 128]);  mm_116 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:82 in forward, code: hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)
        add_187: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(convert_element_type_640, view_298);  convert_element_type_640 = view_298 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:154 in rms_norm, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_649: "f32[128, 901, 128]" = torch.ops.prims.convert_element_type.default(add_187, torch.float32);  add_187 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:156 in rms_norm, code: variance = hidden_states.square().mean(-1, keepdim=True)
        pow_58: "f32[128, 901, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_649, 2)
        mean_57: "f32[128, 901, 1]" = torch.ops.aten.mean.dim(pow_58, [-1], True);  pow_58 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:157 in rms_norm, code: hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        add_188: "f32[128, 901, 1]" = torch.ops.aten.add.Tensor(mean_57, 1e-05);  mean_57 = None
        rsqrt_57: "f32[128, 901, 1]" = torch.ops.aten.rsqrt.default(add_188);  add_188 = None
        mul_234: "f32[128, 901, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_649, rsqrt_57);  convert_element_type_649 = rsqrt_57 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:158 in rms_norm, code: return hidden_states.to(input_dtype)
        convert_element_type_650: "bf16[128, 901, 128]" = torch.ops.prims.convert_element_type.default(mul_234, torch.bfloat16);  mul_234 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_299: "bf16[115328, 128]" = torch.ops.aten.reshape.default(convert_element_type_650, [115328, 128])
        mm_117: "bf16[115328, 384]" = torch.ops.aten.mm.default(view_299, permute_102);  view_299 = None
        view_300: "bf16[128, 901, 384]" = torch.ops.aten.reshape.default(mm_117, [128, 901, 384]);  mm_117 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:119 in forward, code: qkv = qkv.view(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        view_301: "bf16[128, 901, 6, 64]" = torch.ops.aten.reshape.default(view_300, [128, 901, 6, 64]);  view_300 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:120 in forward, code: query = qkv[:, :, :self.num_heads]
        slice_383: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_301, 2, 0, 2)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:121 in forward, code: key = qkv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
        slice_386: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_301, 2, 2, 4)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:122 in forward, code: value = qkv[:, :, self.num_heads + self.num_key_value_heads:]
        slice_389: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_301, 2, 4, 9223372036854775807);  view_301 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:34 in apply_rotary_pos_emb, code: q = q.to(cos.dtype)
        convert_element_type_654: "f32[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(slice_383, torch.float32);  slice_383 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:35 in apply_rotary_pos_emb, code: k = k.to(cos.dtype)
        convert_element_type_655: "f32[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(slice_386, torch.float32);  slice_386 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:37 in apply_rotary_pos_emb, code: q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
        mul_235: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_654, unsqueeze_96)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:25 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_390: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_654, 3, 0, 32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:26 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_391: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_654, 3, 32, 9223372036854775807);  convert_element_type_654 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:27 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_58: "f32[128, 901, 2, 32]" = torch.ops.aten.neg.default(slice_391);  slice_391 = None
        cat_60: "f32[128, 901, 2, 64]" = torch.ops.aten.cat.default([neg_58, slice_390], -1);  neg_58 = slice_390 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:37 in apply_rotary_pos_emb, code: q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
        mul_236: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(cat_60, unsqueeze_97);  cat_60 = None
        add_189: "f32[128, 901, 2, 64]" = torch.ops.aten.add.Tensor(mul_235, mul_236);  mul_235 = mul_236 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:38 in apply_rotary_pos_emb, code: k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
        mul_237: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_655, unsqueeze_96)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:25 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_392: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_655, 3, 0, 32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:26 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_393: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_655, 3, 32, 9223372036854775807);  convert_element_type_655 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:27 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_59: "f32[128, 901, 2, 32]" = torch.ops.aten.neg.default(slice_393);  slice_393 = None
        cat_61: "f32[128, 901, 2, 64]" = torch.ops.aten.cat.default([neg_59, slice_392], -1);  neg_59 = slice_392 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:38 in apply_rotary_pos_emb, code: k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
        mul_238: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(cat_61, unsqueeze_97);  cat_61 = None
        add_190: "f32[128, 901, 2, 64]" = torch.ops.aten.add.Tensor(mul_237, mul_238);  mul_237 = mul_238 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:40 in apply_rotary_pos_emb, code: return q_embed.to(orig_dtype), k_embed.to(orig_dtype)
        convert_element_type_656: "bf16[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(add_189, torch.bfloat16);  add_189 = None
        convert_element_type_657: "bf16[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(add_190, torch.bfloat16);  add_190 = None
        
         # File: C:\Users\arroy\anaconda3\envs\hrm_v1\lib\site-packages\flash_attn\flash_attn_interface.py:839 in forward, code: out_padded, softmax_lse, S_dmask, rng_state = _wrapped_flash_attn_forward(
        _flash_attn_forward_29 = torch.ops.flash_attn._flash_attn_forward.default(convert_element_type_656, convert_element_type_657, slice_389, 0.0, 0.125, False, -1, -1, 0.0, None, False);  convert_element_type_656 = convert_element_type_657 = slice_389 = None
        getitem_174: "bf16[128, 901, 2, 64]" = _flash_attn_forward_29[0];  _flash_attn_forward_29 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:135 in forward, code: attn_output = attn_output.view(batch_size, seq_len, self.output_size)  # type: ignore
        view_302: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(getitem_174, [128, 901, 128]);  getitem_174 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_303: "bf16[115328, 128]" = torch.ops.aten.reshape.default(view_302, [115328, 128]);  view_302 = None
        mm_118: "bf16[115328, 128]" = torch.ops.aten.mm.default(view_303, permute_103);  view_303 = None
        view_304: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(mm_118, [128, 901, 128]);  mm_118 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:80 in forward, code: hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        add_191: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(convert_element_type_650, view_304);  convert_element_type_650 = view_304 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:154 in rms_norm, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_661: "f32[128, 901, 128]" = torch.ops.prims.convert_element_type.default(add_191, torch.float32);  add_191 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:156 in rms_norm, code: variance = hidden_states.square().mean(-1, keepdim=True)
        pow_59: "f32[128, 901, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_661, 2)
        mean_58: "f32[128, 901, 1]" = torch.ops.aten.mean.dim(pow_59, [-1], True);  pow_59 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:157 in rms_norm, code: hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        add_192: "f32[128, 901, 1]" = torch.ops.aten.add.Tensor(mean_58, 1e-05);  mean_58 = None
        rsqrt_58: "f32[128, 901, 1]" = torch.ops.aten.rsqrt.default(add_192);  add_192 = None
        mul_239: "f32[128, 901, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_661, rsqrt_58);  convert_element_type_661 = rsqrt_58 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:158 in rms_norm, code: return hidden_states.to(input_dtype)
        convert_element_type_662: "bf16[128, 901, 128]" = torch.ops.prims.convert_element_type.default(mul_239, torch.bfloat16);  mul_239 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_305: "bf16[115328, 128]" = torch.ops.aten.reshape.default(convert_element_type_662, [115328, 128])
        mm_119: "bf16[115328, 1024]" = torch.ops.aten.mm.default(view_305, permute_104);  view_305 = None
        view_306: "bf16[128, 901, 1024]" = torch.ops.aten.reshape.default(mm_119, [128, 901, 1024]);  mm_119 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:148 in forward, code: gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        split_29 = torch.ops.aten.split.Tensor(view_306, 512, -1);  view_306 = None
        getitem_178: "bf16[128, 901, 512]" = split_29[0]
        getitem_179: "bf16[128, 901, 512]" = split_29[1];  split_29 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:149 in forward, code: return self.down_proj(F.silu(gate) * up)
        convert_element_type_666: "f32[128, 901, 512]" = torch.ops.prims.convert_element_type.default(getitem_178, torch.float32);  getitem_178 = None
        sigmoid_29: "f32[128, 901, 512]" = torch.ops.aten.sigmoid.default(convert_element_type_666)
        mul_240: "f32[128, 901, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_666, sigmoid_29);  convert_element_type_666 = sigmoid_29 = None
        convert_element_type_667: "bf16[128, 901, 512]" = torch.ops.prims.convert_element_type.default(mul_240, torch.bfloat16);  mul_240 = None
        mul_241: "bf16[128, 901, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_667, getitem_179);  convert_element_type_667 = getitem_179 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_307: "bf16[115328, 512]" = torch.ops.aten.reshape.default(mul_241, [115328, 512]);  mul_241 = None
        mm_120: "bf16[115328, 128]" = torch.ops.aten.mm.default(view_307, permute_105);  view_307 = None
        view_308: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(mm_120, [128, 901, 128]);  mm_120 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:82 in forward, code: hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)
        add_193: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(convert_element_type_662, view_308);  convert_element_type_662 = view_308 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:154 in rms_norm, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_671: "f32[128, 901, 128]" = torch.ops.prims.convert_element_type.default(add_193, torch.float32);  add_193 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:156 in rms_norm, code: variance = hidden_states.square().mean(-1, keepdim=True)
        pow_60: "f32[128, 901, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_671, 2)
        mean_59: "f32[128, 901, 1]" = torch.ops.aten.mean.dim(pow_60, [-1], True);  pow_60 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:157 in rms_norm, code: hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        add_194: "f32[128, 901, 1]" = torch.ops.aten.add.Tensor(mean_59, 1e-05);  mean_59 = None
        rsqrt_59: "f32[128, 901, 1]" = torch.ops.aten.rsqrt.default(add_194);  add_194 = None
        mul_242: "f32[128, 901, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_671, rsqrt_59);  convert_element_type_671 = rsqrt_59 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:158 in rms_norm, code: return hidden_states.to(input_dtype)
        convert_element_type_672: "bf16[128, 901, 128]" = torch.ops.prims.convert_element_type.default(mul_242, torch.bfloat16);  mul_242 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_309: "bf16[115328, 128]" = torch.ops.aten.reshape.default(convert_element_type_672, [115328, 128])
        mm_121: "bf16[115328, 384]" = torch.ops.aten.mm.default(view_309, permute_106);  view_309 = None
        view_310: "bf16[128, 901, 384]" = torch.ops.aten.reshape.default(mm_121, [128, 901, 384]);  mm_121 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:119 in forward, code: qkv = qkv.view(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        view_311: "bf16[128, 901, 6, 64]" = torch.ops.aten.reshape.default(view_310, [128, 901, 6, 64]);  view_310 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:120 in forward, code: query = qkv[:, :, :self.num_heads]
        slice_396: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_311, 2, 0, 2)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:121 in forward, code: key = qkv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
        slice_399: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_311, 2, 2, 4)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:122 in forward, code: value = qkv[:, :, self.num_heads + self.num_key_value_heads:]
        slice_402: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_311, 2, 4, 9223372036854775807);  view_311 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:34 in apply_rotary_pos_emb, code: q = q.to(cos.dtype)
        convert_element_type_676: "f32[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(slice_396, torch.float32);  slice_396 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:35 in apply_rotary_pos_emb, code: k = k.to(cos.dtype)
        convert_element_type_677: "f32[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(slice_399, torch.float32);  slice_399 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:37 in apply_rotary_pos_emb, code: q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
        mul_243: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_676, unsqueeze_96)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:25 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_403: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_676, 3, 0, 32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:26 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_404: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_676, 3, 32, 9223372036854775807);  convert_element_type_676 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:27 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_60: "f32[128, 901, 2, 32]" = torch.ops.aten.neg.default(slice_404);  slice_404 = None
        cat_62: "f32[128, 901, 2, 64]" = torch.ops.aten.cat.default([neg_60, slice_403], -1);  neg_60 = slice_403 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:37 in apply_rotary_pos_emb, code: q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
        mul_244: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(cat_62, unsqueeze_97);  cat_62 = None
        add_195: "f32[128, 901, 2, 64]" = torch.ops.aten.add.Tensor(mul_243, mul_244);  mul_243 = mul_244 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:38 in apply_rotary_pos_emb, code: k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
        mul_245: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_677, unsqueeze_96)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:25 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_405: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_677, 3, 0, 32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:26 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_406: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_677, 3, 32, 9223372036854775807);  convert_element_type_677 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:27 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_61: "f32[128, 901, 2, 32]" = torch.ops.aten.neg.default(slice_406);  slice_406 = None
        cat_63: "f32[128, 901, 2, 64]" = torch.ops.aten.cat.default([neg_61, slice_405], -1);  neg_61 = slice_405 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:38 in apply_rotary_pos_emb, code: k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
        mul_246: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(cat_63, unsqueeze_97);  cat_63 = None
        add_196: "f32[128, 901, 2, 64]" = torch.ops.aten.add.Tensor(mul_245, mul_246);  mul_245 = mul_246 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:40 in apply_rotary_pos_emb, code: return q_embed.to(orig_dtype), k_embed.to(orig_dtype)
        convert_element_type_678: "bf16[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(add_195, torch.bfloat16);  add_195 = None
        convert_element_type_679: "bf16[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(add_196, torch.bfloat16);  add_196 = None
        
         # File: C:\Users\arroy\anaconda3\envs\hrm_v1\lib\site-packages\flash_attn\flash_attn_interface.py:839 in forward, code: out_padded, softmax_lse, S_dmask, rng_state = _wrapped_flash_attn_forward(
        _flash_attn_forward_30 = torch.ops.flash_attn._flash_attn_forward.default(convert_element_type_678, convert_element_type_679, slice_402, 0.0, 0.125, False, -1, -1, 0.0, None, False);  convert_element_type_678 = convert_element_type_679 = slice_402 = None
        getitem_180: "bf16[128, 901, 2, 64]" = _flash_attn_forward_30[0];  _flash_attn_forward_30 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:135 in forward, code: attn_output = attn_output.view(batch_size, seq_len, self.output_size)  # type: ignore
        view_312: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(getitem_180, [128, 901, 128]);  getitem_180 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_313: "bf16[115328, 128]" = torch.ops.aten.reshape.default(view_312, [115328, 128]);  view_312 = None
        mm_122: "bf16[115328, 128]" = torch.ops.aten.mm.default(view_313, permute_107);  view_313 = None
        view_314: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(mm_122, [128, 901, 128]);  mm_122 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:80 in forward, code: hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        add_197: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(convert_element_type_672, view_314);  convert_element_type_672 = view_314 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:154 in rms_norm, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_683: "f32[128, 901, 128]" = torch.ops.prims.convert_element_type.default(add_197, torch.float32);  add_197 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:156 in rms_norm, code: variance = hidden_states.square().mean(-1, keepdim=True)
        pow_61: "f32[128, 901, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_683, 2)
        mean_60: "f32[128, 901, 1]" = torch.ops.aten.mean.dim(pow_61, [-1], True);  pow_61 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:157 in rms_norm, code: hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        add_198: "f32[128, 901, 1]" = torch.ops.aten.add.Tensor(mean_60, 1e-05);  mean_60 = None
        rsqrt_60: "f32[128, 901, 1]" = torch.ops.aten.rsqrt.default(add_198);  add_198 = None
        mul_247: "f32[128, 901, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_683, rsqrt_60);  convert_element_type_683 = rsqrt_60 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:158 in rms_norm, code: return hidden_states.to(input_dtype)
        convert_element_type_684: "bf16[128, 901, 128]" = torch.ops.prims.convert_element_type.default(mul_247, torch.bfloat16);  mul_247 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_315: "bf16[115328, 128]" = torch.ops.aten.reshape.default(convert_element_type_684, [115328, 128])
        mm_123: "bf16[115328, 1024]" = torch.ops.aten.mm.default(view_315, permute_108);  view_315 = None
        view_316: "bf16[128, 901, 1024]" = torch.ops.aten.reshape.default(mm_123, [128, 901, 1024]);  mm_123 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:148 in forward, code: gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        split_30 = torch.ops.aten.split.Tensor(view_316, 512, -1);  view_316 = None
        getitem_184: "bf16[128, 901, 512]" = split_30[0]
        getitem_185: "bf16[128, 901, 512]" = split_30[1];  split_30 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:149 in forward, code: return self.down_proj(F.silu(gate) * up)
        convert_element_type_688: "f32[128, 901, 512]" = torch.ops.prims.convert_element_type.default(getitem_184, torch.float32);  getitem_184 = None
        sigmoid_30: "f32[128, 901, 512]" = torch.ops.aten.sigmoid.default(convert_element_type_688)
        mul_248: "f32[128, 901, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_688, sigmoid_30);  convert_element_type_688 = sigmoid_30 = None
        convert_element_type_689: "bf16[128, 901, 512]" = torch.ops.prims.convert_element_type.default(mul_248, torch.bfloat16);  mul_248 = None
        mul_249: "bf16[128, 901, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_689, getitem_185);  convert_element_type_689 = getitem_185 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_317: "bf16[115328, 512]" = torch.ops.aten.reshape.default(mul_249, [115328, 512]);  mul_249 = None
        mm_124: "bf16[115328, 128]" = torch.ops.aten.mm.default(view_317, permute_109);  view_317 = None
        view_318: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(mm_124, [128, 901, 128]);  mm_124 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:82 in forward, code: hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)
        add_199: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(convert_element_type_684, view_318);  convert_element_type_684 = view_318 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:154 in rms_norm, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_693: "f32[128, 901, 128]" = torch.ops.prims.convert_element_type.default(add_199, torch.float32);  add_199 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:156 in rms_norm, code: variance = hidden_states.square().mean(-1, keepdim=True)
        pow_62: "f32[128, 901, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_693, 2)
        mean_61: "f32[128, 901, 1]" = torch.ops.aten.mean.dim(pow_62, [-1], True);  pow_62 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:157 in rms_norm, code: hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        add_200: "f32[128, 901, 1]" = torch.ops.aten.add.Tensor(mean_61, 1e-05);  mean_61 = None
        rsqrt_61: "f32[128, 901, 1]" = torch.ops.aten.rsqrt.default(add_200);  add_200 = None
        mul_250: "f32[128, 901, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_693, rsqrt_61);  convert_element_type_693 = rsqrt_61 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:158 in rms_norm, code: return hidden_states.to(input_dtype)
        convert_element_type_694: "bf16[128, 901, 128]" = torch.ops.prims.convert_element_type.default(mul_250, torch.bfloat16);  mul_250 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_319: "bf16[115328, 128]" = torch.ops.aten.reshape.default(convert_element_type_694, [115328, 128])
        mm_125: "bf16[115328, 384]" = torch.ops.aten.mm.default(view_319, permute_110);  view_319 = None
        view_320: "bf16[128, 901, 384]" = torch.ops.aten.reshape.default(mm_125, [128, 901, 384]);  mm_125 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:119 in forward, code: qkv = qkv.view(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        view_321: "bf16[128, 901, 6, 64]" = torch.ops.aten.reshape.default(view_320, [128, 901, 6, 64]);  view_320 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:120 in forward, code: query = qkv[:, :, :self.num_heads]
        slice_409: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_321, 2, 0, 2)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:121 in forward, code: key = qkv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
        slice_412: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_321, 2, 2, 4)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:122 in forward, code: value = qkv[:, :, self.num_heads + self.num_key_value_heads:]
        slice_415: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_321, 2, 4, 9223372036854775807);  view_321 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:34 in apply_rotary_pos_emb, code: q = q.to(cos.dtype)
        convert_element_type_698: "f32[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(slice_409, torch.float32);  slice_409 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:35 in apply_rotary_pos_emb, code: k = k.to(cos.dtype)
        convert_element_type_699: "f32[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(slice_412, torch.float32);  slice_412 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:37 in apply_rotary_pos_emb, code: q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
        mul_251: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_698, unsqueeze_96)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:25 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_416: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_698, 3, 0, 32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:26 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_417: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_698, 3, 32, 9223372036854775807);  convert_element_type_698 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:27 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_62: "f32[128, 901, 2, 32]" = torch.ops.aten.neg.default(slice_417);  slice_417 = None
        cat_64: "f32[128, 901, 2, 64]" = torch.ops.aten.cat.default([neg_62, slice_416], -1);  neg_62 = slice_416 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:37 in apply_rotary_pos_emb, code: q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
        mul_252: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(cat_64, unsqueeze_97);  cat_64 = None
        add_201: "f32[128, 901, 2, 64]" = torch.ops.aten.add.Tensor(mul_251, mul_252);  mul_251 = mul_252 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:38 in apply_rotary_pos_emb, code: k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
        mul_253: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_699, unsqueeze_96)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:25 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_418: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_699, 3, 0, 32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:26 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_419: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_699, 3, 32, 9223372036854775807);  convert_element_type_699 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:27 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_63: "f32[128, 901, 2, 32]" = torch.ops.aten.neg.default(slice_419);  slice_419 = None
        cat_65: "f32[128, 901, 2, 64]" = torch.ops.aten.cat.default([neg_63, slice_418], -1);  neg_63 = slice_418 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:38 in apply_rotary_pos_emb, code: k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
        mul_254: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(cat_65, unsqueeze_97);  cat_65 = None
        add_202: "f32[128, 901, 2, 64]" = torch.ops.aten.add.Tensor(mul_253, mul_254);  mul_253 = mul_254 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:40 in apply_rotary_pos_emb, code: return q_embed.to(orig_dtype), k_embed.to(orig_dtype)
        convert_element_type_700: "bf16[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(add_201, torch.bfloat16);  add_201 = None
        convert_element_type_701: "bf16[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(add_202, torch.bfloat16);  add_202 = None
        
         # File: C:\Users\arroy\anaconda3\envs\hrm_v1\lib\site-packages\flash_attn\flash_attn_interface.py:839 in forward, code: out_padded, softmax_lse, S_dmask, rng_state = _wrapped_flash_attn_forward(
        _flash_attn_forward_31 = torch.ops.flash_attn._flash_attn_forward.default(convert_element_type_700, convert_element_type_701, slice_415, 0.0, 0.125, False, -1, -1, 0.0, None, False);  convert_element_type_700 = convert_element_type_701 = slice_415 = None
        getitem_186: "bf16[128, 901, 2, 64]" = _flash_attn_forward_31[0];  _flash_attn_forward_31 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:135 in forward, code: attn_output = attn_output.view(batch_size, seq_len, self.output_size)  # type: ignore
        view_322: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(getitem_186, [128, 901, 128]);  getitem_186 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_323: "bf16[115328, 128]" = torch.ops.aten.reshape.default(view_322, [115328, 128]);  view_322 = None
        mm_126: "bf16[115328, 128]" = torch.ops.aten.mm.default(view_323, permute_111);  view_323 = None
        view_324: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(mm_126, [128, 901, 128]);  mm_126 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:80 in forward, code: hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        add_203: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(convert_element_type_694, view_324);  convert_element_type_694 = view_324 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:154 in rms_norm, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_705: "f32[128, 901, 128]" = torch.ops.prims.convert_element_type.default(add_203, torch.float32);  add_203 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:156 in rms_norm, code: variance = hidden_states.square().mean(-1, keepdim=True)
        pow_63: "f32[128, 901, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_705, 2)
        mean_62: "f32[128, 901, 1]" = torch.ops.aten.mean.dim(pow_63, [-1], True);  pow_63 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:157 in rms_norm, code: hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        add_204: "f32[128, 901, 1]" = torch.ops.aten.add.Tensor(mean_62, 1e-05);  mean_62 = None
        rsqrt_62: "f32[128, 901, 1]" = torch.ops.aten.rsqrt.default(add_204);  add_204 = None
        mul_255: "f32[128, 901, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_705, rsqrt_62);  convert_element_type_705 = rsqrt_62 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:158 in rms_norm, code: return hidden_states.to(input_dtype)
        convert_element_type_706: "bf16[128, 901, 128]" = torch.ops.prims.convert_element_type.default(mul_255, torch.bfloat16);  mul_255 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_325: "bf16[115328, 128]" = torch.ops.aten.reshape.default(convert_element_type_706, [115328, 128])
        mm_127: "bf16[115328, 1024]" = torch.ops.aten.mm.default(view_325, permute_112);  view_325 = None
        view_326: "bf16[128, 901, 1024]" = torch.ops.aten.reshape.default(mm_127, [128, 901, 1024]);  mm_127 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:148 in forward, code: gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        split_31 = torch.ops.aten.split.Tensor(view_326, 512, -1);  view_326 = None
        getitem_190: "bf16[128, 901, 512]" = split_31[0]
        getitem_191: "bf16[128, 901, 512]" = split_31[1];  split_31 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:149 in forward, code: return self.down_proj(F.silu(gate) * up)
        convert_element_type_710: "f32[128, 901, 512]" = torch.ops.prims.convert_element_type.default(getitem_190, torch.float32);  getitem_190 = None
        sigmoid_31: "f32[128, 901, 512]" = torch.ops.aten.sigmoid.default(convert_element_type_710)
        mul_256: "f32[128, 901, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_710, sigmoid_31);  convert_element_type_710 = sigmoid_31 = None
        convert_element_type_711: "bf16[128, 901, 512]" = torch.ops.prims.convert_element_type.default(mul_256, torch.bfloat16);  mul_256 = None
        mul_257: "bf16[128, 901, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_711, getitem_191);  convert_element_type_711 = getitem_191 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_327: "bf16[115328, 512]" = torch.ops.aten.reshape.default(mul_257, [115328, 512]);  mul_257 = None
        mm_128: "bf16[115328, 128]" = torch.ops.aten.mm.default(view_327, permute_113);  view_327 = None
        view_328: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(mm_128, [128, 901, 128]);  mm_128 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:82 in forward, code: hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)
        add_205: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(convert_element_type_706, view_328);  convert_element_type_706 = view_328 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:154 in rms_norm, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_715: "f32[128, 901, 128]" = torch.ops.prims.convert_element_type.default(add_205, torch.float32);  add_205 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:156 in rms_norm, code: variance = hidden_states.square().mean(-1, keepdim=True)
        pow_64: "f32[128, 901, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_715, 2)
        mean_63: "f32[128, 901, 1]" = torch.ops.aten.mean.dim(pow_64, [-1], True);  pow_64 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:157 in rms_norm, code: hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        add_206: "f32[128, 901, 1]" = torch.ops.aten.add.Tensor(mean_63, 1e-05);  mean_63 = None
        rsqrt_63: "f32[128, 901, 1]" = torch.ops.aten.rsqrt.default(add_206);  add_206 = None
        mul_258: "f32[128, 901, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_715, rsqrt_63);  convert_element_type_715 = rsqrt_63 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:158 in rms_norm, code: return hidden_states.to(input_dtype)
        convert_element_type_716: "bf16[128, 901, 128]" = torch.ops.prims.convert_element_type.default(mul_258, torch.bfloat16);  mul_258 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:94 in forward, code: hidden_states = hidden_states + input_injection
        add_207: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(convert_element_type_529, convert_element_type_716)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        convert_element_type_717: "bf16[384, 128]" = torch.ops.prims.convert_element_type.default(primals_35, torch.bfloat16);  primals_35 = None
        permute_130: "bf16[128, 384]" = torch.ops.aten.permute.default(convert_element_type_717, [1, 0]);  convert_element_type_717 = None
        view_329: "bf16[115328, 128]" = torch.ops.aten.reshape.default(add_207, [115328, 128])
        mm_129: "bf16[115328, 384]" = torch.ops.aten.mm.default(view_329, permute_130);  view_329 = None
        view_330: "bf16[128, 901, 384]" = torch.ops.aten.reshape.default(mm_129, [128, 901, 384]);  mm_129 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:119 in forward, code: qkv = qkv.view(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        view_331: "bf16[128, 901, 6, 64]" = torch.ops.aten.reshape.default(view_330, [128, 901, 6, 64]);  view_330 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:120 in forward, code: query = qkv[:, :, :self.num_heads]
        slice_422: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_331, 2, 0, 2)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:121 in forward, code: key = qkv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
        slice_425: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_331, 2, 2, 4)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:122 in forward, code: value = qkv[:, :, self.num_heads + self.num_key_value_heads:]
        slice_428: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_331, 2, 4, 9223372036854775807);  view_331 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:34 in apply_rotary_pos_emb, code: q = q.to(cos.dtype)
        convert_element_type_720: "f32[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(slice_422, torch.float32);  slice_422 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:35 in apply_rotary_pos_emb, code: k = k.to(cos.dtype)
        convert_element_type_721: "f32[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(slice_425, torch.float32);  slice_425 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:37 in apply_rotary_pos_emb, code: q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
        mul_259: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_720, unsqueeze_96)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:25 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_429: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_720, 3, 0, 32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:26 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_430: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_720, 3, 32, 9223372036854775807);  convert_element_type_720 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:27 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_64: "f32[128, 901, 2, 32]" = torch.ops.aten.neg.default(slice_430);  slice_430 = None
        cat_66: "f32[128, 901, 2, 64]" = torch.ops.aten.cat.default([neg_64, slice_429], -1);  neg_64 = slice_429 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:37 in apply_rotary_pos_emb, code: q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
        mul_260: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(cat_66, unsqueeze_97);  cat_66 = None
        add_208: "f32[128, 901, 2, 64]" = torch.ops.aten.add.Tensor(mul_259, mul_260);  mul_259 = mul_260 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:38 in apply_rotary_pos_emb, code: k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
        mul_261: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_721, unsqueeze_96)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:25 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_431: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_721, 3, 0, 32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:26 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_432: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_721, 3, 32, 9223372036854775807);  convert_element_type_721 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:27 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_65: "f32[128, 901, 2, 32]" = torch.ops.aten.neg.default(slice_432);  slice_432 = None
        cat_67: "f32[128, 901, 2, 64]" = torch.ops.aten.cat.default([neg_65, slice_431], -1);  neg_65 = slice_431 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:38 in apply_rotary_pos_emb, code: k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
        mul_262: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(cat_67, unsqueeze_97);  cat_67 = None
        add_209: "f32[128, 901, 2, 64]" = torch.ops.aten.add.Tensor(mul_261, mul_262);  mul_261 = mul_262 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:40 in apply_rotary_pos_emb, code: return q_embed.to(orig_dtype), k_embed.to(orig_dtype)
        convert_element_type_722: "bf16[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(add_208, torch.bfloat16);  add_208 = None
        convert_element_type_723: "bf16[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(add_209, torch.bfloat16);  add_209 = None
        
         # File: C:\Users\arroy\anaconda3\envs\hrm_v1\lib\site-packages\flash_attn\flash_attn_interface.py:839 in forward, code: out_padded, softmax_lse, S_dmask, rng_state = _wrapped_flash_attn_forward(
        _flash_attn_forward_32 = torch.ops.flash_attn._flash_attn_forward.default(convert_element_type_722, convert_element_type_723, slice_428, 0.0, 0.125, False, -1, -1, 0.0, None, False);  convert_element_type_722 = convert_element_type_723 = slice_428 = None
        getitem_192: "bf16[128, 901, 2, 64]" = _flash_attn_forward_32[0];  _flash_attn_forward_32 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:135 in forward, code: attn_output = attn_output.view(batch_size, seq_len, self.output_size)  # type: ignore
        view_332: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(getitem_192, [128, 901, 128]);  getitem_192 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        convert_element_type_724: "bf16[128, 128]" = torch.ops.prims.convert_element_type.default(primals_36, torch.bfloat16);  primals_36 = None
        permute_131: "bf16[128, 128]" = torch.ops.aten.permute.default(convert_element_type_724, [1, 0]);  convert_element_type_724 = None
        view_333: "bf16[115328, 128]" = torch.ops.aten.reshape.default(view_332, [115328, 128]);  view_332 = None
        mm_130: "bf16[115328, 128]" = torch.ops.aten.mm.default(view_333, permute_131);  view_333 = None
        view_334: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(mm_130, [128, 901, 128]);  mm_130 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:80 in forward, code: hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        add_210: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(add_207, view_334);  add_207 = view_334 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:154 in rms_norm, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_727: "f32[128, 901, 128]" = torch.ops.prims.convert_element_type.default(add_210, torch.float32);  add_210 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:156 in rms_norm, code: variance = hidden_states.square().mean(-1, keepdim=True)
        pow_65: "f32[128, 901, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_727, 2)
        mean_64: "f32[128, 901, 1]" = torch.ops.aten.mean.dim(pow_65, [-1], True);  pow_65 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:157 in rms_norm, code: hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        add_211: "f32[128, 901, 1]" = torch.ops.aten.add.Tensor(mean_64, 1e-05);  mean_64 = None
        rsqrt_64: "f32[128, 901, 1]" = torch.ops.aten.rsqrt.default(add_211);  add_211 = None
        mul_263: "f32[128, 901, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_727, rsqrt_64);  convert_element_type_727 = rsqrt_64 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:158 in rms_norm, code: return hidden_states.to(input_dtype)
        convert_element_type_728: "bf16[128, 901, 128]" = torch.ops.prims.convert_element_type.default(mul_263, torch.bfloat16);  mul_263 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        convert_element_type_729: "bf16[1024, 128]" = torch.ops.prims.convert_element_type.default(primals_37, torch.bfloat16);  primals_37 = None
        permute_132: "bf16[128, 1024]" = torch.ops.aten.permute.default(convert_element_type_729, [1, 0]);  convert_element_type_729 = None
        view_335: "bf16[115328, 128]" = torch.ops.aten.reshape.default(convert_element_type_728, [115328, 128])
        mm_131: "bf16[115328, 1024]" = torch.ops.aten.mm.default(view_335, permute_132);  view_335 = None
        view_336: "bf16[128, 901, 1024]" = torch.ops.aten.reshape.default(mm_131, [128, 901, 1024]);  mm_131 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:148 in forward, code: gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        split_32 = torch.ops.aten.split.Tensor(view_336, 512, -1);  view_336 = None
        getitem_196: "bf16[128, 901, 512]" = split_32[0]
        getitem_197: "bf16[128, 901, 512]" = split_32[1];  split_32 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:149 in forward, code: return self.down_proj(F.silu(gate) * up)
        convert_element_type_732: "f32[128, 901, 512]" = torch.ops.prims.convert_element_type.default(getitem_196, torch.float32);  getitem_196 = None
        sigmoid_32: "f32[128, 901, 512]" = torch.ops.aten.sigmoid.default(convert_element_type_732)
        mul_264: "f32[128, 901, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_732, sigmoid_32);  convert_element_type_732 = sigmoid_32 = None
        convert_element_type_733: "bf16[128, 901, 512]" = torch.ops.prims.convert_element_type.default(mul_264, torch.bfloat16);  mul_264 = None
        mul_265: "bf16[128, 901, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_733, getitem_197);  convert_element_type_733 = getitem_197 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        convert_element_type_734: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(primals_38, torch.bfloat16);  primals_38 = None
        permute_133: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_734, [1, 0]);  convert_element_type_734 = None
        view_337: "bf16[115328, 512]" = torch.ops.aten.reshape.default(mul_265, [115328, 512]);  mul_265 = None
        mm_132: "bf16[115328, 128]" = torch.ops.aten.mm.default(view_337, permute_133);  view_337 = None
        view_338: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(mm_132, [128, 901, 128]);  mm_132 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:82 in forward, code: hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)
        add_212: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(convert_element_type_728, view_338);  convert_element_type_728 = view_338 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:154 in rms_norm, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_737: "f32[128, 901, 128]" = torch.ops.prims.convert_element_type.default(add_212, torch.float32);  add_212 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:156 in rms_norm, code: variance = hidden_states.square().mean(-1, keepdim=True)
        pow_66: "f32[128, 901, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_737, 2)
        mean_65: "f32[128, 901, 1]" = torch.ops.aten.mean.dim(pow_66, [-1], True);  pow_66 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:157 in rms_norm, code: hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        add_213: "f32[128, 901, 1]" = torch.ops.aten.add.Tensor(mean_65, 1e-05);  mean_65 = None
        rsqrt_65: "f32[128, 901, 1]" = torch.ops.aten.rsqrt.default(add_213);  add_213 = None
        mul_266: "f32[128, 901, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_737, rsqrt_65);  convert_element_type_737 = rsqrt_65 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:158 in rms_norm, code: return hidden_states.to(input_dtype)
        convert_element_type_738: "bf16[128, 901, 128]" = torch.ops.prims.convert_element_type.default(mul_266, torch.bfloat16);  mul_266 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        convert_element_type_739: "bf16[384, 128]" = torch.ops.prims.convert_element_type.default(primals_39, torch.bfloat16);  primals_39 = None
        permute_134: "bf16[128, 384]" = torch.ops.aten.permute.default(convert_element_type_739, [1, 0]);  convert_element_type_739 = None
        view_339: "bf16[115328, 128]" = torch.ops.aten.reshape.default(convert_element_type_738, [115328, 128])
        mm_133: "bf16[115328, 384]" = torch.ops.aten.mm.default(view_339, permute_134);  view_339 = None
        view_340: "bf16[128, 901, 384]" = torch.ops.aten.reshape.default(mm_133, [128, 901, 384]);  mm_133 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:119 in forward, code: qkv = qkv.view(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        view_341: "bf16[128, 901, 6, 64]" = torch.ops.aten.reshape.default(view_340, [128, 901, 6, 64]);  view_340 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:120 in forward, code: query = qkv[:, :, :self.num_heads]
        slice_435: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_341, 2, 0, 2)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:121 in forward, code: key = qkv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
        slice_438: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_341, 2, 2, 4)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:122 in forward, code: value = qkv[:, :, self.num_heads + self.num_key_value_heads:]
        slice_441: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_341, 2, 4, 9223372036854775807);  view_341 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:34 in apply_rotary_pos_emb, code: q = q.to(cos.dtype)
        convert_element_type_742: "f32[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(slice_435, torch.float32);  slice_435 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:35 in apply_rotary_pos_emb, code: k = k.to(cos.dtype)
        convert_element_type_743: "f32[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(slice_438, torch.float32);  slice_438 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:37 in apply_rotary_pos_emb, code: q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
        mul_267: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_742, unsqueeze_96)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:25 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_442: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_742, 3, 0, 32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:26 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_443: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_742, 3, 32, 9223372036854775807);  convert_element_type_742 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:27 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_66: "f32[128, 901, 2, 32]" = torch.ops.aten.neg.default(slice_443);  slice_443 = None
        cat_68: "f32[128, 901, 2, 64]" = torch.ops.aten.cat.default([neg_66, slice_442], -1);  neg_66 = slice_442 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:37 in apply_rotary_pos_emb, code: q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
        mul_268: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(cat_68, unsqueeze_97);  cat_68 = None
        add_214: "f32[128, 901, 2, 64]" = torch.ops.aten.add.Tensor(mul_267, mul_268);  mul_267 = mul_268 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:38 in apply_rotary_pos_emb, code: k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
        mul_269: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_743, unsqueeze_96)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:25 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_444: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_743, 3, 0, 32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:26 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_445: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_743, 3, 32, 9223372036854775807);  convert_element_type_743 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:27 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_67: "f32[128, 901, 2, 32]" = torch.ops.aten.neg.default(slice_445);  slice_445 = None
        cat_69: "f32[128, 901, 2, 64]" = torch.ops.aten.cat.default([neg_67, slice_444], -1);  neg_67 = slice_444 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:38 in apply_rotary_pos_emb, code: k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
        mul_270: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(cat_69, unsqueeze_97);  cat_69 = None
        add_215: "f32[128, 901, 2, 64]" = torch.ops.aten.add.Tensor(mul_269, mul_270);  mul_269 = mul_270 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:40 in apply_rotary_pos_emb, code: return q_embed.to(orig_dtype), k_embed.to(orig_dtype)
        convert_element_type_744: "bf16[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(add_214, torch.bfloat16);  add_214 = None
        convert_element_type_745: "bf16[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(add_215, torch.bfloat16);  add_215 = None
        
         # File: C:\Users\arroy\anaconda3\envs\hrm_v1\lib\site-packages\flash_attn\flash_attn_interface.py:839 in forward, code: out_padded, softmax_lse, S_dmask, rng_state = _wrapped_flash_attn_forward(
        _flash_attn_forward_33 = torch.ops.flash_attn._flash_attn_forward.default(convert_element_type_744, convert_element_type_745, slice_441, 0.0, 0.125, False, -1, -1, 0.0, None, False);  convert_element_type_744 = convert_element_type_745 = slice_441 = None
        getitem_198: "bf16[128, 901, 2, 64]" = _flash_attn_forward_33[0];  _flash_attn_forward_33 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:135 in forward, code: attn_output = attn_output.view(batch_size, seq_len, self.output_size)  # type: ignore
        view_342: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(getitem_198, [128, 901, 128]);  getitem_198 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        convert_element_type_746: "bf16[128, 128]" = torch.ops.prims.convert_element_type.default(primals_40, torch.bfloat16);  primals_40 = None
        permute_135: "bf16[128, 128]" = torch.ops.aten.permute.default(convert_element_type_746, [1, 0]);  convert_element_type_746 = None
        view_343: "bf16[115328, 128]" = torch.ops.aten.reshape.default(view_342, [115328, 128]);  view_342 = None
        mm_134: "bf16[115328, 128]" = torch.ops.aten.mm.default(view_343, permute_135);  view_343 = None
        view_344: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(mm_134, [128, 901, 128]);  mm_134 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:80 in forward, code: hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        add_216: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(convert_element_type_738, view_344);  convert_element_type_738 = view_344 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:154 in rms_norm, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_749: "f32[128, 901, 128]" = torch.ops.prims.convert_element_type.default(add_216, torch.float32);  add_216 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:156 in rms_norm, code: variance = hidden_states.square().mean(-1, keepdim=True)
        pow_67: "f32[128, 901, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_749, 2)
        mean_66: "f32[128, 901, 1]" = torch.ops.aten.mean.dim(pow_67, [-1], True);  pow_67 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:157 in rms_norm, code: hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        add_217: "f32[128, 901, 1]" = torch.ops.aten.add.Tensor(mean_66, 1e-05);  mean_66 = None
        rsqrt_66: "f32[128, 901, 1]" = torch.ops.aten.rsqrt.default(add_217);  add_217 = None
        mul_271: "f32[128, 901, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_749, rsqrt_66);  convert_element_type_749 = rsqrt_66 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:158 in rms_norm, code: return hidden_states.to(input_dtype)
        convert_element_type_750: "bf16[128, 901, 128]" = torch.ops.prims.convert_element_type.default(mul_271, torch.bfloat16);  mul_271 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        convert_element_type_751: "bf16[1024, 128]" = torch.ops.prims.convert_element_type.default(primals_41, torch.bfloat16);  primals_41 = None
        permute_136: "bf16[128, 1024]" = torch.ops.aten.permute.default(convert_element_type_751, [1, 0]);  convert_element_type_751 = None
        view_345: "bf16[115328, 128]" = torch.ops.aten.reshape.default(convert_element_type_750, [115328, 128])
        mm_135: "bf16[115328, 1024]" = torch.ops.aten.mm.default(view_345, permute_136);  view_345 = None
        view_346: "bf16[128, 901, 1024]" = torch.ops.aten.reshape.default(mm_135, [128, 901, 1024]);  mm_135 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:148 in forward, code: gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        split_33 = torch.ops.aten.split.Tensor(view_346, 512, -1);  view_346 = None
        getitem_202: "bf16[128, 901, 512]" = split_33[0]
        getitem_203: "bf16[128, 901, 512]" = split_33[1];  split_33 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:149 in forward, code: return self.down_proj(F.silu(gate) * up)
        convert_element_type_754: "f32[128, 901, 512]" = torch.ops.prims.convert_element_type.default(getitem_202, torch.float32);  getitem_202 = None
        sigmoid_33: "f32[128, 901, 512]" = torch.ops.aten.sigmoid.default(convert_element_type_754)
        mul_272: "f32[128, 901, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_754, sigmoid_33);  convert_element_type_754 = sigmoid_33 = None
        convert_element_type_755: "bf16[128, 901, 512]" = torch.ops.prims.convert_element_type.default(mul_272, torch.bfloat16);  mul_272 = None
        mul_273: "bf16[128, 901, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_755, getitem_203);  convert_element_type_755 = getitem_203 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        convert_element_type_756: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(primals_42, torch.bfloat16);  primals_42 = None
        permute_137: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_756, [1, 0]);  convert_element_type_756 = None
        view_347: "bf16[115328, 512]" = torch.ops.aten.reshape.default(mul_273, [115328, 512]);  mul_273 = None
        mm_136: "bf16[115328, 128]" = torch.ops.aten.mm.default(view_347, permute_137);  view_347 = None
        view_348: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(mm_136, [128, 901, 128]);  mm_136 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:82 in forward, code: hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)
        add_218: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(convert_element_type_750, view_348);  convert_element_type_750 = view_348 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:154 in rms_norm, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_759: "f32[128, 901, 128]" = torch.ops.prims.convert_element_type.default(add_218, torch.float32);  add_218 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:156 in rms_norm, code: variance = hidden_states.square().mean(-1, keepdim=True)
        pow_68: "f32[128, 901, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_759, 2)
        mean_67: "f32[128, 901, 1]" = torch.ops.aten.mean.dim(pow_68, [-1], True);  pow_68 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:157 in rms_norm, code: hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        add_219: "f32[128, 901, 1]" = torch.ops.aten.add.Tensor(mean_67, 1e-05);  mean_67 = None
        rsqrt_67: "f32[128, 901, 1]" = torch.ops.aten.rsqrt.default(add_219);  add_219 = None
        mul_274: "f32[128, 901, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_759, rsqrt_67);  convert_element_type_759 = rsqrt_67 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:158 in rms_norm, code: return hidden_states.to(input_dtype)
        convert_element_type_760: "bf16[128, 901, 128]" = torch.ops.prims.convert_element_type.default(mul_274, torch.bfloat16);  mul_274 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        convert_element_type_761: "bf16[384, 128]" = torch.ops.prims.convert_element_type.default(primals_43, torch.bfloat16);  primals_43 = None
        permute_138: "bf16[128, 384]" = torch.ops.aten.permute.default(convert_element_type_761, [1, 0]);  convert_element_type_761 = None
        view_349: "bf16[115328, 128]" = torch.ops.aten.reshape.default(convert_element_type_760, [115328, 128])
        mm_137: "bf16[115328, 384]" = torch.ops.aten.mm.default(view_349, permute_138);  view_349 = None
        view_350: "bf16[128, 901, 384]" = torch.ops.aten.reshape.default(mm_137, [128, 901, 384]);  mm_137 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:119 in forward, code: qkv = qkv.view(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        view_351: "bf16[128, 901, 6, 64]" = torch.ops.aten.reshape.default(view_350, [128, 901, 6, 64]);  view_350 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:120 in forward, code: query = qkv[:, :, :self.num_heads]
        slice_448: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_351, 2, 0, 2)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:121 in forward, code: key = qkv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
        slice_451: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_351, 2, 2, 4)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:122 in forward, code: value = qkv[:, :, self.num_heads + self.num_key_value_heads:]
        slice_454: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_351, 2, 4, 9223372036854775807);  view_351 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:34 in apply_rotary_pos_emb, code: q = q.to(cos.dtype)
        convert_element_type_764: "f32[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(slice_448, torch.float32);  slice_448 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:35 in apply_rotary_pos_emb, code: k = k.to(cos.dtype)
        convert_element_type_765: "f32[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(slice_451, torch.float32);  slice_451 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:37 in apply_rotary_pos_emb, code: q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
        mul_275: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_764, unsqueeze_96)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:25 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_455: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_764, 3, 0, 32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:26 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_456: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_764, 3, 32, 9223372036854775807);  convert_element_type_764 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:27 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_68: "f32[128, 901, 2, 32]" = torch.ops.aten.neg.default(slice_456);  slice_456 = None
        cat_70: "f32[128, 901, 2, 64]" = torch.ops.aten.cat.default([neg_68, slice_455], -1);  neg_68 = slice_455 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:37 in apply_rotary_pos_emb, code: q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
        mul_276: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(cat_70, unsqueeze_97);  cat_70 = None
        add_220: "f32[128, 901, 2, 64]" = torch.ops.aten.add.Tensor(mul_275, mul_276);  mul_275 = mul_276 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:38 in apply_rotary_pos_emb, code: k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
        mul_277: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_765, unsqueeze_96)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:25 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_457: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_765, 3, 0, 32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:26 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_458: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_765, 3, 32, 9223372036854775807);  convert_element_type_765 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:27 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_69: "f32[128, 901, 2, 32]" = torch.ops.aten.neg.default(slice_458);  slice_458 = None
        cat_71: "f32[128, 901, 2, 64]" = torch.ops.aten.cat.default([neg_69, slice_457], -1);  neg_69 = slice_457 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:38 in apply_rotary_pos_emb, code: k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
        mul_278: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(cat_71, unsqueeze_97);  cat_71 = None
        add_221: "f32[128, 901, 2, 64]" = torch.ops.aten.add.Tensor(mul_277, mul_278);  mul_277 = mul_278 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:40 in apply_rotary_pos_emb, code: return q_embed.to(orig_dtype), k_embed.to(orig_dtype)
        convert_element_type_766: "bf16[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(add_220, torch.bfloat16);  add_220 = None
        convert_element_type_767: "bf16[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(add_221, torch.bfloat16);  add_221 = None
        
         # File: C:\Users\arroy\anaconda3\envs\hrm_v1\lib\site-packages\flash_attn\flash_attn_interface.py:839 in forward, code: out_padded, softmax_lse, S_dmask, rng_state = _wrapped_flash_attn_forward(
        _flash_attn_forward_34 = torch.ops.flash_attn._flash_attn_forward.default(convert_element_type_766, convert_element_type_767, slice_454, 0.0, 0.125, False, -1, -1, 0.0, None, False);  convert_element_type_766 = convert_element_type_767 = slice_454 = None
        getitem_204: "bf16[128, 901, 2, 64]" = _flash_attn_forward_34[0];  _flash_attn_forward_34 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:135 in forward, code: attn_output = attn_output.view(batch_size, seq_len, self.output_size)  # type: ignore
        view_352: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(getitem_204, [128, 901, 128]);  getitem_204 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        convert_element_type_768: "bf16[128, 128]" = torch.ops.prims.convert_element_type.default(primals_44, torch.bfloat16);  primals_44 = None
        permute_139: "bf16[128, 128]" = torch.ops.aten.permute.default(convert_element_type_768, [1, 0]);  convert_element_type_768 = None
        view_353: "bf16[115328, 128]" = torch.ops.aten.reshape.default(view_352, [115328, 128]);  view_352 = None
        mm_138: "bf16[115328, 128]" = torch.ops.aten.mm.default(view_353, permute_139);  view_353 = None
        view_354: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(mm_138, [128, 901, 128]);  mm_138 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:80 in forward, code: hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        add_222: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(convert_element_type_760, view_354);  convert_element_type_760 = view_354 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:154 in rms_norm, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_771: "f32[128, 901, 128]" = torch.ops.prims.convert_element_type.default(add_222, torch.float32);  add_222 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:156 in rms_norm, code: variance = hidden_states.square().mean(-1, keepdim=True)
        pow_69: "f32[128, 901, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_771, 2)
        mean_68: "f32[128, 901, 1]" = torch.ops.aten.mean.dim(pow_69, [-1], True);  pow_69 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:157 in rms_norm, code: hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        add_223: "f32[128, 901, 1]" = torch.ops.aten.add.Tensor(mean_68, 1e-05);  mean_68 = None
        rsqrt_68: "f32[128, 901, 1]" = torch.ops.aten.rsqrt.default(add_223);  add_223 = None
        mul_279: "f32[128, 901, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_771, rsqrt_68);  convert_element_type_771 = rsqrt_68 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:158 in rms_norm, code: return hidden_states.to(input_dtype)
        convert_element_type_772: "bf16[128, 901, 128]" = torch.ops.prims.convert_element_type.default(mul_279, torch.bfloat16);  mul_279 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        convert_element_type_773: "bf16[1024, 128]" = torch.ops.prims.convert_element_type.default(primals_45, torch.bfloat16);  primals_45 = None
        permute_140: "bf16[128, 1024]" = torch.ops.aten.permute.default(convert_element_type_773, [1, 0]);  convert_element_type_773 = None
        view_355: "bf16[115328, 128]" = torch.ops.aten.reshape.default(convert_element_type_772, [115328, 128])
        mm_139: "bf16[115328, 1024]" = torch.ops.aten.mm.default(view_355, permute_140);  view_355 = None
        view_356: "bf16[128, 901, 1024]" = torch.ops.aten.reshape.default(mm_139, [128, 901, 1024]);  mm_139 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:148 in forward, code: gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        split_34 = torch.ops.aten.split.Tensor(view_356, 512, -1);  view_356 = None
        getitem_208: "bf16[128, 901, 512]" = split_34[0]
        getitem_209: "bf16[128, 901, 512]" = split_34[1];  split_34 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:149 in forward, code: return self.down_proj(F.silu(gate) * up)
        convert_element_type_776: "f32[128, 901, 512]" = torch.ops.prims.convert_element_type.default(getitem_208, torch.float32);  getitem_208 = None
        sigmoid_34: "f32[128, 901, 512]" = torch.ops.aten.sigmoid.default(convert_element_type_776)
        mul_280: "f32[128, 901, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_776, sigmoid_34);  convert_element_type_776 = sigmoid_34 = None
        convert_element_type_777: "bf16[128, 901, 512]" = torch.ops.prims.convert_element_type.default(mul_280, torch.bfloat16);  mul_280 = None
        mul_281: "bf16[128, 901, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_777, getitem_209);  convert_element_type_777 = getitem_209 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        convert_element_type_778: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(primals_46, torch.bfloat16);  primals_46 = None
        permute_141: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_778, [1, 0]);  convert_element_type_778 = None
        view_357: "bf16[115328, 512]" = torch.ops.aten.reshape.default(mul_281, [115328, 512]);  mul_281 = None
        mm_140: "bf16[115328, 128]" = torch.ops.aten.mm.default(view_357, permute_141);  view_357 = None
        view_358: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(mm_140, [128, 901, 128]);  mm_140 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:82 in forward, code: hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)
        add_224: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(convert_element_type_772, view_358);  convert_element_type_772 = view_358 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:154 in rms_norm, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_781: "f32[128, 901, 128]" = torch.ops.prims.convert_element_type.default(add_224, torch.float32);  add_224 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:156 in rms_norm, code: variance = hidden_states.square().mean(-1, keepdim=True)
        pow_70: "f32[128, 901, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_781, 2)
        mean_69: "f32[128, 901, 1]" = torch.ops.aten.mean.dim(pow_70, [-1], True);  pow_70 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:157 in rms_norm, code: hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        add_225: "f32[128, 901, 1]" = torch.ops.aten.add.Tensor(mean_69, 1e-05);  mean_69 = None
        rsqrt_69: "f32[128, 901, 1]" = torch.ops.aten.rsqrt.default(add_225);  add_225 = None
        mul_282: "f32[128, 901, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_781, rsqrt_69);  convert_element_type_781 = rsqrt_69 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:158 in rms_norm, code: return hidden_states.to(input_dtype)
        convert_element_type_782: "bf16[128, 901, 128]" = torch.ops.prims.convert_element_type.default(mul_282, torch.bfloat16);  mul_282 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        convert_element_type_783: "bf16[384, 128]" = torch.ops.prims.convert_element_type.default(primals_47, torch.bfloat16);  primals_47 = None
        permute_142: "bf16[128, 384]" = torch.ops.aten.permute.default(convert_element_type_783, [1, 0]);  convert_element_type_783 = None
        view_359: "bf16[115328, 128]" = torch.ops.aten.reshape.default(convert_element_type_782, [115328, 128])
        mm_141: "bf16[115328, 384]" = torch.ops.aten.mm.default(view_359, permute_142);  view_359 = None
        view_360: "bf16[128, 901, 384]" = torch.ops.aten.reshape.default(mm_141, [128, 901, 384]);  mm_141 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:119 in forward, code: qkv = qkv.view(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        view_361: "bf16[128, 901, 6, 64]" = torch.ops.aten.reshape.default(view_360, [128, 901, 6, 64]);  view_360 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:120 in forward, code: query = qkv[:, :, :self.num_heads]
        slice_461: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_361, 2, 0, 2)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:121 in forward, code: key = qkv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
        slice_464: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_361, 2, 2, 4)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:122 in forward, code: value = qkv[:, :, self.num_heads + self.num_key_value_heads:]
        slice_467: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_361, 2, 4, 9223372036854775807);  view_361 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:34 in apply_rotary_pos_emb, code: q = q.to(cos.dtype)
        convert_element_type_786: "f32[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(slice_461, torch.float32);  slice_461 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:35 in apply_rotary_pos_emb, code: k = k.to(cos.dtype)
        convert_element_type_787: "f32[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(slice_464, torch.float32);  slice_464 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:37 in apply_rotary_pos_emb, code: q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
        mul_283: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_786, unsqueeze_96)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:25 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_468: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_786, 3, 0, 32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:26 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_469: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_786, 3, 32, 9223372036854775807);  convert_element_type_786 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:27 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_70: "f32[128, 901, 2, 32]" = torch.ops.aten.neg.default(slice_469);  slice_469 = None
        cat_72: "f32[128, 901, 2, 64]" = torch.ops.aten.cat.default([neg_70, slice_468], -1);  neg_70 = slice_468 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:37 in apply_rotary_pos_emb, code: q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
        mul_284: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(cat_72, unsqueeze_97);  cat_72 = None
        add_226: "f32[128, 901, 2, 64]" = torch.ops.aten.add.Tensor(mul_283, mul_284);  mul_283 = mul_284 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:38 in apply_rotary_pos_emb, code: k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
        mul_285: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_787, unsqueeze_96)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:25 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_470: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_787, 3, 0, 32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:26 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_471: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_787, 3, 32, 9223372036854775807);  convert_element_type_787 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:27 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_71: "f32[128, 901, 2, 32]" = torch.ops.aten.neg.default(slice_471);  slice_471 = None
        cat_73: "f32[128, 901, 2, 64]" = torch.ops.aten.cat.default([neg_71, slice_470], -1);  neg_71 = slice_470 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:38 in apply_rotary_pos_emb, code: k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
        mul_286: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(cat_73, unsqueeze_97);  cat_73 = None
        add_227: "f32[128, 901, 2, 64]" = torch.ops.aten.add.Tensor(mul_285, mul_286);  mul_285 = mul_286 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:40 in apply_rotary_pos_emb, code: return q_embed.to(orig_dtype), k_embed.to(orig_dtype)
        convert_element_type_788: "bf16[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(add_226, torch.bfloat16);  add_226 = None
        convert_element_type_789: "bf16[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(add_227, torch.bfloat16);  add_227 = None
        
         # File: C:\Users\arroy\anaconda3\envs\hrm_v1\lib\site-packages\flash_attn\flash_attn_interface.py:839 in forward, code: out_padded, softmax_lse, S_dmask, rng_state = _wrapped_flash_attn_forward(
        _flash_attn_forward_35 = torch.ops.flash_attn._flash_attn_forward.default(convert_element_type_788, convert_element_type_789, slice_467, 0.0, 0.125, False, -1, -1, 0.0, None, False);  convert_element_type_788 = convert_element_type_789 = slice_467 = None
        getitem_210: "bf16[128, 901, 2, 64]" = _flash_attn_forward_35[0];  _flash_attn_forward_35 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:135 in forward, code: attn_output = attn_output.view(batch_size, seq_len, self.output_size)  # type: ignore
        view_362: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(getitem_210, [128, 901, 128]);  getitem_210 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        convert_element_type_790: "bf16[128, 128]" = torch.ops.prims.convert_element_type.default(primals_48, torch.bfloat16);  primals_48 = None
        permute_143: "bf16[128, 128]" = torch.ops.aten.permute.default(convert_element_type_790, [1, 0]);  convert_element_type_790 = None
        view_363: "bf16[115328, 128]" = torch.ops.aten.reshape.default(view_362, [115328, 128]);  view_362 = None
        mm_142: "bf16[115328, 128]" = torch.ops.aten.mm.default(view_363, permute_143);  view_363 = None
        view_364: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(mm_142, [128, 901, 128]);  mm_142 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:80 in forward, code: hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        add_228: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(convert_element_type_782, view_364);  convert_element_type_782 = view_364 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:154 in rms_norm, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_793: "f32[128, 901, 128]" = torch.ops.prims.convert_element_type.default(add_228, torch.float32);  add_228 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:156 in rms_norm, code: variance = hidden_states.square().mean(-1, keepdim=True)
        pow_71: "f32[128, 901, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_793, 2)
        mean_70: "f32[128, 901, 1]" = torch.ops.aten.mean.dim(pow_71, [-1], True);  pow_71 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:157 in rms_norm, code: hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        add_229: "f32[128, 901, 1]" = torch.ops.aten.add.Tensor(mean_70, 1e-05);  mean_70 = None
        rsqrt_70: "f32[128, 901, 1]" = torch.ops.aten.rsqrt.default(add_229);  add_229 = None
        mul_287: "f32[128, 901, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_793, rsqrt_70);  convert_element_type_793 = rsqrt_70 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:158 in rms_norm, code: return hidden_states.to(input_dtype)
        convert_element_type_794: "bf16[128, 901, 128]" = torch.ops.prims.convert_element_type.default(mul_287, torch.bfloat16);  mul_287 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        convert_element_type_795: "bf16[1024, 128]" = torch.ops.prims.convert_element_type.default(primals_49, torch.bfloat16);  primals_49 = None
        permute_144: "bf16[128, 1024]" = torch.ops.aten.permute.default(convert_element_type_795, [1, 0]);  convert_element_type_795 = None
        view_365: "bf16[115328, 128]" = torch.ops.aten.reshape.default(convert_element_type_794, [115328, 128])
        mm_143: "bf16[115328, 1024]" = torch.ops.aten.mm.default(view_365, permute_144);  view_365 = None
        view_366: "bf16[128, 901, 1024]" = torch.ops.aten.reshape.default(mm_143, [128, 901, 1024]);  mm_143 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:148 in forward, code: gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        split_35 = torch.ops.aten.split.Tensor(view_366, 512, -1);  view_366 = None
        getitem_214: "bf16[128, 901, 512]" = split_35[0]
        getitem_215: "bf16[128, 901, 512]" = split_35[1];  split_35 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:149 in forward, code: return self.down_proj(F.silu(gate) * up)
        convert_element_type_798: "f32[128, 901, 512]" = torch.ops.prims.convert_element_type.default(getitem_214, torch.float32);  getitem_214 = None
        sigmoid_35: "f32[128, 901, 512]" = torch.ops.aten.sigmoid.default(convert_element_type_798)
        mul_288: "f32[128, 901, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_798, sigmoid_35);  convert_element_type_798 = sigmoid_35 = None
        convert_element_type_799: "bf16[128, 901, 512]" = torch.ops.prims.convert_element_type.default(mul_288, torch.bfloat16);  mul_288 = None
        mul_289: "bf16[128, 901, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_799, getitem_215);  convert_element_type_799 = getitem_215 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        convert_element_type_800: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(primals_50, torch.bfloat16);  primals_50 = None
        permute_145: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_800, [1, 0]);  convert_element_type_800 = None
        view_367: "bf16[115328, 512]" = torch.ops.aten.reshape.default(mul_289, [115328, 512]);  mul_289 = None
        mm_144: "bf16[115328, 128]" = torch.ops.aten.mm.default(view_367, permute_145);  view_367 = None
        view_368: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(mm_144, [128, 901, 128]);  mm_144 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:82 in forward, code: hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)
        add_230: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(convert_element_type_794, view_368);  convert_element_type_794 = view_368 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:154 in rms_norm, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_803: "f32[128, 901, 128]" = torch.ops.prims.convert_element_type.default(add_230, torch.float32);  add_230 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:156 in rms_norm, code: variance = hidden_states.square().mean(-1, keepdim=True)
        pow_72: "f32[128, 901, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_803, 2)
        mean_71: "f32[128, 901, 1]" = torch.ops.aten.mean.dim(pow_72, [-1], True);  pow_72 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:157 in rms_norm, code: hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        add_231: "f32[128, 901, 1]" = torch.ops.aten.add.Tensor(mean_71, 1e-05);  mean_71 = None
        rsqrt_71: "f32[128, 901, 1]" = torch.ops.aten.rsqrt.default(add_231);  add_231 = None
        mul_290: "f32[128, 901, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_803, rsqrt_71);  convert_element_type_803 = rsqrt_71 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:158 in rms_norm, code: return hidden_states.to(input_dtype)
        convert_element_type_804: "bf16[128, 901, 128]" = torch.ops.prims.convert_element_type.default(mul_290, torch.bfloat16);  mul_290 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:195 in forward, code: z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
        add_232: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(convert_element_type_804, mul_194);  mul_194 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:94 in forward, code: hidden_states = hidden_states + input_injection
        add_233: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(convert_element_type_716, add_232);  convert_element_type_716 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_369: "bf16[115328, 128]" = torch.ops.aten.reshape.default(add_233, [115328, 128])
        mm_145: "bf16[115328, 384]" = torch.ops.aten.mm.default(view_369, permute_98);  view_369 = None
        view_370: "bf16[128, 901, 384]" = torch.ops.aten.reshape.default(mm_145, [128, 901, 384]);  mm_145 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:119 in forward, code: qkv = qkv.view(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        view_371: "bf16[128, 901, 6, 64]" = torch.ops.aten.reshape.default(view_370, [128, 901, 6, 64]);  view_370 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:120 in forward, code: query = qkv[:, :, :self.num_heads]
        slice_474: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_371, 2, 0, 2)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:121 in forward, code: key = qkv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
        slice_477: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_371, 2, 2, 4)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:122 in forward, code: value = qkv[:, :, self.num_heads + self.num_key_value_heads:]
        slice_480: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_371, 2, 4, 9223372036854775807);  view_371 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:34 in apply_rotary_pos_emb, code: q = q.to(cos.dtype)
        convert_element_type_808: "f32[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(slice_474, torch.float32);  slice_474 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:35 in apply_rotary_pos_emb, code: k = k.to(cos.dtype)
        convert_element_type_809: "f32[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(slice_477, torch.float32);  slice_477 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:37 in apply_rotary_pos_emb, code: q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
        mul_291: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_808, unsqueeze_96)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:25 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_481: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_808, 3, 0, 32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:26 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_482: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_808, 3, 32, 9223372036854775807);  convert_element_type_808 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:27 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_72: "f32[128, 901, 2, 32]" = torch.ops.aten.neg.default(slice_482);  slice_482 = None
        cat_74: "f32[128, 901, 2, 64]" = torch.ops.aten.cat.default([neg_72, slice_481], -1);  neg_72 = slice_481 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:37 in apply_rotary_pos_emb, code: q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
        mul_292: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(cat_74, unsqueeze_97);  cat_74 = None
        add_234: "f32[128, 901, 2, 64]" = torch.ops.aten.add.Tensor(mul_291, mul_292);  mul_291 = mul_292 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:38 in apply_rotary_pos_emb, code: k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
        mul_293: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_809, unsqueeze_96)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:25 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_483: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_809, 3, 0, 32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:26 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_484: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_809, 3, 32, 9223372036854775807);  convert_element_type_809 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:27 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_73: "f32[128, 901, 2, 32]" = torch.ops.aten.neg.default(slice_484);  slice_484 = None
        cat_75: "f32[128, 901, 2, 64]" = torch.ops.aten.cat.default([neg_73, slice_483], -1);  neg_73 = slice_483 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:38 in apply_rotary_pos_emb, code: k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
        mul_294: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(cat_75, unsqueeze_97);  cat_75 = None
        add_235: "f32[128, 901, 2, 64]" = torch.ops.aten.add.Tensor(mul_293, mul_294);  mul_293 = mul_294 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:40 in apply_rotary_pos_emb, code: return q_embed.to(orig_dtype), k_embed.to(orig_dtype)
        convert_element_type_810: "bf16[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(add_234, torch.bfloat16);  add_234 = None
        convert_element_type_811: "bf16[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(add_235, torch.bfloat16);  add_235 = None
        
         # File: C:\Users\arroy\anaconda3\envs\hrm_v1\lib\site-packages\flash_attn\flash_attn_interface.py:839 in forward, code: out_padded, softmax_lse, S_dmask, rng_state = _wrapped_flash_attn_forward(
        _flash_attn_forward_36 = torch.ops.flash_attn._flash_attn_forward.default(convert_element_type_810, convert_element_type_811, slice_480, 0.0, 0.125, False, -1, -1, 0.0, None, False);  convert_element_type_810 = convert_element_type_811 = slice_480 = None
        getitem_216: "bf16[128, 901, 2, 64]" = _flash_attn_forward_36[0];  _flash_attn_forward_36 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:135 in forward, code: attn_output = attn_output.view(batch_size, seq_len, self.output_size)  # type: ignore
        view_372: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(getitem_216, [128, 901, 128]);  getitem_216 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_373: "bf16[115328, 128]" = torch.ops.aten.reshape.default(view_372, [115328, 128]);  view_372 = None
        mm_146: "bf16[115328, 128]" = torch.ops.aten.mm.default(view_373, permute_99);  view_373 = None
        view_374: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(mm_146, [128, 901, 128]);  mm_146 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:80 in forward, code: hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        add_236: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(add_233, view_374);  add_233 = view_374 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:154 in rms_norm, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_815: "f32[128, 901, 128]" = torch.ops.prims.convert_element_type.default(add_236, torch.float32);  add_236 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:156 in rms_norm, code: variance = hidden_states.square().mean(-1, keepdim=True)
        pow_73: "f32[128, 901, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_815, 2)
        mean_72: "f32[128, 901, 1]" = torch.ops.aten.mean.dim(pow_73, [-1], True);  pow_73 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:157 in rms_norm, code: hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        add_237: "f32[128, 901, 1]" = torch.ops.aten.add.Tensor(mean_72, 1e-05);  mean_72 = None
        rsqrt_72: "f32[128, 901, 1]" = torch.ops.aten.rsqrt.default(add_237);  add_237 = None
        mul_295: "f32[128, 901, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_815, rsqrt_72);  convert_element_type_815 = rsqrt_72 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:158 in rms_norm, code: return hidden_states.to(input_dtype)
        convert_element_type_816: "bf16[128, 901, 128]" = torch.ops.prims.convert_element_type.default(mul_295, torch.bfloat16);  mul_295 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_375: "bf16[115328, 128]" = torch.ops.aten.reshape.default(convert_element_type_816, [115328, 128])
        mm_147: "bf16[115328, 1024]" = torch.ops.aten.mm.default(view_375, permute_100);  view_375 = None
        view_376: "bf16[128, 901, 1024]" = torch.ops.aten.reshape.default(mm_147, [128, 901, 1024]);  mm_147 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:148 in forward, code: gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        split_36 = torch.ops.aten.split.Tensor(view_376, 512, -1);  view_376 = None
        getitem_220: "bf16[128, 901, 512]" = split_36[0]
        getitem_221: "bf16[128, 901, 512]" = split_36[1];  split_36 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:149 in forward, code: return self.down_proj(F.silu(gate) * up)
        convert_element_type_820: "f32[128, 901, 512]" = torch.ops.prims.convert_element_type.default(getitem_220, torch.float32);  getitem_220 = None
        sigmoid_36: "f32[128, 901, 512]" = torch.ops.aten.sigmoid.default(convert_element_type_820)
        mul_296: "f32[128, 901, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_820, sigmoid_36);  convert_element_type_820 = sigmoid_36 = None
        convert_element_type_821: "bf16[128, 901, 512]" = torch.ops.prims.convert_element_type.default(mul_296, torch.bfloat16);  mul_296 = None
        mul_297: "bf16[128, 901, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_821, getitem_221);  convert_element_type_821 = getitem_221 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_377: "bf16[115328, 512]" = torch.ops.aten.reshape.default(mul_297, [115328, 512]);  mul_297 = None
        mm_148: "bf16[115328, 128]" = torch.ops.aten.mm.default(view_377, permute_101);  view_377 = None
        view_378: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(mm_148, [128, 901, 128]);  mm_148 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:82 in forward, code: hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)
        add_238: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(convert_element_type_816, view_378);  convert_element_type_816 = view_378 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:154 in rms_norm, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_825: "f32[128, 901, 128]" = torch.ops.prims.convert_element_type.default(add_238, torch.float32);  add_238 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:156 in rms_norm, code: variance = hidden_states.square().mean(-1, keepdim=True)
        pow_74: "f32[128, 901, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_825, 2)
        mean_73: "f32[128, 901, 1]" = torch.ops.aten.mean.dim(pow_74, [-1], True);  pow_74 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:157 in rms_norm, code: hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        add_239: "f32[128, 901, 1]" = torch.ops.aten.add.Tensor(mean_73, 1e-05);  mean_73 = None
        rsqrt_73: "f32[128, 901, 1]" = torch.ops.aten.rsqrt.default(add_239);  add_239 = None
        mul_298: "f32[128, 901, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_825, rsqrt_73);  convert_element_type_825 = rsqrt_73 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:158 in rms_norm, code: return hidden_states.to(input_dtype)
        convert_element_type_826: "bf16[128, 901, 128]" = torch.ops.prims.convert_element_type.default(mul_298, torch.bfloat16);  mul_298 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_379: "bf16[115328, 128]" = torch.ops.aten.reshape.default(convert_element_type_826, [115328, 128])
        mm_149: "bf16[115328, 384]" = torch.ops.aten.mm.default(view_379, permute_102);  view_379 = None
        view_380: "bf16[128, 901, 384]" = torch.ops.aten.reshape.default(mm_149, [128, 901, 384]);  mm_149 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:119 in forward, code: qkv = qkv.view(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        view_381: "bf16[128, 901, 6, 64]" = torch.ops.aten.reshape.default(view_380, [128, 901, 6, 64]);  view_380 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:120 in forward, code: query = qkv[:, :, :self.num_heads]
        slice_487: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_381, 2, 0, 2)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:121 in forward, code: key = qkv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
        slice_490: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_381, 2, 2, 4)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:122 in forward, code: value = qkv[:, :, self.num_heads + self.num_key_value_heads:]
        slice_493: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_381, 2, 4, 9223372036854775807);  view_381 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:34 in apply_rotary_pos_emb, code: q = q.to(cos.dtype)
        convert_element_type_830: "f32[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(slice_487, torch.float32);  slice_487 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:35 in apply_rotary_pos_emb, code: k = k.to(cos.dtype)
        convert_element_type_831: "f32[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(slice_490, torch.float32);  slice_490 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:37 in apply_rotary_pos_emb, code: q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
        mul_299: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_830, unsqueeze_96)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:25 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_494: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_830, 3, 0, 32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:26 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_495: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_830, 3, 32, 9223372036854775807);  convert_element_type_830 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:27 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_74: "f32[128, 901, 2, 32]" = torch.ops.aten.neg.default(slice_495);  slice_495 = None
        cat_76: "f32[128, 901, 2, 64]" = torch.ops.aten.cat.default([neg_74, slice_494], -1);  neg_74 = slice_494 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:37 in apply_rotary_pos_emb, code: q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
        mul_300: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(cat_76, unsqueeze_97);  cat_76 = None
        add_240: "f32[128, 901, 2, 64]" = torch.ops.aten.add.Tensor(mul_299, mul_300);  mul_299 = mul_300 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:38 in apply_rotary_pos_emb, code: k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
        mul_301: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_831, unsqueeze_96)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:25 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_496: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_831, 3, 0, 32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:26 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_497: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_831, 3, 32, 9223372036854775807);  convert_element_type_831 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:27 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_75: "f32[128, 901, 2, 32]" = torch.ops.aten.neg.default(slice_497);  slice_497 = None
        cat_77: "f32[128, 901, 2, 64]" = torch.ops.aten.cat.default([neg_75, slice_496], -1);  neg_75 = slice_496 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:38 in apply_rotary_pos_emb, code: k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
        mul_302: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(cat_77, unsqueeze_97);  cat_77 = None
        add_241: "f32[128, 901, 2, 64]" = torch.ops.aten.add.Tensor(mul_301, mul_302);  mul_301 = mul_302 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:40 in apply_rotary_pos_emb, code: return q_embed.to(orig_dtype), k_embed.to(orig_dtype)
        convert_element_type_832: "bf16[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(add_240, torch.bfloat16);  add_240 = None
        convert_element_type_833: "bf16[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(add_241, torch.bfloat16);  add_241 = None
        
         # File: C:\Users\arroy\anaconda3\envs\hrm_v1\lib\site-packages\flash_attn\flash_attn_interface.py:839 in forward, code: out_padded, softmax_lse, S_dmask, rng_state = _wrapped_flash_attn_forward(
        _flash_attn_forward_37 = torch.ops.flash_attn._flash_attn_forward.default(convert_element_type_832, convert_element_type_833, slice_493, 0.0, 0.125, False, -1, -1, 0.0, None, False);  convert_element_type_832 = convert_element_type_833 = slice_493 = None
        getitem_222: "bf16[128, 901, 2, 64]" = _flash_attn_forward_37[0];  _flash_attn_forward_37 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:135 in forward, code: attn_output = attn_output.view(batch_size, seq_len, self.output_size)  # type: ignore
        view_382: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(getitem_222, [128, 901, 128]);  getitem_222 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_383: "bf16[115328, 128]" = torch.ops.aten.reshape.default(view_382, [115328, 128]);  view_382 = None
        mm_150: "bf16[115328, 128]" = torch.ops.aten.mm.default(view_383, permute_103);  view_383 = None
        view_384: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(mm_150, [128, 901, 128]);  mm_150 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:80 in forward, code: hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        add_242: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(convert_element_type_826, view_384);  convert_element_type_826 = view_384 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:154 in rms_norm, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_837: "f32[128, 901, 128]" = torch.ops.prims.convert_element_type.default(add_242, torch.float32);  add_242 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:156 in rms_norm, code: variance = hidden_states.square().mean(-1, keepdim=True)
        pow_75: "f32[128, 901, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_837, 2)
        mean_74: "f32[128, 901, 1]" = torch.ops.aten.mean.dim(pow_75, [-1], True);  pow_75 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:157 in rms_norm, code: hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        add_243: "f32[128, 901, 1]" = torch.ops.aten.add.Tensor(mean_74, 1e-05);  mean_74 = None
        rsqrt_74: "f32[128, 901, 1]" = torch.ops.aten.rsqrt.default(add_243);  add_243 = None
        mul_303: "f32[128, 901, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_837, rsqrt_74);  convert_element_type_837 = rsqrt_74 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:158 in rms_norm, code: return hidden_states.to(input_dtype)
        convert_element_type_838: "bf16[128, 901, 128]" = torch.ops.prims.convert_element_type.default(mul_303, torch.bfloat16);  mul_303 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_385: "bf16[115328, 128]" = torch.ops.aten.reshape.default(convert_element_type_838, [115328, 128])
        mm_151: "bf16[115328, 1024]" = torch.ops.aten.mm.default(view_385, permute_104);  view_385 = None
        view_386: "bf16[128, 901, 1024]" = torch.ops.aten.reshape.default(mm_151, [128, 901, 1024]);  mm_151 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:148 in forward, code: gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        split_37 = torch.ops.aten.split.Tensor(view_386, 512, -1);  view_386 = None
        getitem_226: "bf16[128, 901, 512]" = split_37[0]
        getitem_227: "bf16[128, 901, 512]" = split_37[1];  split_37 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:149 in forward, code: return self.down_proj(F.silu(gate) * up)
        convert_element_type_842: "f32[128, 901, 512]" = torch.ops.prims.convert_element_type.default(getitem_226, torch.float32);  getitem_226 = None
        sigmoid_37: "f32[128, 901, 512]" = torch.ops.aten.sigmoid.default(convert_element_type_842)
        mul_304: "f32[128, 901, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_842, sigmoid_37);  convert_element_type_842 = sigmoid_37 = None
        convert_element_type_843: "bf16[128, 901, 512]" = torch.ops.prims.convert_element_type.default(mul_304, torch.bfloat16);  mul_304 = None
        mul_305: "bf16[128, 901, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_843, getitem_227);  convert_element_type_843 = getitem_227 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_387: "bf16[115328, 512]" = torch.ops.aten.reshape.default(mul_305, [115328, 512]);  mul_305 = None
        mm_152: "bf16[115328, 128]" = torch.ops.aten.mm.default(view_387, permute_105);  view_387 = None
        view_388: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(mm_152, [128, 901, 128]);  mm_152 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:82 in forward, code: hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)
        add_244: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(convert_element_type_838, view_388);  convert_element_type_838 = view_388 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:154 in rms_norm, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_847: "f32[128, 901, 128]" = torch.ops.prims.convert_element_type.default(add_244, torch.float32);  add_244 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:156 in rms_norm, code: variance = hidden_states.square().mean(-1, keepdim=True)
        pow_76: "f32[128, 901, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_847, 2)
        mean_75: "f32[128, 901, 1]" = torch.ops.aten.mean.dim(pow_76, [-1], True);  pow_76 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:157 in rms_norm, code: hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        add_245: "f32[128, 901, 1]" = torch.ops.aten.add.Tensor(mean_75, 1e-05);  mean_75 = None
        rsqrt_75: "f32[128, 901, 1]" = torch.ops.aten.rsqrt.default(add_245);  add_245 = None
        mul_306: "f32[128, 901, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_847, rsqrt_75);  convert_element_type_847 = rsqrt_75 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:158 in rms_norm, code: return hidden_states.to(input_dtype)
        convert_element_type_848: "bf16[128, 901, 128]" = torch.ops.prims.convert_element_type.default(mul_306, torch.bfloat16);  mul_306 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_389: "bf16[115328, 128]" = torch.ops.aten.reshape.default(convert_element_type_848, [115328, 128])
        mm_153: "bf16[115328, 384]" = torch.ops.aten.mm.default(view_389, permute_106);  view_389 = None
        view_390: "bf16[128, 901, 384]" = torch.ops.aten.reshape.default(mm_153, [128, 901, 384]);  mm_153 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:119 in forward, code: qkv = qkv.view(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        view_391: "bf16[128, 901, 6, 64]" = torch.ops.aten.reshape.default(view_390, [128, 901, 6, 64]);  view_390 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:120 in forward, code: query = qkv[:, :, :self.num_heads]
        slice_500: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_391, 2, 0, 2)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:121 in forward, code: key = qkv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
        slice_503: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_391, 2, 2, 4)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:122 in forward, code: value = qkv[:, :, self.num_heads + self.num_key_value_heads:]
        slice_506: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_391, 2, 4, 9223372036854775807);  view_391 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:34 in apply_rotary_pos_emb, code: q = q.to(cos.dtype)
        convert_element_type_852: "f32[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(slice_500, torch.float32);  slice_500 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:35 in apply_rotary_pos_emb, code: k = k.to(cos.dtype)
        convert_element_type_853: "f32[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(slice_503, torch.float32);  slice_503 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:37 in apply_rotary_pos_emb, code: q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
        mul_307: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_852, unsqueeze_96)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:25 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_507: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_852, 3, 0, 32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:26 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_508: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_852, 3, 32, 9223372036854775807);  convert_element_type_852 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:27 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_76: "f32[128, 901, 2, 32]" = torch.ops.aten.neg.default(slice_508);  slice_508 = None
        cat_78: "f32[128, 901, 2, 64]" = torch.ops.aten.cat.default([neg_76, slice_507], -1);  neg_76 = slice_507 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:37 in apply_rotary_pos_emb, code: q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
        mul_308: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(cat_78, unsqueeze_97);  cat_78 = None
        add_246: "f32[128, 901, 2, 64]" = torch.ops.aten.add.Tensor(mul_307, mul_308);  mul_307 = mul_308 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:38 in apply_rotary_pos_emb, code: k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
        mul_309: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_853, unsqueeze_96)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:25 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_509: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_853, 3, 0, 32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:26 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_510: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_853, 3, 32, 9223372036854775807);  convert_element_type_853 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:27 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_77: "f32[128, 901, 2, 32]" = torch.ops.aten.neg.default(slice_510);  slice_510 = None
        cat_79: "f32[128, 901, 2, 64]" = torch.ops.aten.cat.default([neg_77, slice_509], -1);  neg_77 = slice_509 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:38 in apply_rotary_pos_emb, code: k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
        mul_310: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(cat_79, unsqueeze_97);  cat_79 = None
        add_247: "f32[128, 901, 2, 64]" = torch.ops.aten.add.Tensor(mul_309, mul_310);  mul_309 = mul_310 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:40 in apply_rotary_pos_emb, code: return q_embed.to(orig_dtype), k_embed.to(orig_dtype)
        convert_element_type_854: "bf16[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(add_246, torch.bfloat16);  add_246 = None
        convert_element_type_855: "bf16[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(add_247, torch.bfloat16);  add_247 = None
        
         # File: C:\Users\arroy\anaconda3\envs\hrm_v1\lib\site-packages\flash_attn\flash_attn_interface.py:839 in forward, code: out_padded, softmax_lse, S_dmask, rng_state = _wrapped_flash_attn_forward(
        _flash_attn_forward_38 = torch.ops.flash_attn._flash_attn_forward.default(convert_element_type_854, convert_element_type_855, slice_506, 0.0, 0.125, False, -1, -1, 0.0, None, False);  convert_element_type_854 = convert_element_type_855 = slice_506 = None
        getitem_228: "bf16[128, 901, 2, 64]" = _flash_attn_forward_38[0];  _flash_attn_forward_38 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:135 in forward, code: attn_output = attn_output.view(batch_size, seq_len, self.output_size)  # type: ignore
        view_392: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(getitem_228, [128, 901, 128]);  getitem_228 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_393: "bf16[115328, 128]" = torch.ops.aten.reshape.default(view_392, [115328, 128]);  view_392 = None
        mm_154: "bf16[115328, 128]" = torch.ops.aten.mm.default(view_393, permute_107);  view_393 = None
        view_394: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(mm_154, [128, 901, 128]);  mm_154 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:80 in forward, code: hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        add_248: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(convert_element_type_848, view_394);  convert_element_type_848 = view_394 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:154 in rms_norm, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_859: "f32[128, 901, 128]" = torch.ops.prims.convert_element_type.default(add_248, torch.float32);  add_248 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:156 in rms_norm, code: variance = hidden_states.square().mean(-1, keepdim=True)
        pow_77: "f32[128, 901, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_859, 2)
        mean_76: "f32[128, 901, 1]" = torch.ops.aten.mean.dim(pow_77, [-1], True);  pow_77 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:157 in rms_norm, code: hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        add_249: "f32[128, 901, 1]" = torch.ops.aten.add.Tensor(mean_76, 1e-05);  mean_76 = None
        rsqrt_76: "f32[128, 901, 1]" = torch.ops.aten.rsqrt.default(add_249);  add_249 = None
        mul_311: "f32[128, 901, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_859, rsqrt_76);  convert_element_type_859 = rsqrt_76 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:158 in rms_norm, code: return hidden_states.to(input_dtype)
        convert_element_type_860: "bf16[128, 901, 128]" = torch.ops.prims.convert_element_type.default(mul_311, torch.bfloat16);  mul_311 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_395: "bf16[115328, 128]" = torch.ops.aten.reshape.default(convert_element_type_860, [115328, 128])
        mm_155: "bf16[115328, 1024]" = torch.ops.aten.mm.default(view_395, permute_108);  view_395 = None
        view_396: "bf16[128, 901, 1024]" = torch.ops.aten.reshape.default(mm_155, [128, 901, 1024]);  mm_155 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:148 in forward, code: gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        split_38 = torch.ops.aten.split.Tensor(view_396, 512, -1);  view_396 = None
        getitem_232: "bf16[128, 901, 512]" = split_38[0]
        getitem_233: "bf16[128, 901, 512]" = split_38[1];  split_38 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:149 in forward, code: return self.down_proj(F.silu(gate) * up)
        convert_element_type_864: "f32[128, 901, 512]" = torch.ops.prims.convert_element_type.default(getitem_232, torch.float32);  getitem_232 = None
        sigmoid_38: "f32[128, 901, 512]" = torch.ops.aten.sigmoid.default(convert_element_type_864)
        mul_312: "f32[128, 901, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_864, sigmoid_38);  convert_element_type_864 = sigmoid_38 = None
        convert_element_type_865: "bf16[128, 901, 512]" = torch.ops.prims.convert_element_type.default(mul_312, torch.bfloat16);  mul_312 = None
        mul_313: "bf16[128, 901, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_865, getitem_233);  convert_element_type_865 = getitem_233 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_397: "bf16[115328, 512]" = torch.ops.aten.reshape.default(mul_313, [115328, 512]);  mul_313 = None
        mm_156: "bf16[115328, 128]" = torch.ops.aten.mm.default(view_397, permute_109);  view_397 = None
        view_398: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(mm_156, [128, 901, 128]);  mm_156 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:82 in forward, code: hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)
        add_250: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(convert_element_type_860, view_398);  convert_element_type_860 = view_398 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:154 in rms_norm, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_869: "f32[128, 901, 128]" = torch.ops.prims.convert_element_type.default(add_250, torch.float32);  add_250 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:156 in rms_norm, code: variance = hidden_states.square().mean(-1, keepdim=True)
        pow_78: "f32[128, 901, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_869, 2)
        mean_77: "f32[128, 901, 1]" = torch.ops.aten.mean.dim(pow_78, [-1], True);  pow_78 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:157 in rms_norm, code: hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        add_251: "f32[128, 901, 1]" = torch.ops.aten.add.Tensor(mean_77, 1e-05);  mean_77 = None
        rsqrt_77: "f32[128, 901, 1]" = torch.ops.aten.rsqrt.default(add_251);  add_251 = None
        mul_314: "f32[128, 901, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_869, rsqrt_77);  convert_element_type_869 = rsqrt_77 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:158 in rms_norm, code: return hidden_states.to(input_dtype)
        convert_element_type_870: "bf16[128, 901, 128]" = torch.ops.prims.convert_element_type.default(mul_314, torch.bfloat16);  mul_314 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_399: "bf16[115328, 128]" = torch.ops.aten.reshape.default(convert_element_type_870, [115328, 128])
        mm_157: "bf16[115328, 384]" = torch.ops.aten.mm.default(view_399, permute_110);  view_399 = None
        view_400: "bf16[128, 901, 384]" = torch.ops.aten.reshape.default(mm_157, [128, 901, 384]);  mm_157 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:119 in forward, code: qkv = qkv.view(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        view_401: "bf16[128, 901, 6, 64]" = torch.ops.aten.reshape.default(view_400, [128, 901, 6, 64]);  view_400 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:120 in forward, code: query = qkv[:, :, :self.num_heads]
        slice_513: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_401, 2, 0, 2)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:121 in forward, code: key = qkv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
        slice_516: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_401, 2, 2, 4)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:122 in forward, code: value = qkv[:, :, self.num_heads + self.num_key_value_heads:]
        slice_519: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_401, 2, 4, 9223372036854775807);  view_401 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:34 in apply_rotary_pos_emb, code: q = q.to(cos.dtype)
        convert_element_type_874: "f32[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(slice_513, torch.float32);  slice_513 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:35 in apply_rotary_pos_emb, code: k = k.to(cos.dtype)
        convert_element_type_875: "f32[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(slice_516, torch.float32);  slice_516 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:37 in apply_rotary_pos_emb, code: q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
        mul_315: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_874, unsqueeze_96)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:25 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_520: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_874, 3, 0, 32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:26 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_521: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_874, 3, 32, 9223372036854775807);  convert_element_type_874 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:27 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_78: "f32[128, 901, 2, 32]" = torch.ops.aten.neg.default(slice_521);  slice_521 = None
        cat_80: "f32[128, 901, 2, 64]" = torch.ops.aten.cat.default([neg_78, slice_520], -1);  neg_78 = slice_520 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:37 in apply_rotary_pos_emb, code: q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
        mul_316: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(cat_80, unsqueeze_97);  cat_80 = None
        add_252: "f32[128, 901, 2, 64]" = torch.ops.aten.add.Tensor(mul_315, mul_316);  mul_315 = mul_316 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:38 in apply_rotary_pos_emb, code: k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
        mul_317: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_875, unsqueeze_96)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:25 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_522: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_875, 3, 0, 32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:26 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_523: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_875, 3, 32, 9223372036854775807);  convert_element_type_875 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:27 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_79: "f32[128, 901, 2, 32]" = torch.ops.aten.neg.default(slice_523);  slice_523 = None
        cat_81: "f32[128, 901, 2, 64]" = torch.ops.aten.cat.default([neg_79, slice_522], -1);  neg_79 = slice_522 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:38 in apply_rotary_pos_emb, code: k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
        mul_318: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(cat_81, unsqueeze_97);  cat_81 = None
        add_253: "f32[128, 901, 2, 64]" = torch.ops.aten.add.Tensor(mul_317, mul_318);  mul_317 = mul_318 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:40 in apply_rotary_pos_emb, code: return q_embed.to(orig_dtype), k_embed.to(orig_dtype)
        convert_element_type_876: "bf16[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(add_252, torch.bfloat16);  add_252 = None
        convert_element_type_877: "bf16[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(add_253, torch.bfloat16);  add_253 = None
        
         # File: C:\Users\arroy\anaconda3\envs\hrm_v1\lib\site-packages\flash_attn\flash_attn_interface.py:839 in forward, code: out_padded, softmax_lse, S_dmask, rng_state = _wrapped_flash_attn_forward(
        _flash_attn_forward_39 = torch.ops.flash_attn._flash_attn_forward.default(convert_element_type_876, convert_element_type_877, slice_519, 0.0, 0.125, False, -1, -1, 0.0, None, False);  convert_element_type_876 = convert_element_type_877 = slice_519 = None
        getitem_234: "bf16[128, 901, 2, 64]" = _flash_attn_forward_39[0];  _flash_attn_forward_39 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:135 in forward, code: attn_output = attn_output.view(batch_size, seq_len, self.output_size)  # type: ignore
        view_402: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(getitem_234, [128, 901, 128]);  getitem_234 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_403: "bf16[115328, 128]" = torch.ops.aten.reshape.default(view_402, [115328, 128]);  view_402 = None
        mm_158: "bf16[115328, 128]" = torch.ops.aten.mm.default(view_403, permute_111);  view_403 = None
        view_404: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(mm_158, [128, 901, 128]);  mm_158 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:80 in forward, code: hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        add_254: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(convert_element_type_870, view_404);  convert_element_type_870 = view_404 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:154 in rms_norm, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_881: "f32[128, 901, 128]" = torch.ops.prims.convert_element_type.default(add_254, torch.float32);  add_254 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:156 in rms_norm, code: variance = hidden_states.square().mean(-1, keepdim=True)
        pow_79: "f32[128, 901, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_881, 2)
        mean_78: "f32[128, 901, 1]" = torch.ops.aten.mean.dim(pow_79, [-1], True);  pow_79 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:157 in rms_norm, code: hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        add_255: "f32[128, 901, 1]" = torch.ops.aten.add.Tensor(mean_78, 1e-05);  mean_78 = None
        rsqrt_78: "f32[128, 901, 1]" = torch.ops.aten.rsqrt.default(add_255);  add_255 = None
        mul_319: "f32[128, 901, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_881, rsqrt_78);  convert_element_type_881 = rsqrt_78 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:158 in rms_norm, code: return hidden_states.to(input_dtype)
        convert_element_type_882: "bf16[128, 901, 128]" = torch.ops.prims.convert_element_type.default(mul_319, torch.bfloat16);  mul_319 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_405: "bf16[115328, 128]" = torch.ops.aten.reshape.default(convert_element_type_882, [115328, 128])
        mm_159: "bf16[115328, 1024]" = torch.ops.aten.mm.default(view_405, permute_112);  view_405 = None
        view_406: "bf16[128, 901, 1024]" = torch.ops.aten.reshape.default(mm_159, [128, 901, 1024]);  mm_159 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:148 in forward, code: gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        split_39 = torch.ops.aten.split.Tensor(view_406, 512, -1);  view_406 = None
        getitem_238: "bf16[128, 901, 512]" = split_39[0]
        getitem_239: "bf16[128, 901, 512]" = split_39[1];  split_39 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:149 in forward, code: return self.down_proj(F.silu(gate) * up)
        convert_element_type_886: "f32[128, 901, 512]" = torch.ops.prims.convert_element_type.default(getitem_238, torch.float32);  getitem_238 = None
        sigmoid_39: "f32[128, 901, 512]" = torch.ops.aten.sigmoid.default(convert_element_type_886)
        mul_320: "f32[128, 901, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_886, sigmoid_39);  convert_element_type_886 = sigmoid_39 = None
        convert_element_type_887: "bf16[128, 901, 512]" = torch.ops.prims.convert_element_type.default(mul_320, torch.bfloat16);  mul_320 = None
        mul_321: "bf16[128, 901, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_887, getitem_239);  convert_element_type_887 = getitem_239 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_407: "bf16[115328, 512]" = torch.ops.aten.reshape.default(mul_321, [115328, 512]);  mul_321 = None
        mm_160: "bf16[115328, 128]" = torch.ops.aten.mm.default(view_407, permute_113);  view_407 = None
        view_408: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(mm_160, [128, 901, 128]);  mm_160 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:82 in forward, code: hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)
        add_256: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(convert_element_type_882, view_408);  convert_element_type_882 = view_408 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:154 in rms_norm, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_891: "f32[128, 901, 128]" = torch.ops.prims.convert_element_type.default(add_256, torch.float32);  add_256 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:156 in rms_norm, code: variance = hidden_states.square().mean(-1, keepdim=True)
        pow_80: "f32[128, 901, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_891, 2)
        mean_79: "f32[128, 901, 1]" = torch.ops.aten.mean.dim(pow_80, [-1], True);  pow_80 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:157 in rms_norm, code: hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        add_257: "f32[128, 901, 1]" = torch.ops.aten.add.Tensor(mean_79, 1e-05);  mean_79 = None
        rsqrt_79: "f32[128, 901, 1]" = torch.ops.aten.rsqrt.default(add_257);  add_257 = None
        mul_322: "f32[128, 901, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_891, rsqrt_79);  convert_element_type_891 = rsqrt_79 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:158 in rms_norm, code: return hidden_states.to(input_dtype)
        convert_element_type_892: "bf16[128, 901, 128]" = torch.ops.prims.convert_element_type.default(mul_322, torch.bfloat16);  mul_322 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:94 in forward, code: hidden_states = hidden_states + input_injection
        add_259: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(convert_element_type_892, add_232);  convert_element_type_892 = add_232 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_409: "bf16[115328, 128]" = torch.ops.aten.reshape.default(add_259, [115328, 128])
        mm_161: "bf16[115328, 384]" = torch.ops.aten.mm.default(view_409, permute_98);  view_409 = permute_98 = None
        view_410: "bf16[128, 901, 384]" = torch.ops.aten.reshape.default(mm_161, [128, 901, 384]);  mm_161 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:119 in forward, code: qkv = qkv.view(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        view_411: "bf16[128, 901, 6, 64]" = torch.ops.aten.reshape.default(view_410, [128, 901, 6, 64]);  view_410 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:120 in forward, code: query = qkv[:, :, :self.num_heads]
        slice_526: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_411, 2, 0, 2)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:121 in forward, code: key = qkv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
        slice_529: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_411, 2, 2, 4)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:122 in forward, code: value = qkv[:, :, self.num_heads + self.num_key_value_heads:]
        slice_532: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_411, 2, 4, 9223372036854775807);  view_411 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:34 in apply_rotary_pos_emb, code: q = q.to(cos.dtype)
        convert_element_type_896: "f32[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(slice_526, torch.float32);  slice_526 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:35 in apply_rotary_pos_emb, code: k = k.to(cos.dtype)
        convert_element_type_897: "f32[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(slice_529, torch.float32);  slice_529 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:37 in apply_rotary_pos_emb, code: q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
        mul_323: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_896, unsqueeze_96)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:25 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_533: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_896, 3, 0, 32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:26 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_534: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_896, 3, 32, 9223372036854775807);  convert_element_type_896 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:27 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_80: "f32[128, 901, 2, 32]" = torch.ops.aten.neg.default(slice_534);  slice_534 = None
        cat_82: "f32[128, 901, 2, 64]" = torch.ops.aten.cat.default([neg_80, slice_533], -1);  neg_80 = slice_533 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:37 in apply_rotary_pos_emb, code: q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
        mul_324: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(cat_82, unsqueeze_97);  cat_82 = None
        add_260: "f32[128, 901, 2, 64]" = torch.ops.aten.add.Tensor(mul_323, mul_324);  mul_323 = mul_324 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:38 in apply_rotary_pos_emb, code: k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
        mul_325: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_897, unsqueeze_96)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:25 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_535: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_897, 3, 0, 32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:26 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_536: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_897, 3, 32, 9223372036854775807);  convert_element_type_897 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:27 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_81: "f32[128, 901, 2, 32]" = torch.ops.aten.neg.default(slice_536);  slice_536 = None
        cat_83: "f32[128, 901, 2, 64]" = torch.ops.aten.cat.default([neg_81, slice_535], -1);  neg_81 = slice_535 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:38 in apply_rotary_pos_emb, code: k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
        mul_326: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(cat_83, unsqueeze_97);  cat_83 = None
        add_261: "f32[128, 901, 2, 64]" = torch.ops.aten.add.Tensor(mul_325, mul_326);  mul_325 = mul_326 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:40 in apply_rotary_pos_emb, code: return q_embed.to(orig_dtype), k_embed.to(orig_dtype)
        convert_element_type_898: "bf16[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(add_260, torch.bfloat16);  add_260 = None
        convert_element_type_899: "bf16[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(add_261, torch.bfloat16);  add_261 = None
        
         # File: C:\Users\arroy\anaconda3\envs\hrm_v1\lib\site-packages\flash_attn\flash_attn_interface.py:839 in forward, code: out_padded, softmax_lse, S_dmask, rng_state = _wrapped_flash_attn_forward(
        _flash_attn_forward_40 = torch.ops.flash_attn._flash_attn_forward.default(convert_element_type_898, convert_element_type_899, slice_532, 0.0, 0.125, False, -1, -1, 0.0, None, False);  convert_element_type_898 = convert_element_type_899 = slice_532 = None
        getitem_240: "bf16[128, 901, 2, 64]" = _flash_attn_forward_40[0];  _flash_attn_forward_40 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:135 in forward, code: attn_output = attn_output.view(batch_size, seq_len, self.output_size)  # type: ignore
        view_412: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(getitem_240, [128, 901, 128]);  getitem_240 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_413: "bf16[115328, 128]" = torch.ops.aten.reshape.default(view_412, [115328, 128]);  view_412 = None
        mm_162: "bf16[115328, 128]" = torch.ops.aten.mm.default(view_413, permute_99);  view_413 = permute_99 = None
        view_414: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(mm_162, [128, 901, 128]);  mm_162 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:80 in forward, code: hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        add_262: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(add_259, view_414);  add_259 = view_414 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:154 in rms_norm, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_903: "f32[128, 901, 128]" = torch.ops.prims.convert_element_type.default(add_262, torch.float32);  add_262 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:156 in rms_norm, code: variance = hidden_states.square().mean(-1, keepdim=True)
        pow_81: "f32[128, 901, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_903, 2)
        mean_80: "f32[128, 901, 1]" = torch.ops.aten.mean.dim(pow_81, [-1], True);  pow_81 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:157 in rms_norm, code: hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        add_263: "f32[128, 901, 1]" = torch.ops.aten.add.Tensor(mean_80, 1e-05);  mean_80 = None
        rsqrt_80: "f32[128, 901, 1]" = torch.ops.aten.rsqrt.default(add_263);  add_263 = None
        mul_327: "f32[128, 901, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_903, rsqrt_80);  convert_element_type_903 = rsqrt_80 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:158 in rms_norm, code: return hidden_states.to(input_dtype)
        convert_element_type_904: "bf16[128, 901, 128]" = torch.ops.prims.convert_element_type.default(mul_327, torch.bfloat16);  mul_327 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_415: "bf16[115328, 128]" = torch.ops.aten.reshape.default(convert_element_type_904, [115328, 128])
        mm_163: "bf16[115328, 1024]" = torch.ops.aten.mm.default(view_415, permute_100);  view_415 = permute_100 = None
        view_416: "bf16[128, 901, 1024]" = torch.ops.aten.reshape.default(mm_163, [128, 901, 1024]);  mm_163 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:148 in forward, code: gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        split_40 = torch.ops.aten.split.Tensor(view_416, 512, -1);  view_416 = None
        getitem_244: "bf16[128, 901, 512]" = split_40[0]
        getitem_245: "bf16[128, 901, 512]" = split_40[1];  split_40 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:149 in forward, code: return self.down_proj(F.silu(gate) * up)
        convert_element_type_908: "f32[128, 901, 512]" = torch.ops.prims.convert_element_type.default(getitem_244, torch.float32);  getitem_244 = None
        sigmoid_40: "f32[128, 901, 512]" = torch.ops.aten.sigmoid.default(convert_element_type_908)
        mul_328: "f32[128, 901, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_908, sigmoid_40);  convert_element_type_908 = sigmoid_40 = None
        convert_element_type_909: "bf16[128, 901, 512]" = torch.ops.prims.convert_element_type.default(mul_328, torch.bfloat16);  mul_328 = None
        mul_329: "bf16[128, 901, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_909, getitem_245);  convert_element_type_909 = getitem_245 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_417: "bf16[115328, 512]" = torch.ops.aten.reshape.default(mul_329, [115328, 512]);  mul_329 = None
        mm_164: "bf16[115328, 128]" = torch.ops.aten.mm.default(view_417, permute_101);  view_417 = permute_101 = None
        view_418: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(mm_164, [128, 901, 128]);  mm_164 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:82 in forward, code: hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)
        add_264: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(convert_element_type_904, view_418);  convert_element_type_904 = view_418 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:154 in rms_norm, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_913: "f32[128, 901, 128]" = torch.ops.prims.convert_element_type.default(add_264, torch.float32);  add_264 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:156 in rms_norm, code: variance = hidden_states.square().mean(-1, keepdim=True)
        pow_82: "f32[128, 901, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_913, 2)
        mean_81: "f32[128, 901, 1]" = torch.ops.aten.mean.dim(pow_82, [-1], True);  pow_82 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:157 in rms_norm, code: hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        add_265: "f32[128, 901, 1]" = torch.ops.aten.add.Tensor(mean_81, 1e-05);  mean_81 = None
        rsqrt_81: "f32[128, 901, 1]" = torch.ops.aten.rsqrt.default(add_265);  add_265 = None
        mul_330: "f32[128, 901, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_913, rsqrt_81);  convert_element_type_913 = rsqrt_81 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:158 in rms_norm, code: return hidden_states.to(input_dtype)
        convert_element_type_914: "bf16[128, 901, 128]" = torch.ops.prims.convert_element_type.default(mul_330, torch.bfloat16);  mul_330 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_419: "bf16[115328, 128]" = torch.ops.aten.reshape.default(convert_element_type_914, [115328, 128])
        mm_165: "bf16[115328, 384]" = torch.ops.aten.mm.default(view_419, permute_102);  view_419 = permute_102 = None
        view_420: "bf16[128, 901, 384]" = torch.ops.aten.reshape.default(mm_165, [128, 901, 384]);  mm_165 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:119 in forward, code: qkv = qkv.view(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        view_421: "bf16[128, 901, 6, 64]" = torch.ops.aten.reshape.default(view_420, [128, 901, 6, 64]);  view_420 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:120 in forward, code: query = qkv[:, :, :self.num_heads]
        slice_539: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_421, 2, 0, 2)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:121 in forward, code: key = qkv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
        slice_542: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_421, 2, 2, 4)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:122 in forward, code: value = qkv[:, :, self.num_heads + self.num_key_value_heads:]
        slice_545: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_421, 2, 4, 9223372036854775807);  view_421 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:34 in apply_rotary_pos_emb, code: q = q.to(cos.dtype)
        convert_element_type_918: "f32[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(slice_539, torch.float32);  slice_539 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:35 in apply_rotary_pos_emb, code: k = k.to(cos.dtype)
        convert_element_type_919: "f32[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(slice_542, torch.float32);  slice_542 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:37 in apply_rotary_pos_emb, code: q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
        mul_331: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_918, unsqueeze_96)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:25 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_546: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_918, 3, 0, 32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:26 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_547: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_918, 3, 32, 9223372036854775807);  convert_element_type_918 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:27 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_82: "f32[128, 901, 2, 32]" = torch.ops.aten.neg.default(slice_547);  slice_547 = None
        cat_84: "f32[128, 901, 2, 64]" = torch.ops.aten.cat.default([neg_82, slice_546], -1);  neg_82 = slice_546 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:37 in apply_rotary_pos_emb, code: q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
        mul_332: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(cat_84, unsqueeze_97);  cat_84 = None
        add_266: "f32[128, 901, 2, 64]" = torch.ops.aten.add.Tensor(mul_331, mul_332);  mul_331 = mul_332 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:38 in apply_rotary_pos_emb, code: k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
        mul_333: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_919, unsqueeze_96)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:25 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_548: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_919, 3, 0, 32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:26 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_549: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_919, 3, 32, 9223372036854775807);  convert_element_type_919 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:27 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_83: "f32[128, 901, 2, 32]" = torch.ops.aten.neg.default(slice_549);  slice_549 = None
        cat_85: "f32[128, 901, 2, 64]" = torch.ops.aten.cat.default([neg_83, slice_548], -1);  neg_83 = slice_548 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:38 in apply_rotary_pos_emb, code: k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
        mul_334: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(cat_85, unsqueeze_97);  cat_85 = None
        add_267: "f32[128, 901, 2, 64]" = torch.ops.aten.add.Tensor(mul_333, mul_334);  mul_333 = mul_334 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:40 in apply_rotary_pos_emb, code: return q_embed.to(orig_dtype), k_embed.to(orig_dtype)
        convert_element_type_920: "bf16[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(add_266, torch.bfloat16);  add_266 = None
        convert_element_type_921: "bf16[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(add_267, torch.bfloat16);  add_267 = None
        
         # File: C:\Users\arroy\anaconda3\envs\hrm_v1\lib\site-packages\flash_attn\flash_attn_interface.py:839 in forward, code: out_padded, softmax_lse, S_dmask, rng_state = _wrapped_flash_attn_forward(
        _flash_attn_forward_41 = torch.ops.flash_attn._flash_attn_forward.default(convert_element_type_920, convert_element_type_921, slice_545, 0.0, 0.125, False, -1, -1, 0.0, None, False);  convert_element_type_920 = convert_element_type_921 = slice_545 = None
        getitem_246: "bf16[128, 901, 2, 64]" = _flash_attn_forward_41[0];  _flash_attn_forward_41 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:135 in forward, code: attn_output = attn_output.view(batch_size, seq_len, self.output_size)  # type: ignore
        view_422: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(getitem_246, [128, 901, 128]);  getitem_246 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_423: "bf16[115328, 128]" = torch.ops.aten.reshape.default(view_422, [115328, 128]);  view_422 = None
        mm_166: "bf16[115328, 128]" = torch.ops.aten.mm.default(view_423, permute_103);  view_423 = permute_103 = None
        view_424: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(mm_166, [128, 901, 128]);  mm_166 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:80 in forward, code: hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        add_268: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(convert_element_type_914, view_424);  convert_element_type_914 = view_424 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:154 in rms_norm, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_925: "f32[128, 901, 128]" = torch.ops.prims.convert_element_type.default(add_268, torch.float32);  add_268 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:156 in rms_norm, code: variance = hidden_states.square().mean(-1, keepdim=True)
        pow_83: "f32[128, 901, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_925, 2)
        mean_82: "f32[128, 901, 1]" = torch.ops.aten.mean.dim(pow_83, [-1], True);  pow_83 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:157 in rms_norm, code: hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        add_269: "f32[128, 901, 1]" = torch.ops.aten.add.Tensor(mean_82, 1e-05);  mean_82 = None
        rsqrt_82: "f32[128, 901, 1]" = torch.ops.aten.rsqrt.default(add_269);  add_269 = None
        mul_335: "f32[128, 901, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_925, rsqrt_82);  convert_element_type_925 = rsqrt_82 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:158 in rms_norm, code: return hidden_states.to(input_dtype)
        convert_element_type_926: "bf16[128, 901, 128]" = torch.ops.prims.convert_element_type.default(mul_335, torch.bfloat16);  mul_335 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_425: "bf16[115328, 128]" = torch.ops.aten.reshape.default(convert_element_type_926, [115328, 128])
        mm_167: "bf16[115328, 1024]" = torch.ops.aten.mm.default(view_425, permute_104);  view_425 = permute_104 = None
        view_426: "bf16[128, 901, 1024]" = torch.ops.aten.reshape.default(mm_167, [128, 901, 1024]);  mm_167 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:148 in forward, code: gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        split_41 = torch.ops.aten.split.Tensor(view_426, 512, -1);  view_426 = None
        getitem_250: "bf16[128, 901, 512]" = split_41[0]
        getitem_251: "bf16[128, 901, 512]" = split_41[1];  split_41 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:149 in forward, code: return self.down_proj(F.silu(gate) * up)
        convert_element_type_930: "f32[128, 901, 512]" = torch.ops.prims.convert_element_type.default(getitem_250, torch.float32);  getitem_250 = None
        sigmoid_41: "f32[128, 901, 512]" = torch.ops.aten.sigmoid.default(convert_element_type_930)
        mul_336: "f32[128, 901, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_930, sigmoid_41);  convert_element_type_930 = sigmoid_41 = None
        convert_element_type_931: "bf16[128, 901, 512]" = torch.ops.prims.convert_element_type.default(mul_336, torch.bfloat16);  mul_336 = None
        mul_337: "bf16[128, 901, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_931, getitem_251);  convert_element_type_931 = getitem_251 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_427: "bf16[115328, 512]" = torch.ops.aten.reshape.default(mul_337, [115328, 512]);  mul_337 = None
        mm_168: "bf16[115328, 128]" = torch.ops.aten.mm.default(view_427, permute_105);  view_427 = permute_105 = None
        view_428: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(mm_168, [128, 901, 128]);  mm_168 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:82 in forward, code: hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)
        add_270: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(convert_element_type_926, view_428);  convert_element_type_926 = view_428 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:154 in rms_norm, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_935: "f32[128, 901, 128]" = torch.ops.prims.convert_element_type.default(add_270, torch.float32);  add_270 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:156 in rms_norm, code: variance = hidden_states.square().mean(-1, keepdim=True)
        pow_84: "f32[128, 901, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_935, 2)
        mean_83: "f32[128, 901, 1]" = torch.ops.aten.mean.dim(pow_84, [-1], True);  pow_84 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:157 in rms_norm, code: hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        add_271: "f32[128, 901, 1]" = torch.ops.aten.add.Tensor(mean_83, 1e-05);  mean_83 = None
        rsqrt_83: "f32[128, 901, 1]" = torch.ops.aten.rsqrt.default(add_271);  add_271 = None
        mul_338: "f32[128, 901, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_935, rsqrt_83);  convert_element_type_935 = rsqrt_83 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:158 in rms_norm, code: return hidden_states.to(input_dtype)
        convert_element_type_936: "bf16[128, 901, 128]" = torch.ops.prims.convert_element_type.default(mul_338, torch.bfloat16);  mul_338 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_429: "bf16[115328, 128]" = torch.ops.aten.reshape.default(convert_element_type_936, [115328, 128])
        mm_169: "bf16[115328, 384]" = torch.ops.aten.mm.default(view_429, permute_106);  view_429 = permute_106 = None
        view_430: "bf16[128, 901, 384]" = torch.ops.aten.reshape.default(mm_169, [128, 901, 384]);  mm_169 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:119 in forward, code: qkv = qkv.view(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        view_431: "bf16[128, 901, 6, 64]" = torch.ops.aten.reshape.default(view_430, [128, 901, 6, 64]);  view_430 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:120 in forward, code: query = qkv[:, :, :self.num_heads]
        slice_552: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_431, 2, 0, 2)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:121 in forward, code: key = qkv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
        slice_555: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_431, 2, 2, 4)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:122 in forward, code: value = qkv[:, :, self.num_heads + self.num_key_value_heads:]
        slice_558: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_431, 2, 4, 9223372036854775807);  view_431 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:34 in apply_rotary_pos_emb, code: q = q.to(cos.dtype)
        convert_element_type_940: "f32[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(slice_552, torch.float32);  slice_552 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:35 in apply_rotary_pos_emb, code: k = k.to(cos.dtype)
        convert_element_type_941: "f32[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(slice_555, torch.float32);  slice_555 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:37 in apply_rotary_pos_emb, code: q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
        mul_339: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_940, unsqueeze_96)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:25 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_559: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_940, 3, 0, 32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:26 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_560: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_940, 3, 32, 9223372036854775807);  convert_element_type_940 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:27 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_84: "f32[128, 901, 2, 32]" = torch.ops.aten.neg.default(slice_560);  slice_560 = None
        cat_86: "f32[128, 901, 2, 64]" = torch.ops.aten.cat.default([neg_84, slice_559], -1);  neg_84 = slice_559 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:37 in apply_rotary_pos_emb, code: q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
        mul_340: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(cat_86, unsqueeze_97);  cat_86 = None
        add_272: "f32[128, 901, 2, 64]" = torch.ops.aten.add.Tensor(mul_339, mul_340);  mul_339 = mul_340 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:38 in apply_rotary_pos_emb, code: k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
        mul_341: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_941, unsqueeze_96)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:25 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_561: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_941, 3, 0, 32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:26 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_562: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_941, 3, 32, 9223372036854775807);  convert_element_type_941 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:27 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_85: "f32[128, 901, 2, 32]" = torch.ops.aten.neg.default(slice_562);  slice_562 = None
        cat_87: "f32[128, 901, 2, 64]" = torch.ops.aten.cat.default([neg_85, slice_561], -1);  neg_85 = slice_561 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:38 in apply_rotary_pos_emb, code: k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
        mul_342: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(cat_87, unsqueeze_97);  cat_87 = None
        add_273: "f32[128, 901, 2, 64]" = torch.ops.aten.add.Tensor(mul_341, mul_342);  mul_341 = mul_342 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:40 in apply_rotary_pos_emb, code: return q_embed.to(orig_dtype), k_embed.to(orig_dtype)
        convert_element_type_942: "bf16[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(add_272, torch.bfloat16);  add_272 = None
        convert_element_type_943: "bf16[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(add_273, torch.bfloat16);  add_273 = None
        
         # File: C:\Users\arroy\anaconda3\envs\hrm_v1\lib\site-packages\flash_attn\flash_attn_interface.py:839 in forward, code: out_padded, softmax_lse, S_dmask, rng_state = _wrapped_flash_attn_forward(
        _flash_attn_forward_42 = torch.ops.flash_attn._flash_attn_forward.default(convert_element_type_942, convert_element_type_943, slice_558, 0.0, 0.125, False, -1, -1, 0.0, None, False);  convert_element_type_942 = convert_element_type_943 = slice_558 = None
        getitem_252: "bf16[128, 901, 2, 64]" = _flash_attn_forward_42[0];  _flash_attn_forward_42 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:135 in forward, code: attn_output = attn_output.view(batch_size, seq_len, self.output_size)  # type: ignore
        view_432: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(getitem_252, [128, 901, 128]);  getitem_252 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_433: "bf16[115328, 128]" = torch.ops.aten.reshape.default(view_432, [115328, 128]);  view_432 = None
        mm_170: "bf16[115328, 128]" = torch.ops.aten.mm.default(view_433, permute_107);  view_433 = permute_107 = None
        view_434: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(mm_170, [128, 901, 128]);  mm_170 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:80 in forward, code: hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        add_274: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(convert_element_type_936, view_434);  convert_element_type_936 = view_434 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:154 in rms_norm, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_947: "f32[128, 901, 128]" = torch.ops.prims.convert_element_type.default(add_274, torch.float32);  add_274 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:156 in rms_norm, code: variance = hidden_states.square().mean(-1, keepdim=True)
        pow_85: "f32[128, 901, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_947, 2)
        mean_84: "f32[128, 901, 1]" = torch.ops.aten.mean.dim(pow_85, [-1], True);  pow_85 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:157 in rms_norm, code: hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        add_275: "f32[128, 901, 1]" = torch.ops.aten.add.Tensor(mean_84, 1e-05);  mean_84 = None
        rsqrt_84: "f32[128, 901, 1]" = torch.ops.aten.rsqrt.default(add_275);  add_275 = None
        mul_343: "f32[128, 901, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_947, rsqrt_84);  convert_element_type_947 = rsqrt_84 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:158 in rms_norm, code: return hidden_states.to(input_dtype)
        convert_element_type_948: "bf16[128, 901, 128]" = torch.ops.prims.convert_element_type.default(mul_343, torch.bfloat16);  mul_343 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_435: "bf16[115328, 128]" = torch.ops.aten.reshape.default(convert_element_type_948, [115328, 128])
        mm_171: "bf16[115328, 1024]" = torch.ops.aten.mm.default(view_435, permute_108);  view_435 = permute_108 = None
        view_436: "bf16[128, 901, 1024]" = torch.ops.aten.reshape.default(mm_171, [128, 901, 1024]);  mm_171 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:148 in forward, code: gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        split_42 = torch.ops.aten.split.Tensor(view_436, 512, -1);  view_436 = None
        getitem_256: "bf16[128, 901, 512]" = split_42[0]
        getitem_257: "bf16[128, 901, 512]" = split_42[1];  split_42 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:149 in forward, code: return self.down_proj(F.silu(gate) * up)
        convert_element_type_952: "f32[128, 901, 512]" = torch.ops.prims.convert_element_type.default(getitem_256, torch.float32);  getitem_256 = None
        sigmoid_42: "f32[128, 901, 512]" = torch.ops.aten.sigmoid.default(convert_element_type_952)
        mul_344: "f32[128, 901, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_952, sigmoid_42);  convert_element_type_952 = sigmoid_42 = None
        convert_element_type_953: "bf16[128, 901, 512]" = torch.ops.prims.convert_element_type.default(mul_344, torch.bfloat16);  mul_344 = None
        mul_345: "bf16[128, 901, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_953, getitem_257);  convert_element_type_953 = getitem_257 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_437: "bf16[115328, 512]" = torch.ops.aten.reshape.default(mul_345, [115328, 512]);  mul_345 = None
        mm_172: "bf16[115328, 128]" = torch.ops.aten.mm.default(view_437, permute_109);  view_437 = permute_109 = None
        view_438: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(mm_172, [128, 901, 128]);  mm_172 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:82 in forward, code: hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)
        add_276: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(convert_element_type_948, view_438);  convert_element_type_948 = view_438 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:154 in rms_norm, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_957: "f32[128, 901, 128]" = torch.ops.prims.convert_element_type.default(add_276, torch.float32);  add_276 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:156 in rms_norm, code: variance = hidden_states.square().mean(-1, keepdim=True)
        pow_86: "f32[128, 901, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_957, 2)
        mean_85: "f32[128, 901, 1]" = torch.ops.aten.mean.dim(pow_86, [-1], True);  pow_86 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:157 in rms_norm, code: hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        add_277: "f32[128, 901, 1]" = torch.ops.aten.add.Tensor(mean_85, 1e-05);  mean_85 = None
        rsqrt_85: "f32[128, 901, 1]" = torch.ops.aten.rsqrt.default(add_277);  add_277 = None
        mul_346: "f32[128, 901, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_957, rsqrt_85);  convert_element_type_957 = rsqrt_85 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:158 in rms_norm, code: return hidden_states.to(input_dtype)
        convert_element_type_958: "bf16[128, 901, 128]" = torch.ops.prims.convert_element_type.default(mul_346, torch.bfloat16);  mul_346 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_439: "bf16[115328, 128]" = torch.ops.aten.reshape.default(convert_element_type_958, [115328, 128])
        mm_173: "bf16[115328, 384]" = torch.ops.aten.mm.default(view_439, permute_110);  view_439 = permute_110 = None
        view_440: "bf16[128, 901, 384]" = torch.ops.aten.reshape.default(mm_173, [128, 901, 384]);  mm_173 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:119 in forward, code: qkv = qkv.view(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        view_441: "bf16[128, 901, 6, 64]" = torch.ops.aten.reshape.default(view_440, [128, 901, 6, 64]);  view_440 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:120 in forward, code: query = qkv[:, :, :self.num_heads]
        slice_565: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_441, 2, 0, 2)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:121 in forward, code: key = qkv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
        slice_568: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_441, 2, 2, 4)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:122 in forward, code: value = qkv[:, :, self.num_heads + self.num_key_value_heads:]
        slice_571: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_441, 2, 4, 9223372036854775807);  view_441 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:34 in apply_rotary_pos_emb, code: q = q.to(cos.dtype)
        convert_element_type_962: "f32[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(slice_565, torch.float32);  slice_565 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:35 in apply_rotary_pos_emb, code: k = k.to(cos.dtype)
        convert_element_type_963: "f32[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(slice_568, torch.float32);  slice_568 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:37 in apply_rotary_pos_emb, code: q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
        mul_347: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_962, unsqueeze_96)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:25 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_572: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_962, 3, 0, 32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:26 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_573: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_962, 3, 32, 9223372036854775807);  convert_element_type_962 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:27 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_86: "f32[128, 901, 2, 32]" = torch.ops.aten.neg.default(slice_573);  slice_573 = None
        cat_88: "f32[128, 901, 2, 64]" = torch.ops.aten.cat.default([neg_86, slice_572], -1);  neg_86 = slice_572 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:37 in apply_rotary_pos_emb, code: q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
        mul_348: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(cat_88, unsqueeze_97);  cat_88 = None
        add_278: "f32[128, 901, 2, 64]" = torch.ops.aten.add.Tensor(mul_347, mul_348);  mul_347 = mul_348 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:38 in apply_rotary_pos_emb, code: k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
        mul_349: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_963, unsqueeze_96)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:25 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_574: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_963, 3, 0, 32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:26 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_575: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_963, 3, 32, 9223372036854775807);  convert_element_type_963 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:27 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_87: "f32[128, 901, 2, 32]" = torch.ops.aten.neg.default(slice_575);  slice_575 = None
        cat_89: "f32[128, 901, 2, 64]" = torch.ops.aten.cat.default([neg_87, slice_574], -1);  neg_87 = slice_574 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:38 in apply_rotary_pos_emb, code: k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
        mul_350: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(cat_89, unsqueeze_97);  cat_89 = None
        add_279: "f32[128, 901, 2, 64]" = torch.ops.aten.add.Tensor(mul_349, mul_350);  mul_349 = mul_350 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:40 in apply_rotary_pos_emb, code: return q_embed.to(orig_dtype), k_embed.to(orig_dtype)
        convert_element_type_964: "bf16[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(add_278, torch.bfloat16);  add_278 = None
        convert_element_type_965: "bf16[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(add_279, torch.bfloat16);  add_279 = None
        
         # File: C:\Users\arroy\anaconda3\envs\hrm_v1\lib\site-packages\flash_attn\flash_attn_interface.py:839 in forward, code: out_padded, softmax_lse, S_dmask, rng_state = _wrapped_flash_attn_forward(
        _flash_attn_forward_43 = torch.ops.flash_attn._flash_attn_forward.default(convert_element_type_964, convert_element_type_965, slice_571, 0.0, 0.125, False, -1, -1, 0.0, None, False);  convert_element_type_964 = convert_element_type_965 = slice_571 = None
        getitem_258: "bf16[128, 901, 2, 64]" = _flash_attn_forward_43[0];  _flash_attn_forward_43 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:135 in forward, code: attn_output = attn_output.view(batch_size, seq_len, self.output_size)  # type: ignore
        view_442: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(getitem_258, [128, 901, 128]);  getitem_258 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_443: "bf16[115328, 128]" = torch.ops.aten.reshape.default(view_442, [115328, 128]);  view_442 = None
        mm_174: "bf16[115328, 128]" = torch.ops.aten.mm.default(view_443, permute_111);  view_443 = permute_111 = None
        view_444: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(mm_174, [128, 901, 128]);  mm_174 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:80 in forward, code: hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        add_280: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(convert_element_type_958, view_444);  convert_element_type_958 = view_444 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:154 in rms_norm, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_969: "f32[128, 901, 128]" = torch.ops.prims.convert_element_type.default(add_280, torch.float32);  add_280 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:156 in rms_norm, code: variance = hidden_states.square().mean(-1, keepdim=True)
        pow_87: "f32[128, 901, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_969, 2)
        mean_86: "f32[128, 901, 1]" = torch.ops.aten.mean.dim(pow_87, [-1], True);  pow_87 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:157 in rms_norm, code: hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        add_281: "f32[128, 901, 1]" = torch.ops.aten.add.Tensor(mean_86, 1e-05);  mean_86 = None
        rsqrt_86: "f32[128, 901, 1]" = torch.ops.aten.rsqrt.default(add_281);  add_281 = None
        mul_351: "f32[128, 901, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_969, rsqrt_86);  convert_element_type_969 = rsqrt_86 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:158 in rms_norm, code: return hidden_states.to(input_dtype)
        convert_element_type_970: "bf16[128, 901, 128]" = torch.ops.prims.convert_element_type.default(mul_351, torch.bfloat16);  mul_351 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_445: "bf16[115328, 128]" = torch.ops.aten.reshape.default(convert_element_type_970, [115328, 128])
        mm_175: "bf16[115328, 1024]" = torch.ops.aten.mm.default(view_445, permute_112);  view_445 = permute_112 = None
        view_446: "bf16[128, 901, 1024]" = torch.ops.aten.reshape.default(mm_175, [128, 901, 1024]);  mm_175 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:148 in forward, code: gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        split_43 = torch.ops.aten.split.Tensor(view_446, 512, -1);  view_446 = None
        getitem_262: "bf16[128, 901, 512]" = split_43[0]
        getitem_263: "bf16[128, 901, 512]" = split_43[1];  split_43 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:149 in forward, code: return self.down_proj(F.silu(gate) * up)
        convert_element_type_974: "f32[128, 901, 512]" = torch.ops.prims.convert_element_type.default(getitem_262, torch.float32);  getitem_262 = None
        sigmoid_43: "f32[128, 901, 512]" = torch.ops.aten.sigmoid.default(convert_element_type_974)
        mul_352: "f32[128, 901, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_974, sigmoid_43);  convert_element_type_974 = sigmoid_43 = None
        convert_element_type_975: "bf16[128, 901, 512]" = torch.ops.prims.convert_element_type.default(mul_352, torch.bfloat16);  mul_352 = None
        mul_353: "bf16[128, 901, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_975, getitem_263);  convert_element_type_975 = getitem_263 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_447: "bf16[115328, 512]" = torch.ops.aten.reshape.default(mul_353, [115328, 512]);  mul_353 = None
        mm_176: "bf16[115328, 128]" = torch.ops.aten.mm.default(view_447, permute_113);  view_447 = permute_113 = None
        view_448: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(mm_176, [128, 901, 128]);  mm_176 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:82 in forward, code: hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)
        add_282: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(convert_element_type_970, view_448);  convert_element_type_970 = view_448 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:154 in rms_norm, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_979: "f32[128, 901, 128]" = torch.ops.prims.convert_element_type.default(add_282, torch.float32);  add_282 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:156 in rms_norm, code: variance = hidden_states.square().mean(-1, keepdim=True)
        pow_88: "f32[128, 901, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_979, 2)
        mean_87: "f32[128, 901, 1]" = torch.ops.aten.mean.dim(pow_88, [-1], True);  pow_88 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:157 in rms_norm, code: hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        add_283: "f32[128, 901, 1]" = torch.ops.aten.add.Tensor(mean_87, 1e-05);  mean_87 = None
        rsqrt_87: "f32[128, 901, 1]" = torch.ops.aten.rsqrt.default(add_283);  add_283 = None
        mul_354: "f32[128, 901, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_979, rsqrt_87);  convert_element_type_979 = rsqrt_87 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:158 in rms_norm, code: return hidden_states.to(input_dtype)
        convert_element_type_980: "bf16[128, 901, 128]" = torch.ops.prims.convert_element_type.default(mul_354, torch.bfloat16);  mul_354 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:94 in forward, code: hidden_states = hidden_states + input_injection
        add_284: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(convert_element_type_804, convert_element_type_980);  convert_element_type_804 = convert_element_type_980 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_449: "bf16[115328, 128]" = torch.ops.aten.reshape.default(add_284, [115328, 128])
        mm_177: "bf16[115328, 384]" = torch.ops.aten.mm.default(view_449, permute_130);  view_449 = permute_130 = None
        view_450: "bf16[128, 901, 384]" = torch.ops.aten.reshape.default(mm_177, [128, 901, 384]);  mm_177 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:119 in forward, code: qkv = qkv.view(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        view_451: "bf16[128, 901, 6, 64]" = torch.ops.aten.reshape.default(view_450, [128, 901, 6, 64]);  view_450 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:120 in forward, code: query = qkv[:, :, :self.num_heads]
        slice_578: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_451, 2, 0, 2)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:121 in forward, code: key = qkv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
        slice_581: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_451, 2, 2, 4)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:122 in forward, code: value = qkv[:, :, self.num_heads + self.num_key_value_heads:]
        slice_584: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_451, 2, 4, 9223372036854775807);  view_451 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:34 in apply_rotary_pos_emb, code: q = q.to(cos.dtype)
        convert_element_type_984: "f32[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(slice_578, torch.float32);  slice_578 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:35 in apply_rotary_pos_emb, code: k = k.to(cos.dtype)
        convert_element_type_985: "f32[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(slice_581, torch.float32);  slice_581 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:37 in apply_rotary_pos_emb, code: q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
        mul_355: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_984, unsqueeze_96)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:25 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_585: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_984, 3, 0, 32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:26 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_586: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_984, 3, 32, 9223372036854775807);  convert_element_type_984 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:27 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_88: "f32[128, 901, 2, 32]" = torch.ops.aten.neg.default(slice_586);  slice_586 = None
        cat_90: "f32[128, 901, 2, 64]" = torch.ops.aten.cat.default([neg_88, slice_585], -1);  neg_88 = slice_585 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:37 in apply_rotary_pos_emb, code: q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
        mul_356: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(cat_90, unsqueeze_97);  cat_90 = None
        add_285: "f32[128, 901, 2, 64]" = torch.ops.aten.add.Tensor(mul_355, mul_356);  mul_355 = mul_356 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:38 in apply_rotary_pos_emb, code: k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
        mul_357: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_985, unsqueeze_96)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:25 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_587: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_985, 3, 0, 32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:26 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_588: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_985, 3, 32, 9223372036854775807);  convert_element_type_985 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:27 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_89: "f32[128, 901, 2, 32]" = torch.ops.aten.neg.default(slice_588);  slice_588 = None
        cat_91: "f32[128, 901, 2, 64]" = torch.ops.aten.cat.default([neg_89, slice_587], -1);  neg_89 = slice_587 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:38 in apply_rotary_pos_emb, code: k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
        mul_358: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(cat_91, unsqueeze_97);  cat_91 = None
        add_286: "f32[128, 901, 2, 64]" = torch.ops.aten.add.Tensor(mul_357, mul_358);  mul_357 = mul_358 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:40 in apply_rotary_pos_emb, code: return q_embed.to(orig_dtype), k_embed.to(orig_dtype)
        convert_element_type_986: "bf16[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(add_285, torch.bfloat16);  add_285 = None
        convert_element_type_987: "bf16[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(add_286, torch.bfloat16);  add_286 = None
        
         # File: C:\Users\arroy\anaconda3\envs\hrm_v1\lib\site-packages\flash_attn\flash_attn_interface.py:839 in forward, code: out_padded, softmax_lse, S_dmask, rng_state = _wrapped_flash_attn_forward(
        _flash_attn_forward_44 = torch.ops.flash_attn._flash_attn_forward.default(convert_element_type_986, convert_element_type_987, slice_584, 0.0, 0.125, False, -1, -1, 0.0, None, False);  convert_element_type_986 = convert_element_type_987 = slice_584 = None
        getitem_264: "bf16[128, 901, 2, 64]" = _flash_attn_forward_44[0];  _flash_attn_forward_44 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:135 in forward, code: attn_output = attn_output.view(batch_size, seq_len, self.output_size)  # type: ignore
        view_452: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(getitem_264, [128, 901, 128]);  getitem_264 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_453: "bf16[115328, 128]" = torch.ops.aten.reshape.default(view_452, [115328, 128]);  view_452 = None
        mm_178: "bf16[115328, 128]" = torch.ops.aten.mm.default(view_453, permute_131);  view_453 = permute_131 = None
        view_454: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(mm_178, [128, 901, 128]);  mm_178 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:80 in forward, code: hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        add_287: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(add_284, view_454);  add_284 = view_454 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:154 in rms_norm, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_991: "f32[128, 901, 128]" = torch.ops.prims.convert_element_type.default(add_287, torch.float32);  add_287 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:156 in rms_norm, code: variance = hidden_states.square().mean(-1, keepdim=True)
        pow_89: "f32[128, 901, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_991, 2)
        mean_88: "f32[128, 901, 1]" = torch.ops.aten.mean.dim(pow_89, [-1], True);  pow_89 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:157 in rms_norm, code: hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        add_288: "f32[128, 901, 1]" = torch.ops.aten.add.Tensor(mean_88, 1e-05);  mean_88 = None
        rsqrt_88: "f32[128, 901, 1]" = torch.ops.aten.rsqrt.default(add_288);  add_288 = None
        mul_359: "f32[128, 901, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_991, rsqrt_88);  convert_element_type_991 = rsqrt_88 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:158 in rms_norm, code: return hidden_states.to(input_dtype)
        convert_element_type_992: "bf16[128, 901, 128]" = torch.ops.prims.convert_element_type.default(mul_359, torch.bfloat16);  mul_359 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_455: "bf16[115328, 128]" = torch.ops.aten.reshape.default(convert_element_type_992, [115328, 128])
        mm_179: "bf16[115328, 1024]" = torch.ops.aten.mm.default(view_455, permute_132);  view_455 = permute_132 = None
        view_456: "bf16[128, 901, 1024]" = torch.ops.aten.reshape.default(mm_179, [128, 901, 1024]);  mm_179 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:148 in forward, code: gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        split_44 = torch.ops.aten.split.Tensor(view_456, 512, -1);  view_456 = None
        getitem_268: "bf16[128, 901, 512]" = split_44[0]
        getitem_269: "bf16[128, 901, 512]" = split_44[1];  split_44 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:149 in forward, code: return self.down_proj(F.silu(gate) * up)
        convert_element_type_996: "f32[128, 901, 512]" = torch.ops.prims.convert_element_type.default(getitem_268, torch.float32);  getitem_268 = None
        sigmoid_44: "f32[128, 901, 512]" = torch.ops.aten.sigmoid.default(convert_element_type_996)
        mul_360: "f32[128, 901, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_996, sigmoid_44);  convert_element_type_996 = sigmoid_44 = None
        convert_element_type_997: "bf16[128, 901, 512]" = torch.ops.prims.convert_element_type.default(mul_360, torch.bfloat16);  mul_360 = None
        mul_361: "bf16[128, 901, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_997, getitem_269);  convert_element_type_997 = getitem_269 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_457: "bf16[115328, 512]" = torch.ops.aten.reshape.default(mul_361, [115328, 512]);  mul_361 = None
        mm_180: "bf16[115328, 128]" = torch.ops.aten.mm.default(view_457, permute_133);  view_457 = permute_133 = None
        view_458: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(mm_180, [128, 901, 128]);  mm_180 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:82 in forward, code: hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)
        add_289: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(convert_element_type_992, view_458);  convert_element_type_992 = view_458 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:154 in rms_norm, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_1001: "f32[128, 901, 128]" = torch.ops.prims.convert_element_type.default(add_289, torch.float32);  add_289 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:156 in rms_norm, code: variance = hidden_states.square().mean(-1, keepdim=True)
        pow_90: "f32[128, 901, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_1001, 2)
        mean_89: "f32[128, 901, 1]" = torch.ops.aten.mean.dim(pow_90, [-1], True);  pow_90 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:157 in rms_norm, code: hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        add_290: "f32[128, 901, 1]" = torch.ops.aten.add.Tensor(mean_89, 1e-05);  mean_89 = None
        rsqrt_89: "f32[128, 901, 1]" = torch.ops.aten.rsqrt.default(add_290);  add_290 = None
        mul_362: "f32[128, 901, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_1001, rsqrt_89);  convert_element_type_1001 = rsqrt_89 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:158 in rms_norm, code: return hidden_states.to(input_dtype)
        convert_element_type_1002: "bf16[128, 901, 128]" = torch.ops.prims.convert_element_type.default(mul_362, torch.bfloat16);  mul_362 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_459: "bf16[115328, 128]" = torch.ops.aten.reshape.default(convert_element_type_1002, [115328, 128])
        mm_181: "bf16[115328, 384]" = torch.ops.aten.mm.default(view_459, permute_134);  view_459 = permute_134 = None
        view_460: "bf16[128, 901, 384]" = torch.ops.aten.reshape.default(mm_181, [128, 901, 384]);  mm_181 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:119 in forward, code: qkv = qkv.view(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        view_461: "bf16[128, 901, 6, 64]" = torch.ops.aten.reshape.default(view_460, [128, 901, 6, 64]);  view_460 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:120 in forward, code: query = qkv[:, :, :self.num_heads]
        slice_591: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_461, 2, 0, 2)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:121 in forward, code: key = qkv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
        slice_594: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_461, 2, 2, 4)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:122 in forward, code: value = qkv[:, :, self.num_heads + self.num_key_value_heads:]
        slice_597: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_461, 2, 4, 9223372036854775807);  view_461 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:34 in apply_rotary_pos_emb, code: q = q.to(cos.dtype)
        convert_element_type_1006: "f32[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(slice_591, torch.float32);  slice_591 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:35 in apply_rotary_pos_emb, code: k = k.to(cos.dtype)
        convert_element_type_1007: "f32[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(slice_594, torch.float32);  slice_594 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:37 in apply_rotary_pos_emb, code: q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
        mul_363: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_1006, unsqueeze_96)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:25 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_598: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_1006, 3, 0, 32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:26 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_599: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_1006, 3, 32, 9223372036854775807);  convert_element_type_1006 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:27 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_90: "f32[128, 901, 2, 32]" = torch.ops.aten.neg.default(slice_599);  slice_599 = None
        cat_92: "f32[128, 901, 2, 64]" = torch.ops.aten.cat.default([neg_90, slice_598], -1);  neg_90 = slice_598 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:37 in apply_rotary_pos_emb, code: q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
        mul_364: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(cat_92, unsqueeze_97);  cat_92 = None
        add_291: "f32[128, 901, 2, 64]" = torch.ops.aten.add.Tensor(mul_363, mul_364);  mul_363 = mul_364 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:38 in apply_rotary_pos_emb, code: k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
        mul_365: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_1007, unsqueeze_96)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:25 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_600: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_1007, 3, 0, 32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:26 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_601: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_1007, 3, 32, 9223372036854775807);  convert_element_type_1007 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:27 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_91: "f32[128, 901, 2, 32]" = torch.ops.aten.neg.default(slice_601);  slice_601 = None
        cat_93: "f32[128, 901, 2, 64]" = torch.ops.aten.cat.default([neg_91, slice_600], -1);  neg_91 = slice_600 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:38 in apply_rotary_pos_emb, code: k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
        mul_366: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(cat_93, unsqueeze_97);  cat_93 = None
        add_292: "f32[128, 901, 2, 64]" = torch.ops.aten.add.Tensor(mul_365, mul_366);  mul_365 = mul_366 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:40 in apply_rotary_pos_emb, code: return q_embed.to(orig_dtype), k_embed.to(orig_dtype)
        convert_element_type_1008: "bf16[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(add_291, torch.bfloat16);  add_291 = None
        convert_element_type_1009: "bf16[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(add_292, torch.bfloat16);  add_292 = None
        
         # File: C:\Users\arroy\anaconda3\envs\hrm_v1\lib\site-packages\flash_attn\flash_attn_interface.py:839 in forward, code: out_padded, softmax_lse, S_dmask, rng_state = _wrapped_flash_attn_forward(
        _flash_attn_forward_45 = torch.ops.flash_attn._flash_attn_forward.default(convert_element_type_1008, convert_element_type_1009, slice_597, 0.0, 0.125, False, -1, -1, 0.0, None, False);  convert_element_type_1008 = convert_element_type_1009 = slice_597 = None
        getitem_270: "bf16[128, 901, 2, 64]" = _flash_attn_forward_45[0];  _flash_attn_forward_45 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:135 in forward, code: attn_output = attn_output.view(batch_size, seq_len, self.output_size)  # type: ignore
        view_462: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(getitem_270, [128, 901, 128]);  getitem_270 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_463: "bf16[115328, 128]" = torch.ops.aten.reshape.default(view_462, [115328, 128]);  view_462 = None
        mm_182: "bf16[115328, 128]" = torch.ops.aten.mm.default(view_463, permute_135);  view_463 = permute_135 = None
        view_464: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(mm_182, [128, 901, 128]);  mm_182 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:80 in forward, code: hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        add_293: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(convert_element_type_1002, view_464);  convert_element_type_1002 = view_464 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:154 in rms_norm, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_1013: "f32[128, 901, 128]" = torch.ops.prims.convert_element_type.default(add_293, torch.float32);  add_293 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:156 in rms_norm, code: variance = hidden_states.square().mean(-1, keepdim=True)
        pow_91: "f32[128, 901, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_1013, 2)
        mean_90: "f32[128, 901, 1]" = torch.ops.aten.mean.dim(pow_91, [-1], True);  pow_91 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:157 in rms_norm, code: hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        add_294: "f32[128, 901, 1]" = torch.ops.aten.add.Tensor(mean_90, 1e-05);  mean_90 = None
        rsqrt_90: "f32[128, 901, 1]" = torch.ops.aten.rsqrt.default(add_294);  add_294 = None
        mul_367: "f32[128, 901, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_1013, rsqrt_90);  convert_element_type_1013 = rsqrt_90 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:158 in rms_norm, code: return hidden_states.to(input_dtype)
        convert_element_type_1014: "bf16[128, 901, 128]" = torch.ops.prims.convert_element_type.default(mul_367, torch.bfloat16);  mul_367 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_465: "bf16[115328, 128]" = torch.ops.aten.reshape.default(convert_element_type_1014, [115328, 128])
        mm_183: "bf16[115328, 1024]" = torch.ops.aten.mm.default(view_465, permute_136);  view_465 = permute_136 = None
        view_466: "bf16[128, 901, 1024]" = torch.ops.aten.reshape.default(mm_183, [128, 901, 1024]);  mm_183 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:148 in forward, code: gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        split_45 = torch.ops.aten.split.Tensor(view_466, 512, -1);  view_466 = None
        getitem_274: "bf16[128, 901, 512]" = split_45[0]
        getitem_275: "bf16[128, 901, 512]" = split_45[1];  split_45 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:149 in forward, code: return self.down_proj(F.silu(gate) * up)
        convert_element_type_1018: "f32[128, 901, 512]" = torch.ops.prims.convert_element_type.default(getitem_274, torch.float32);  getitem_274 = None
        sigmoid_45: "f32[128, 901, 512]" = torch.ops.aten.sigmoid.default(convert_element_type_1018)
        mul_368: "f32[128, 901, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_1018, sigmoid_45);  convert_element_type_1018 = sigmoid_45 = None
        convert_element_type_1019: "bf16[128, 901, 512]" = torch.ops.prims.convert_element_type.default(mul_368, torch.bfloat16);  mul_368 = None
        mul_369: "bf16[128, 901, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_1019, getitem_275);  convert_element_type_1019 = getitem_275 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_467: "bf16[115328, 512]" = torch.ops.aten.reshape.default(mul_369, [115328, 512]);  mul_369 = None
        mm_184: "bf16[115328, 128]" = torch.ops.aten.mm.default(view_467, permute_137);  view_467 = permute_137 = None
        view_468: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(mm_184, [128, 901, 128]);  mm_184 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:82 in forward, code: hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)
        add_295: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(convert_element_type_1014, view_468);  convert_element_type_1014 = view_468 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:154 in rms_norm, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_1023: "f32[128, 901, 128]" = torch.ops.prims.convert_element_type.default(add_295, torch.float32);  add_295 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:156 in rms_norm, code: variance = hidden_states.square().mean(-1, keepdim=True)
        pow_92: "f32[128, 901, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_1023, 2)
        mean_91: "f32[128, 901, 1]" = torch.ops.aten.mean.dim(pow_92, [-1], True);  pow_92 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:157 in rms_norm, code: hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        add_296: "f32[128, 901, 1]" = torch.ops.aten.add.Tensor(mean_91, 1e-05);  mean_91 = None
        rsqrt_91: "f32[128, 901, 1]" = torch.ops.aten.rsqrt.default(add_296);  add_296 = None
        mul_370: "f32[128, 901, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_1023, rsqrt_91);  convert_element_type_1023 = rsqrt_91 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:158 in rms_norm, code: return hidden_states.to(input_dtype)
        convert_element_type_1024: "bf16[128, 901, 128]" = torch.ops.prims.convert_element_type.default(mul_370, torch.bfloat16);  mul_370 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_469: "bf16[115328, 128]" = torch.ops.aten.reshape.default(convert_element_type_1024, [115328, 128])
        mm_185: "bf16[115328, 384]" = torch.ops.aten.mm.default(view_469, permute_138);  view_469 = permute_138 = None
        view_470: "bf16[128, 901, 384]" = torch.ops.aten.reshape.default(mm_185, [128, 901, 384]);  mm_185 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:119 in forward, code: qkv = qkv.view(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        view_471: "bf16[128, 901, 6, 64]" = torch.ops.aten.reshape.default(view_470, [128, 901, 6, 64]);  view_470 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:120 in forward, code: query = qkv[:, :, :self.num_heads]
        slice_604: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_471, 2, 0, 2)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:121 in forward, code: key = qkv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
        slice_607: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_471, 2, 2, 4)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:122 in forward, code: value = qkv[:, :, self.num_heads + self.num_key_value_heads:]
        slice_610: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_471, 2, 4, 9223372036854775807);  view_471 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:34 in apply_rotary_pos_emb, code: q = q.to(cos.dtype)
        convert_element_type_1028: "f32[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(slice_604, torch.float32);  slice_604 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:35 in apply_rotary_pos_emb, code: k = k.to(cos.dtype)
        convert_element_type_1029: "f32[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(slice_607, torch.float32);  slice_607 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:37 in apply_rotary_pos_emb, code: q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
        mul_371: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_1028, unsqueeze_96)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:25 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_611: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_1028, 3, 0, 32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:26 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_612: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_1028, 3, 32, 9223372036854775807);  convert_element_type_1028 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:27 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_92: "f32[128, 901, 2, 32]" = torch.ops.aten.neg.default(slice_612);  slice_612 = None
        cat_94: "f32[128, 901, 2, 64]" = torch.ops.aten.cat.default([neg_92, slice_611], -1);  neg_92 = slice_611 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:37 in apply_rotary_pos_emb, code: q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
        mul_372: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(cat_94, unsqueeze_97);  cat_94 = None
        add_297: "f32[128, 901, 2, 64]" = torch.ops.aten.add.Tensor(mul_371, mul_372);  mul_371 = mul_372 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:38 in apply_rotary_pos_emb, code: k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
        mul_373: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_1029, unsqueeze_96)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:25 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_613: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_1029, 3, 0, 32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:26 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_614: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_1029, 3, 32, 9223372036854775807);  convert_element_type_1029 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:27 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_93: "f32[128, 901, 2, 32]" = torch.ops.aten.neg.default(slice_614);  slice_614 = None
        cat_95: "f32[128, 901, 2, 64]" = torch.ops.aten.cat.default([neg_93, slice_613], -1);  neg_93 = slice_613 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:38 in apply_rotary_pos_emb, code: k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
        mul_374: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(cat_95, unsqueeze_97);  cat_95 = None
        add_298: "f32[128, 901, 2, 64]" = torch.ops.aten.add.Tensor(mul_373, mul_374);  mul_373 = mul_374 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:40 in apply_rotary_pos_emb, code: return q_embed.to(orig_dtype), k_embed.to(orig_dtype)
        convert_element_type_1030: "bf16[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(add_297, torch.bfloat16);  add_297 = None
        convert_element_type_1031: "bf16[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(add_298, torch.bfloat16);  add_298 = None
        
         # File: C:\Users\arroy\anaconda3\envs\hrm_v1\lib\site-packages\flash_attn\flash_attn_interface.py:839 in forward, code: out_padded, softmax_lse, S_dmask, rng_state = _wrapped_flash_attn_forward(
        _flash_attn_forward_46 = torch.ops.flash_attn._flash_attn_forward.default(convert_element_type_1030, convert_element_type_1031, slice_610, 0.0, 0.125, False, -1, -1, 0.0, None, False);  convert_element_type_1030 = convert_element_type_1031 = slice_610 = None
        getitem_276: "bf16[128, 901, 2, 64]" = _flash_attn_forward_46[0];  _flash_attn_forward_46 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:135 in forward, code: attn_output = attn_output.view(batch_size, seq_len, self.output_size)  # type: ignore
        view_472: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(getitem_276, [128, 901, 128]);  getitem_276 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_473: "bf16[115328, 128]" = torch.ops.aten.reshape.default(view_472, [115328, 128]);  view_472 = None
        mm_186: "bf16[115328, 128]" = torch.ops.aten.mm.default(view_473, permute_139);  view_473 = permute_139 = None
        view_474: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(mm_186, [128, 901, 128]);  mm_186 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:80 in forward, code: hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        add_299: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(convert_element_type_1024, view_474);  convert_element_type_1024 = view_474 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:154 in rms_norm, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_1035: "f32[128, 901, 128]" = torch.ops.prims.convert_element_type.default(add_299, torch.float32);  add_299 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:156 in rms_norm, code: variance = hidden_states.square().mean(-1, keepdim=True)
        pow_93: "f32[128, 901, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_1035, 2)
        mean_92: "f32[128, 901, 1]" = torch.ops.aten.mean.dim(pow_93, [-1], True);  pow_93 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:157 in rms_norm, code: hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        add_300: "f32[128, 901, 1]" = torch.ops.aten.add.Tensor(mean_92, 1e-05);  mean_92 = None
        rsqrt_92: "f32[128, 901, 1]" = torch.ops.aten.rsqrt.default(add_300);  add_300 = None
        mul_375: "f32[128, 901, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_1035, rsqrt_92);  convert_element_type_1035 = rsqrt_92 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:158 in rms_norm, code: return hidden_states.to(input_dtype)
        convert_element_type_1036: "bf16[128, 901, 128]" = torch.ops.prims.convert_element_type.default(mul_375, torch.bfloat16);  mul_375 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_475: "bf16[115328, 128]" = torch.ops.aten.reshape.default(convert_element_type_1036, [115328, 128])
        mm_187: "bf16[115328, 1024]" = torch.ops.aten.mm.default(view_475, permute_140);  view_475 = permute_140 = None
        view_476: "bf16[128, 901, 1024]" = torch.ops.aten.reshape.default(mm_187, [128, 901, 1024]);  mm_187 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:148 in forward, code: gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        split_46 = torch.ops.aten.split.Tensor(view_476, 512, -1);  view_476 = None
        getitem_280: "bf16[128, 901, 512]" = split_46[0]
        getitem_281: "bf16[128, 901, 512]" = split_46[1];  split_46 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:149 in forward, code: return self.down_proj(F.silu(gate) * up)
        convert_element_type_1040: "f32[128, 901, 512]" = torch.ops.prims.convert_element_type.default(getitem_280, torch.float32);  getitem_280 = None
        sigmoid_46: "f32[128, 901, 512]" = torch.ops.aten.sigmoid.default(convert_element_type_1040)
        mul_376: "f32[128, 901, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_1040, sigmoid_46);  convert_element_type_1040 = sigmoid_46 = None
        convert_element_type_1041: "bf16[128, 901, 512]" = torch.ops.prims.convert_element_type.default(mul_376, torch.bfloat16);  mul_376 = None
        mul_377: "bf16[128, 901, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_1041, getitem_281);  convert_element_type_1041 = getitem_281 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_477: "bf16[115328, 512]" = torch.ops.aten.reshape.default(mul_377, [115328, 512]);  mul_377 = None
        mm_188: "bf16[115328, 128]" = torch.ops.aten.mm.default(view_477, permute_141);  view_477 = permute_141 = None
        view_478: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(mm_188, [128, 901, 128]);  mm_188 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:82 in forward, code: hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)
        add_301: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(convert_element_type_1036, view_478);  convert_element_type_1036 = view_478 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:154 in rms_norm, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_1045: "f32[128, 901, 128]" = torch.ops.prims.convert_element_type.default(add_301, torch.float32);  add_301 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:156 in rms_norm, code: variance = hidden_states.square().mean(-1, keepdim=True)
        pow_94: "f32[128, 901, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_1045, 2)
        mean_93: "f32[128, 901, 1]" = torch.ops.aten.mean.dim(pow_94, [-1], True);  pow_94 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:157 in rms_norm, code: hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        add_302: "f32[128, 901, 1]" = torch.ops.aten.add.Tensor(mean_93, 1e-05);  mean_93 = None
        rsqrt_93: "f32[128, 901, 1]" = torch.ops.aten.rsqrt.default(add_302);  add_302 = None
        mul_378: "f32[128, 901, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_1045, rsqrt_93);  convert_element_type_1045 = rsqrt_93 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:158 in rms_norm, code: return hidden_states.to(input_dtype)
        convert_element_type_1046: "bf16[128, 901, 128]" = torch.ops.prims.convert_element_type.default(mul_378, torch.bfloat16);  mul_378 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_479: "bf16[115328, 128]" = torch.ops.aten.reshape.default(convert_element_type_1046, [115328, 128])
        mm_189: "bf16[115328, 384]" = torch.ops.aten.mm.default(view_479, permute_142);  view_479 = permute_142 = None
        view_480: "bf16[128, 901, 384]" = torch.ops.aten.reshape.default(mm_189, [128, 901, 384]);  mm_189 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:119 in forward, code: qkv = qkv.view(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        view_481: "bf16[128, 901, 6, 64]" = torch.ops.aten.reshape.default(view_480, [128, 901, 6, 64]);  view_480 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:120 in forward, code: query = qkv[:, :, :self.num_heads]
        slice_617: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_481, 2, 0, 2)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:121 in forward, code: key = qkv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
        slice_620: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_481, 2, 2, 4)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:122 in forward, code: value = qkv[:, :, self.num_heads + self.num_key_value_heads:]
        slice_623: "bf16[128, 901, 2, 64]" = torch.ops.aten.slice.Tensor(view_481, 2, 4, 9223372036854775807);  view_481 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:34 in apply_rotary_pos_emb, code: q = q.to(cos.dtype)
        convert_element_type_1050: "f32[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(slice_617, torch.float32);  slice_617 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:35 in apply_rotary_pos_emb, code: k = k.to(cos.dtype)
        convert_element_type_1051: "f32[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(slice_620, torch.float32);  slice_620 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:37 in apply_rotary_pos_emb, code: q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
        mul_379: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_1050, unsqueeze_96)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:25 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_624: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_1050, 3, 0, 32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:26 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_625: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_1050, 3, 32, 9223372036854775807);  convert_element_type_1050 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:27 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_94: "f32[128, 901, 2, 32]" = torch.ops.aten.neg.default(slice_625);  slice_625 = None
        cat_96: "f32[128, 901, 2, 64]" = torch.ops.aten.cat.default([neg_94, slice_624], -1);  neg_94 = slice_624 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:37 in apply_rotary_pos_emb, code: q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
        mul_380: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(cat_96, unsqueeze_97);  cat_96 = None
        add_303: "f32[128, 901, 2, 64]" = torch.ops.aten.add.Tensor(mul_379, mul_380);  mul_379 = mul_380 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:38 in apply_rotary_pos_emb, code: k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
        mul_381: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(convert_element_type_1051, unsqueeze_96);  unsqueeze_96 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:25 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_626: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_1051, 3, 0, 32)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:26 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_627: "f32[128, 901, 2, 32]" = torch.ops.aten.slice.Tensor(convert_element_type_1051, 3, 32, 9223372036854775807);  convert_element_type_1051 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:27 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_95: "f32[128, 901, 2, 32]" = torch.ops.aten.neg.default(slice_627);  slice_627 = None
        cat_97: "f32[128, 901, 2, 64]" = torch.ops.aten.cat.default([neg_95, slice_626], -1);  neg_95 = slice_626 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:38 in apply_rotary_pos_emb, code: k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
        mul_382: "f32[128, 901, 2, 64]" = torch.ops.aten.mul.Tensor(cat_97, unsqueeze_97);  cat_97 = unsqueeze_97 = None
        add_304: "f32[128, 901, 2, 64]" = torch.ops.aten.add.Tensor(mul_381, mul_382);  mul_381 = mul_382 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:40 in apply_rotary_pos_emb, code: return q_embed.to(orig_dtype), k_embed.to(orig_dtype)
        convert_element_type_1052: "bf16[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(add_303, torch.bfloat16);  add_303 = None
        convert_element_type_1053: "bf16[128, 901, 2, 64]" = torch.ops.prims.convert_element_type.default(add_304, torch.bfloat16);  add_304 = None
        
         # File: C:\Users\arroy\anaconda3\envs\hrm_v1\lib\site-packages\flash_attn\flash_attn_interface.py:839 in forward, code: out_padded, softmax_lse, S_dmask, rng_state = _wrapped_flash_attn_forward(
        _flash_attn_forward_47 = torch.ops.flash_attn._flash_attn_forward.default(convert_element_type_1052, convert_element_type_1053, slice_623, 0.0, 0.125, False, -1, -1, 0.0, None, False);  convert_element_type_1052 = convert_element_type_1053 = slice_623 = None
        getitem_282: "bf16[128, 901, 2, 64]" = _flash_attn_forward_47[0];  _flash_attn_forward_47 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:135 in forward, code: attn_output = attn_output.view(batch_size, seq_len, self.output_size)  # type: ignore
        view_482: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(getitem_282, [128, 901, 128]);  getitem_282 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_483: "bf16[115328, 128]" = torch.ops.aten.reshape.default(view_482, [115328, 128]);  view_482 = None
        mm_190: "bf16[115328, 128]" = torch.ops.aten.mm.default(view_483, permute_143);  view_483 = permute_143 = None
        view_484: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(mm_190, [128, 901, 128]);  mm_190 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:80 in forward, code: hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        add_305: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(convert_element_type_1046, view_484);  convert_element_type_1046 = view_484 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:154 in rms_norm, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_1057: "f32[128, 901, 128]" = torch.ops.prims.convert_element_type.default(add_305, torch.float32);  add_305 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:156 in rms_norm, code: variance = hidden_states.square().mean(-1, keepdim=True)
        pow_95: "f32[128, 901, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_1057, 2)
        mean_94: "f32[128, 901, 1]" = torch.ops.aten.mean.dim(pow_95, [-1], True);  pow_95 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:157 in rms_norm, code: hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        add_306: "f32[128, 901, 1]" = torch.ops.aten.add.Tensor(mean_94, 1e-05);  mean_94 = None
        rsqrt_94: "f32[128, 901, 1]" = torch.ops.aten.rsqrt.default(add_306);  add_306 = None
        mul_383: "f32[128, 901, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_1057, rsqrt_94);  convert_element_type_1057 = rsqrt_94 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:158 in rms_norm, code: return hidden_states.to(input_dtype)
        convert_element_type_1058: "bf16[128, 901, 128]" = torch.ops.prims.convert_element_type.default(mul_383, torch.bfloat16);  mul_383 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_485: "bf16[115328, 128]" = torch.ops.aten.reshape.default(convert_element_type_1058, [115328, 128])
        mm_191: "bf16[115328, 1024]" = torch.ops.aten.mm.default(view_485, permute_144);  view_485 = permute_144 = None
        view_486: "bf16[128, 901, 1024]" = torch.ops.aten.reshape.default(mm_191, [128, 901, 1024]);  mm_191 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:148 in forward, code: gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        split_47 = torch.ops.aten.split.Tensor(view_486, 512, -1);  view_486 = None
        getitem_286: "bf16[128, 901, 512]" = split_47[0]
        getitem_287: "bf16[128, 901, 512]" = split_47[1];  split_47 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:149 in forward, code: return self.down_proj(F.silu(gate) * up)
        convert_element_type_1062: "f32[128, 901, 512]" = torch.ops.prims.convert_element_type.default(getitem_286, torch.float32);  getitem_286 = None
        sigmoid_47: "f32[128, 901, 512]" = torch.ops.aten.sigmoid.default(convert_element_type_1062)
        mul_384: "f32[128, 901, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_1062, sigmoid_47);  convert_element_type_1062 = sigmoid_47 = None
        convert_element_type_1063: "bf16[128, 901, 512]" = torch.ops.prims.convert_element_type.default(mul_384, torch.bfloat16);  mul_384 = None
        mul_385: "bf16[128, 901, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_1063, getitem_287);  convert_element_type_1063 = getitem_287 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        view_487: "bf16[115328, 512]" = torch.ops.aten.reshape.default(mul_385, [115328, 512]);  mul_385 = None
        mm_192: "bf16[115328, 128]" = torch.ops.aten.mm.default(view_487, permute_145);  view_487 = permute_145 = None
        view_488: "bf16[128, 901, 128]" = torch.ops.aten.reshape.default(mm_192, [128, 901, 128]);  mm_192 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:82 in forward, code: hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)
        add_307: "bf16[128, 901, 128]" = torch.ops.aten.add.Tensor(convert_element_type_1058, view_488);  convert_element_type_1058 = view_488 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:154 in rms_norm, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_1067: "f32[128, 901, 128]" = torch.ops.prims.convert_element_type.default(add_307, torch.float32);  add_307 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:156 in rms_norm, code: variance = hidden_states.square().mean(-1, keepdim=True)
        pow_96: "f32[128, 901, 128]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_1067, 2)
        mean_95: "f32[128, 901, 1]" = torch.ops.aten.mean.dim(pow_96, [-1], True);  pow_96 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:157 in rms_norm, code: hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
        add_308: "f32[128, 901, 1]" = torch.ops.aten.add.Tensor(mean_95, 1e-05);  mean_95 = None
        rsqrt_95: "f32[128, 901, 1]" = torch.ops.aten.rsqrt.default(add_308);  add_308 = None
        mul_386: "f32[128, 901, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_1067, rsqrt_95);  convert_element_type_1067 = rsqrt_95 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:158 in rms_norm, code: return hidden_states.to(input_dtype)
        convert_element_type_1068: "bf16[128, 901, 128]" = torch.ops.prims.convert_element_type.default(mul_386, torch.bfloat16);  mul_386 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:211 in forward, code: q_logits = self.q_head(z_H[:, 0]).to(torch.float32)
        select_3: "bf16[128, 128]" = torch.ops.aten.select.int(convert_element_type_1068, 1, 0);  convert_element_type_1068 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        convert_element_type_1072: "bf16[2, 128]" = torch.ops.prims.convert_element_type.default(primals_52, torch.bfloat16);  primals_52 = None
        convert_element_type_1073: "bf16[2]" = torch.ops.prims.convert_element_type.default(primals_53, torch.bfloat16);  primals_53 = None
        permute_195: "bf16[128, 2]" = torch.ops.aten.permute.default(convert_element_type_1072, [1, 0]);  convert_element_type_1072 = None
        
        # No stacktrace found for following nodes
        mm_default: "bf16[128, 2]" = torch.ops.aten.mm.default(select_3, permute_195);  select_3 = permute_195 = None
        add_tensor: "bf16[128, 2]" = torch.ops.aten.add.Tensor(mm_default, convert_element_type_1073);  mm_default = convert_element_type_1073 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:211 in forward, code: q_logits = self.q_head(z_H[:, 0]).to(torch.float32)
        convert_element_type_1077: "f32[128, 2]" = torch.ops.prims.convert_element_type.default(add_tensor, torch.float32);  add_tensor = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:213 in forward, code: return new_carry, output, (q_logits[..., 0], q_logits[..., 1])
        select_4: "f32[128]" = torch.ops.aten.select.int(convert_element_type_1077, 1, 0)
        select_5: "f32[128]" = torch.ops.aten.select.int(convert_element_type_1077, 1, 1);  convert_element_type_1077 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\hrm\hrm_act_v1.py:281 in forward, code: outputs["target_q_continue"] = torch.sigmoid(torch.where(is_last_step, next_q_halt_logits, torch.maximum(next_q_halt_logits, next_q_continue_logits)))
        maximum: "f32[128]" = torch.ops.aten.maximum.default(select_4, select_5);  select_5 = None
        where_6: "f32[128]" = torch.ops.aten.where.self(ge, select_4, maximum);  ge = select_4 = maximum = None
        sigmoid_48: "f32[128]" = torch.ops.aten.sigmoid.default(where_6);  where_6 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\losses.py:62 in forward, code: mask = labels != IGNORE_LABEL_ID
        ne: "b8[128, 900]" = torch.ops.aten.ne.Scalar(where_4, -100)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\losses.py:63 in forward, code: loss_counts = mask.sum(-1)
        sum_1: "i64[128]" = torch.ops.aten.sum.dim_IntList(ne, [-1])
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\losses.py:64 in forward, code: loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)  # Avoid NaNs in division
        clamp_min: "i64[128]" = torch.ops.aten.clamp_min.default(sum_1, 1)
        unsqueeze_192: "i64[128, 1]" = torch.ops.aten.unsqueeze.default(clamp_min, -1);  clamp_min = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\losses.py:66 in forward, code: is_correct = mask & (torch.argmax(outputs["logits"], dim=-1) == labels)
        argmax: "i64[128, 900]" = torch.ops.aten.argmax.default(slice_314, -1)
        eq: "b8[128, 900]" = torch.ops.aten.eq.Tensor(argmax, where_4);  argmax = None
        bitwise_and_1: "b8[128, 900]" = torch.ops.aten.bitwise_and.Tensor(ne, eq);  eq = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\losses.py:67 in forward, code: seq_is_correct = is_correct.sum(-1) == loss_counts
        sum_2: "i64[128]" = torch.ops.aten.sum.dim_IntList(bitwise_and_1, [-1])
        eq_1: "b8[128]" = torch.ops.aten.eq.Tensor(sum_2, sum_1);  sum_2 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\losses.py:70 in forward, code: valid_metrics = new_carry.halted & (loss_counts > 0)
        gt_1: "b8[128]" = torch.ops.aten.gt.Scalar(sum_1, 0);  sum_1 = None
        bitwise_and_2: "b8[128]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and, gt_1);  gt_1 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\losses.py:72 in forward, code: "count": valid_metrics.sum(),
        sum_3: "i64[]" = torch.ops.aten.sum.default(bitwise_and_2)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\losses.py:74 in forward, code: "accuracy":       torch.where(valid_metrics, (is_correct.to(torch.float32) / loss_divisor).sum(-1), 0).sum(),
        convert_element_type_1078: "f32[128, 900]" = torch.ops.prims.convert_element_type.default(bitwise_and_1, torch.float32);  bitwise_and_1 = None
        div: "f32[128, 900]" = torch.ops.aten.div.Tensor(convert_element_type_1078, unsqueeze_192);  convert_element_type_1078 = None
        sum_4: "f32[128]" = torch.ops.aten.sum.dim_IntList(div, [-1]);  div = None
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_7: "f32[128]" = torch.ops.aten.where.self(bitwise_and_2, sum_4, full_default_1);  sum_4 = full_default_1 = None
        sum_5: "f32[]" = torch.ops.aten.sum.default(where_7);  where_7 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\losses.py:75 in forward, code: "exact_accuracy": (valid_metrics & seq_is_correct).sum(),
        bitwise_and_3: "b8[128]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and_2, eq_1)
        sum_6: "i64[]" = torch.ops.aten.sum.default(bitwise_and_3);  bitwise_and_3 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\losses.py:77 in forward, code: "q_halt_accuracy": (valid_metrics & ((outputs["q_halt_logits"] >= 0) == seq_is_correct)).sum(),
        ge_2: "b8[128]" = torch.ops.aten.ge.Scalar(select_1, 0)
        eq_2: "b8[128]" = torch.ops.aten.eq.Tensor(ge_2, eq_1);  ge_2 = None
        bitwise_and_4: "b8[128]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and_2, eq_2);  eq_2 = None
        sum_7: "i64[]" = torch.ops.aten.sum.default(bitwise_and_4);  bitwise_and_4 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\losses.py:78 in forward, code: "steps":          torch.where(valid_metrics, new_carry.steps, 0).sum(),
        full_default_2: "i32[]" = torch.ops.aten.full.default([], 0, dtype = torch.int32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_8: "i32[128]" = torch.ops.aten.where.self(bitwise_and_2, add_154, full_default_2);  bitwise_and_2 = None
        sum_8: "i64[]" = torch.ops.aten.sum.default(where_8);  where_8 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\losses.py:25 in stablemax_cross_entropy, code: logprobs = log_stablemax(logits.to(torch.float64), dim=-1)
        convert_element_type_1079: "f64[128, 900, 10]" = torch.ops.prims.convert_element_type.default(slice_314, torch.float64)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\losses.py:13 in s, code: x<0,
        lt_1: "b8[128, 900, 10]" = torch.ops.aten.lt.Scalar(convert_element_type_1079, 0)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\losses.py:14 in s, code: 1/(1-x+ epsilon),
        sub: "f64[128, 900, 10]" = torch.ops.aten.sub.Tensor(1, convert_element_type_1079)
        add_309: "f64[128, 900, 10]" = torch.ops.aten.add.Tensor(sub, 1e-30);  sub = None
        reciprocal: "f64[128, 900, 10]" = torch.ops.aten.reciprocal.default(add_309);  add_309 = None
        mul_387: "f64[128, 900, 10]" = torch.ops.aten.mul.Tensor(reciprocal, 1);  reciprocal = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\losses.py:15 in s, code: x + 1
        add_310: "f64[128, 900, 10]" = torch.ops.aten.add.Tensor(convert_element_type_1079, 1)
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\losses.py:12 in s, code: return torch.where(
        where_9: "f64[128, 900, 10]" = torch.ops.aten.where.self(lt_1, mul_387, add_310);  mul_387 = add_310 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\losses.py:21 in log_stablemax, code: return torch.log(s_x/torch.sum(s_x, dim=dim, keepdim=True))
        sum_9: "f64[128, 900, 1]" = torch.ops.aten.sum.dim_IntList(where_9, [-1], True)
        div_1: "f64[128, 900, 10]" = torch.ops.aten.div.Tensor(where_9, sum_9);  where_9 = None
        log: "f64[128, 900, 10]" = torch.ops.aten.log.default(div_1);  div_1 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\losses.py:28 in stablemax_cross_entropy, code: transformed_labels = torch.where(valid_mask, labels, 0)
        where_10: "i32[128, 900]" = torch.ops.aten.where.self(ne, where_4, full_default_2);  full_default_2 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\losses.py:29 in stablemax_cross_entropy, code: prediction_logprobs = torch.gather(logprobs, index=transformed_labels.to(torch.long).unsqueeze(-1), dim=-1).squeeze(-1)
        convert_element_type_1080: "i64[128, 900]" = torch.ops.prims.convert_element_type.default(where_10, torch.int64);  where_10 = None
        unsqueeze_193: "i64[128, 900, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_1080, -1);  convert_element_type_1080 = None
        gather: "f64[128, 900, 1]" = torch.ops.aten.gather.default(log, -1, unsqueeze_193);  log = None
        squeeze: "f64[128, 900]" = torch.ops.aten.squeeze.dim(gather, -1);  gather = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\losses.py:31 in stablemax_cross_entropy, code: return -torch.where(valid_mask, prediction_logprobs, 0)
        full_default_4: "f64[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_11: "f64[128, 900]" = torch.ops.aten.where.self(ne, squeeze, full_default_4);  squeeze = full_default_4 = None
        neg_96: "f64[128, 900]" = torch.ops.aten.neg.default(where_11);  where_11 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\losses.py:83 in forward, code: lm_loss = (self.loss_fn(outputs["logits"], labels, ignore_index=IGNORE_LABEL_ID) / loss_divisor).sum()
        div_2: "f64[128, 900]" = torch.ops.aten.div.Tensor(neg_96, unsqueeze_192);  neg_96 = None
        sum_10: "f64[]" = torch.ops.aten.sum.default(div_2);  div_2 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\losses.py:84 in forward, code: q_halt_loss = F.binary_cross_entropy_with_logits(outputs["q_halt_logits"], seq_is_correct.to(outputs["q_halt_logits"].dtype), reduction="sum")
        convert_element_type_1081: "f32[128]" = torch.ops.prims.convert_element_type.default(eq_1, torch.float32)
        sub_1: "f32[128]" = torch.ops.aten.sub.Tensor(1, convert_element_type_1081);  convert_element_type_1081 = None
        mul_388: "f32[128]" = torch.ops.aten.mul.Tensor(sub_1, select_1);  sub_1 = None
        full_default_5: "f32[]" = torch.ops.aten.full.default([], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        minimum: "f32[128]" = torch.ops.aten.minimum.default(full_default_5, select_1)
        abs_1: "f32[128]" = torch.ops.aten.abs.default(select_1)
        neg_97: "f32[128]" = torch.ops.aten.neg.default(abs_1);  abs_1 = None
        exp: "f32[128]" = torch.ops.aten.exp.default(neg_97);  neg_97 = None
        log1p: "f32[128]" = torch.ops.aten.log1p.default(exp);  exp = None
        sub_2: "f32[128]" = torch.ops.aten.sub.Tensor(minimum, log1p);  minimum = log1p = None
        sub_3: "f32[128]" = torch.ops.aten.sub.Tensor(mul_388, sub_2);  mul_388 = sub_2 = None
        sum_11: "f32[]" = torch.ops.aten.sum.default(sub_3);  sub_3 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\losses.py:94 in forward, code: q_continue_loss = F.binary_cross_entropy_with_logits(outputs["q_continue_logits"], outputs["target_q_continue"], reduction="sum")
        sub_4: "f32[128]" = torch.ops.aten.sub.Tensor(1, sigmoid_48)
        mul_389: "f32[128]" = torch.ops.aten.mul.Tensor(sub_4, select_2);  sub_4 = None
        minimum_1: "f32[128]" = torch.ops.aten.minimum.default(full_default_5, select_2);  full_default_5 = None
        abs_2: "f32[128]" = torch.ops.aten.abs.default(select_2)
        neg_98: "f32[128]" = torch.ops.aten.neg.default(abs_2);  abs_2 = None
        exp_1: "f32[128]" = torch.ops.aten.exp.default(neg_98);  neg_98 = None
        log1p_1: "f32[128]" = torch.ops.aten.log1p.default(exp_1);  exp_1 = None
        sub_5: "f32[128]" = torch.ops.aten.sub.Tensor(minimum_1, log1p_1);  minimum_1 = log1p_1 = None
        sub_6: "f32[128]" = torch.ops.aten.sub.Tensor(mul_389, sub_5);  mul_389 = sub_5 = None
        sum_12: "f32[]" = torch.ops.aten.sum.default(sub_6);  sub_6 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\losses.py:101 in forward, code: return new_carry, lm_loss + 0.5 * (q_halt_loss + q_continue_loss), metrics, detached_outputs, new_carry.halted.all()
        add_311: "f32[]" = torch.ops.aten.add.Tensor(sum_11, sum_12)
        mul_390: "f32[]" = torch.ops.aten.mul.Tensor(add_311, 0.5);  add_311 = None
        add_312: "f64[]" = torch.ops.aten.add.Tensor(sum_10, mul_390);  mul_390 = None
        logical_not: "b8[128]" = torch.ops.aten.logical_not.default(bitwise_and)
        any_1: "b8[]" = torch.ops.aten.any.dims(logical_not);  logical_not = None
        logical_not_1: "b8[]" = torch.ops.aten.logical_not.default(any_1);  any_1 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\losses.py:94 in forward, code: q_continue_loss = F.binary_cross_entropy_with_logits(outputs["q_continue_logits"], outputs["target_q_continue"], reduction="sum")
        sigmoid_49: "f32[128]" = torch.ops.aten.sigmoid.default(select_2)
        sub_7: "f32[128]" = torch.ops.aten.sub.Tensor(sigmoid_49, sigmoid_48);  sigmoid_49 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        permute_196: "bf16[2, 128]" = torch.ops.aten.permute.default(permute_97, [1, 0]);  permute_97 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        permute_206: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_47, [1, 0]);  permute_47 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        permute_211: "bf16[1024, 128]" = torch.ops.aten.permute.default(permute_46, [1, 0]);  permute_46 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        permute_215: "bf16[128, 128]" = torch.ops.aten.permute.default(permute_45, [1, 0]);  permute_45 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        permute_222: "bf16[384, 128]" = torch.ops.aten.permute.default(permute_44, [1, 0]);  permute_44 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        permute_226: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_43, [1, 0]);  permute_43 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        permute_231: "bf16[1024, 128]" = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        permute_235: "bf16[128, 128]" = torch.ops.aten.permute.default(permute_41, [1, 0]);  permute_41 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        permute_242: "bf16[384, 128]" = torch.ops.aten.permute.default(permute_40, [1, 0]);  permute_40 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        permute_246: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_39, [1, 0]);  permute_39 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        permute_251: "bf16[1024, 128]" = torch.ops.aten.permute.default(permute_38, [1, 0]);  permute_38 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        permute_255: "bf16[128, 128]" = torch.ops.aten.permute.default(permute_37, [1, 0]);  permute_37 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        permute_262: "bf16[384, 128]" = torch.ops.aten.permute.default(permute_36, [1, 0]);  permute_36 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        permute_266: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_35, [1, 0]);  permute_35 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        permute_271: "bf16[1024, 128]" = torch.ops.aten.permute.default(permute_34, [1, 0]);  permute_34 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        permute_275: "bf16[128, 128]" = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        permute_282: "bf16[384, 128]" = torch.ops.aten.permute.default(permute_32, [1, 0]);  permute_32 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        permute_286: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_15, [1, 0]);  permute_15 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        permute_291: "bf16[1024, 128]" = torch.ops.aten.permute.default(permute_14, [1, 0]);  permute_14 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        permute_295: "bf16[128, 128]" = torch.ops.aten.permute.default(permute_13, [1, 0]);  permute_13 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        permute_302: "bf16[384, 128]" = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        permute_306: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        permute_311: "bf16[1024, 128]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        permute_315: "bf16[128, 128]" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        permute_322: "bf16[384, 128]" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        permute_326: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_7, [1, 0]);  permute_7 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        permute_331: "bf16[1024, 128]" = torch.ops.aten.permute.default(permute_6, [1, 0]);  permute_6 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        permute_335: "bf16[128, 128]" = torch.ops.aten.permute.default(permute_5, [1, 0]);  permute_5 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        permute_342: "bf16[384, 128]" = torch.ops.aten.permute.default(permute_4, [1, 0]);  permute_4 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        permute_346: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        permute_351: "bf16[1024, 128]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        permute_355: "bf16[128, 128]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
        
         # File: C:\Users\arroy\Projects\hrm_v1\models\layers.py:59 in forward, code: return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
        permute_362: "bf16[384, 128]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        
        # No stacktrace found for following nodes
        copy_: "f32[128, 128]" = torch.ops.aten.copy_.default(primals_16, index_1);  primals_16 = index_1 = copy_ = None
        copy__1: "i32[128]" = torch.ops.aten.copy_.default(primals_18, where_5);  primals_18 = copy__1 = None
        return (add_312, sum_3, sum_5, sum_6, sum_7, sum_8, sum_10, sum_11, sum_12, logical_not_1, slice_314, select_1, select_2, sigmoid_48, convert_element_type_441, convert_element_type_529, where_3, where_4, where_5, bitwise_and, add_154, convert_element_type_538, primals_13, primals_14, where_3, view_166, slice_217, convert_element_type_359, convert_element_type_360, getitem_96, getitem_97, getitem_99, add_107, rsqrt_32, view_172, getitem_100, getitem_101, view_174, add_109, rsqrt_33, view_176, slice_230, convert_element_type_381, convert_element_type_382, getitem_102, getitem_103, getitem_105, add_113, rsqrt_34, view_182, getitem_106, getitem_107, view_184, add_115, rsqrt_35, view_186, slice_243, convert_element_type_403, convert_element_type_404, getitem_108, getitem_109, getitem_111, add_119, rsqrt_36, view_192, getitem_112, getitem_113, view_194, add_121, rsqrt_37, view_196, slice_256, convert_element_type_425, convert_element_type_426, getitem_114, getitem_115, getitem_117, add_125, rsqrt_38, view_202, getitem_118, getitem_119, view_204, add_127, rsqrt_39, view_206, slice_269, convert_element_type_447, convert_element_type_448, getitem_120, getitem_121, getitem_123, add_132, rsqrt_40, view_212, getitem_124, getitem_125, view_214, add_134, rsqrt_41, view_216, slice_282, convert_element_type_469, convert_element_type_470, getitem_126, getitem_127, getitem_129, add_138, rsqrt_42, view_222, getitem_130, getitem_131, view_224, add_140, rsqrt_43, view_226, slice_295, convert_element_type_491, convert_element_type_492, getitem_132, getitem_133, getitem_135, add_144, rsqrt_44, view_232, getitem_136, getitem_137, view_234, add_146, rsqrt_45, view_236, slice_308, convert_element_type_513, convert_element_type_514, getitem_138, getitem_139, getitem_141, add_150, rsqrt_46, view_242, getitem_142, getitem_143, view_244, add_152, rsqrt_47, permute_96, view_246, select, select_1, ne, unsqueeze_192, eq_1, convert_element_type_1079, lt_1, sum_9, unsqueeze_193, sub_7, permute_196, permute_206, permute_211, permute_215, permute_222, permute_226, permute_231, permute_235, permute_242, permute_246, permute_251, permute_255, permute_262, permute_266, permute_271, permute_275, permute_282, permute_286, permute_291, permute_295, permute_302, permute_306, permute_311, permute_315, permute_322, permute_326, permute_331, permute_335, permute_342, permute_346, permute_351, permute_355, permute_362)
        