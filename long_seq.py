import torch
import torch.nn.functional as F
import numpy as np


def process_long_input(model, input_ids, attention_mask, start_tokens, end_tokens):
    # Split the input to 2 overlapping chunks. Now BERT can encode inputs of which the length are up to 1024.
    n, c = input_ids.size()
    start_tokens = torch.tensor(start_tokens).to(input_ids)
    end_tokens = torch.tensor(end_tokens).to(input_ids)
    len_start = start_tokens.size(0)
    len_end = end_tokens.size(0)
    if c <= 512:
        # if document can fit into the encoder
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            output_hidden_states=True,
        )
        sequence_outputs = torch.stack(output[-2][-3:], dim=1)
        sequence_output = sequence_outputs.mean(dim=1)
        attentions = torch.stack(output[-1][-3:],dim=1)
        attention = attentions.mean(dim=1)
        
    else:
        new_input_ids, new_attention_mask, num_seg = [], [], []
        seq_len = attention_mask.sum(1).cpu().numpy().astype(np.int32).tolist()

        # print(attention_mask.shape)
        # print(seq_len)
        # exit(1)
        for i, l_i in enumerate(seq_len): # for each batch
            if l_i <= 512:
                new_input_ids.append(input_ids[i, :512])
                new_attention_mask.append(attention_mask[i, :512])
                num_seg.append(1)
            # elif l_i <= 1024: # split the input into two parts: (0, 512) and (end - 512, end)
            #     input_ids1 = torch.cat([input_ids[i, :512 - len_end], end_tokens], dim=-1)
            #     input_ids2 = torch.cat([start_tokens, input_ids[i, (l_i - 512 + len_start): l_i]], dim=-1)
            #     attention_mask1 = attention_mask[i, :512]
            #     attention_mask2 = attention_mask[i, (l_i - 512): l_i]
            #     new_input_ids.extend([input_ids1, input_ids2])
            #     new_attention_mask.extend([attention_mask1, attention_mask2])
            #     num_seg.append(2)
                
            # elif l_i <= 1536:
            # # else:
            #     input_ids1 = input_ids[i, :512]
            #     input_ids2 = input_ids[i, 512:(512*2)]
            #     input_ids3 = torch.cat([start_tokens, input_ids[i, (l_i - 512 + len_start): l_i]], dim=-1)
            #     attention_mask1 = attention_mask[i, :512]
            #     attention_mask2 = attention_mask[i, 512:(512*2)]
            #     attention_mask3 = attention_mask[i, (l_i - 512): l_i]
            #     new_input_ids.extend([input_ids1, input_ids2, input_ids3])
            #     new_attention_mask.extend([attention_mask1, attention_mask2, attention_mask3])
            #     num_seg.append(3)

            else:
                # Calculate number of chunks needed
                # First chunk: 512 tokens, middle chunks: 512 tokens each, last chunk: overlaps with previous
                content_per_chunk = 512 - len_start - len_end  # Available content space per middle chunk
                remaining_after_first = l_i - (512 - len_end)  # Content after first chunk

                if remaining_after_first <= 512 - len_start:
                    # Only need 2 chunks total
                    chunk1 = torch.cat([input_ids[i, :512 - len_end], end_tokens], dim=-1)
                    chunk2 = torch.cat([start_tokens, input_ids[i, (l_i - 512 + len_start): l_i]], dim=-1)
                    new_input_ids.extend([chunk1, chunk2])
                    new_attention_mask.extend([
                        attention_mask[i, :512],
                        attention_mask[i, (l_i - 512): l_i]
                    ])
                    num_seg.append(2)
                else:
                    # Need multiple chunks
                    chunks_needed = 2  # First + last chunk
                    remaining = remaining_after_first - (512 - len_start)  # After accounting for overlapping last chunk
                    chunks_needed += (remaining + content_per_chunk - 1) // content_per_chunk  # Middle chunks

                    # First chunk (no start tokens)
                    chunk1 = torch.cat([input_ids[i, :512 - len_end], end_tokens], dim=-1)
                    new_input_ids.append(chunk1)
                    new_attention_mask.append(attention_mask[i, :512])

                    current_pos = 512 - len_end
                    for k in range(1, chunks_needed - 1):
                        chunk_end = min(current_pos + content_per_chunk, l_i - (512 - len_start))
                        chunk = torch.cat([
                            start_tokens,
                            input_ids[i, current_pos:current_pos + content_per_chunk],
                            end_tokens
                        ], dim=-1)
                        mask = F.pad(
                            attention_mask[i, current_pos:current_pos + content_per_chunk],
                            (len_start, len_end)
                        )

                        new_input_ids.append(chunk)
                        new_attention_mask.append(mask)
                        current_pos += content_per_chunk

                    last_chunk = torch.cat([start_tokens, input_ids[i, (l_i - 512 + len_start): l_i]], dim=-1)
                    new_input_ids.append(last_chunk)
                    new_attention_mask.append(attention_mask[i, (l_i - 512): l_i])

                    num_seg.append(chunks_needed)
                
        input_ids = torch.stack(new_input_ids, dim=0)
        attention_mask = torch.stack(new_attention_mask, dim=0)

        # print(input_ids.shape) # torch.Size([4, 512])
        # print(attention_mask.shape) # torch.Size([4, 512])
        
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            output_hidden_states=True,
        )
            
        sequence_outputs = torch.stack(output[-2][-3:], dim=1)
        sequence_output = sequence_outputs.mean(dim=1)
        attentions = torch.stack(output[-1][-3:],dim=1)
        attention = attentions.mean(dim=1)

        i = 0
        new_output, new_attention = [], []
        for (n_s, l_i) in zip(num_seg, seq_len):

            # Generalized reconstruction with proper overlap handling
            if n_s == 1:
                output = F.pad(sequence_output[i], (0, 0, 0, c - 512))
                att = F.pad(attention[i], (0, c - 512, 0, c - 512))
                new_output.append(output)
                new_attention.append(att)
            else:
                # Initialize output tensors
                final_output = torch.zeros(c, sequence_output.shape[-1], device=sequence_output.device)
                final_attention = torch.zeros(attention.shape[1], c, c, device=attention.device)
                coverage_count = torch.zeros(c, device=sequence_output.device)

                # Process each chunk
                for chunk_idx in range(n_s):
                    if chunk_idx == 0:
                        # First chunk: use everything except end tokens
                        valid_tokens = 512 - len_end
                        final_output[:valid_tokens] += sequence_output[i + chunk_idx][:valid_tokens]
                        final_attention[:, :valid_tokens, :valid_tokens] += attention[i + chunk_idx][:, :valid_tokens, :valid_tokens]
                        coverage_count[:valid_tokens] += 1

                    elif chunk_idx == n_s - 1:
                        # Last chunk: handle overlap
                        start_pos = l_i - 512 + len_start
                        end_pos = l_i
                        chunk_start = len_start

                        # Add with overlap handling
                        final_output[start_pos:end_pos] += sequence_output[i + chunk_idx][chunk_start:chunk_start + (end_pos - start_pos)]
                        final_attention[:, start_pos:end_pos, start_pos:end_pos] += attention[i + chunk_idx][:, chunk_start:chunk_start + (end_pos - start_pos), chunk_start:chunk_start + (end_pos - start_pos)]
                        coverage_count[start_pos:end_pos] += 1

                    else:
                        # Middle chunk: use content between start/end tokens
                        content_start = 512 - len_end + (chunk_idx - 1) * content_per_chunk
                        content_end = content_start + content_per_chunk
                        chunk_content_start = len_start
                        chunk_content_end = len_start + content_per_chunk

                        final_output[content_start:content_end] += sequence_output[i + chunk_idx][chunk_content_start:chunk_content_end]
                        final_attention[:, content_start:content_end, content_start:content_end] += attention[i + chunk_idx][:, chunk_content_start:chunk_content_end, chunk_content_start:chunk_content_end]
                        coverage_count[content_start:content_end] += 1
                        
                coverage_count = coverage_count + 1e-10
                final_output = final_output / coverage_count.unsqueeze(-1)
                final_attention = final_attention / (final_attention.sum(-1, keepdim=True) + 1e-10)

                new_output.append(final_output)
                new_attention.append(final_attention)

            i += n_s
            
        sequence_output = torch.stack(new_output, dim=0)
        attention = torch.stack(new_attention, dim=0)

        # print(sequence_output.shape)
        # print(attention.shape)

    return sequence_output, attention
