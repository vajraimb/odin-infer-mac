/* CLI and chat loop for Qwen3 Odin inference */

package main

import "core:fmt"
import "core:os"
import "core:strconv"
import "core:strings"
import "core:time"

EOS_TOKEN_ID :: 151645

debug_tokens: bool

time_in_ms :: proc() -> i64 {
	return time.to_unix_nanoseconds(time.now()) / 1_000_000
}

read_stdin :: proc(guide: string, buffer: ^[8192]u8) -> string {
	fmt.print(guide)
	offset := 0
	for offset < 8191 {
		n, err := os.read(os.stdin, buffer[offset:offset + 1])
		if err != os.ERROR_NONE || n == 0 do break
		if buffer[offset] == '\n' do break
		offset += 1
	}
	if offset == 0 do return ""
	result := strings.clone(string(buffer[:offset]))
	for len(result) > 0 && result[len(result) - 1] == '\r' {
		result = result[:len(result) - 1]
	}
	return result
}

chat :: proc(
	transformer: ^Transformer,
	tokenizer: ^Tokenizer,
	sampler: ^Sampler,
	think_on: bool,
	multi_turn: bool,
	tps: bool,
	ttft: bool,
	tb: ^Token_Buffer,
) {
	system_buf: [8192]u8
	user_buf: [8192]u8
	system_prompt: string
	user_prompt: string
	prompt_tokens: [dynamic]int
	defer {
		delete(system_prompt)
		delete(user_prompt)
		delete(prompt_tokens)
	}

	user_turn := true
	next: int
	pos := 0
	num_prompt_tokens := 0
	timer: f64 = -1
	timer2: i64 = -1
	t_ttft: i64 = 0
	count := 0

	for {
		if user_turn {
			if pos == 0 {
				system_prompt = read_stdin("Enter system prompt (or Enter to skip): ", &system_buf)
			}
			user_prompt = read_stdin("Q: ", &user_buf)
			if len(user_prompt) == 0 do break

			if ttft && timer2 == -1 {
				timer2 = time_in_ms()
			}

			rendered: string
			if pos == 0 && len(system_prompt) > 0 {
				rendered = fmt.tprintf(
					"<|im_start|>system\n%s<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n",
					system_prompt,
					user_prompt,
				)
			} else {
				rendered = fmt.tprintf(
					"<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n",
					user_prompt,
				)
			}

			if !think_on {
				rendered = fmt.tprintf("%s<think>\n\n</think>\n", rendered)
			}

			encoded, err := encode(tokenizer, rendered)
			if err != nil {
				fmt.eprintf("encode failed: %v\n", err)
				break
			}
			for id in encoded {
				if id == 0 {
					fmt.fprintln(
						os.stderr,
						"Warning: some prompt text could not be tokenized. Ensure vocab.txt and merges.txt are in the current directory.",
					)
					break
				}
			}

			clear(&prompt_tokens)
			append(&prompt_tokens, ..encoded)

			max_ctx := transformer.config.seq_len
			if len(prompt_tokens) >= max_ctx {
				fmt.fprintf(
					os.stderr,
					"Prompt is %d tokens but context limit is %d; truncating prompt.\n",
					len(prompt_tokens),
					max_ctx,
				)
				resize(&prompt_tokens, max_ctx - 1)
			}

			pos = 0
			user_turn = false

			if multi_turn {
				append_tokens(tb, encoded)
				num_prompt_tokens = len(tb.data)
			} else {
				num_prompt_tokens = len(prompt_tokens)
			}

			fmt.fprintf(os.stderr, "Prefilling %d tokens...\n", num_prompt_tokens)
			fmt.print("A: ")
			os.flush(os.stdout)
		}

		token: int
		if pos < num_prompt_tokens {
			token = multi_turn ? tb.data[pos] : prompt_tokens[pos]
		} else {
			token = next
		}

		if pos >= transformer.config.seq_len {
			fmt.fprintln(os.stderr, "\nContext limit reached; stopping generation.")
			fmt.println()
			user_turn = true
			if tps && timer >= 0 {
				elapsed := f64(time_in_ms()) - timer
				if elapsed > 0 {
					fmt.eprintf("tok/s: %f\n", f64(count) / elapsed * 1000.0)
				}
				timer = -1
				count = 0
			}
			if ttft && t_ttft > 0 {
				fmt.eprintf("TTFT: %d ms\n", t_ttft)
				timer2 = -1
				t_ttft = 0
			}
			continue
		}

		logits: []f32
		when ODIN_OS == .Darwin {
			logits = metal_enabled ? forward_gpu(transformer, token, pos) : forward(transformer, token, pos)
		} else {
			logits = forward(transformer, token, pos)
		}
		next = sample(sampler, logits)
		if debug_tokens && pos < num_prompt_tokens + 3 {
			fmt.fprintf(os.stderr, "[dbg pos=%d tok=%d -> next=%d l0=%.3f]\n", pos, token, next, logits[0])
		}
		pos += 1

		if pos >= num_prompt_tokens {
			if multi_turn {
				append_tokens(tb, []int{next})
				num_prompt_tokens = len(tb.data)
			}

			if next == EOS_TOKEN_ID {
				fmt.println()
				user_turn = true

				if tps && timer >= 0 {
					elapsed := f64(time_in_ms()) - timer
					if elapsed > 0 {
						fmt.eprintf("tok/s: %f\n", f64(count) / elapsed * 1000.0)
					}
					timer = -1
					count = 0
				}
				if ttft {
					fmt.eprintf("TTFT: %d ms\n", t_ttft)
					timer2 = -1
					t_ttft = 0
				}
			} else {
				decoded := decode_token_id(tokenizer, next)
				if len(decoded) > 0 {
					fmt.print(decoded)
					os.flush(os.stdout)
				}
				delete(decoded)
				if ttft && t_ttft == 0 {
					t_ttft = time_in_ms() - timer2
				}
				if tps {
					count += 1
					if timer < 0 {
						timer = f64(time_in_ms())
					}
				}
			}
		}
	}
}

dump_gguf :: proc(path: string) {
	g: GGUF_File
	parse_gguf(path, &g)
	defer free_gguf(&g)

	fmt.printf("tensors=%d metadata=%d\n", len(g.tensors), len(g.metadata))
	counts: map[GGML_Type]int
	defer delete(counts)
	for &t in g.tensors {counts[t.kind] += 1}
	fmt.printf("type histogram: %v\n", counts)
	for name in ([]string{"token_embd.weight", "blk.0.ffn_gate.weight", "blk.0.attn_q.weight", "blk.0.ffn_down.weight"}) {
		if t, ok := gguf_get_tensor(&g, name); ok {
			fmt.printf(
				"%-24s kind=%v dims=%v offset=%d bytes=%d\n",
				name,
				t.kind,
				t.dims,
				t.offset,
				len(t.data),
			)
			ne := 1
			for d in t.dims {ne *= int(d)}
			row_n := int(t.dims[0])
			scratch := make([]f32, row_n)
			defer delete(scratch)
			dequant_row(t.kind, t.data, row_n, scratch)
			fmt.printf("  row0[0:8] =")
			for i in 0 ..< 8 {fmt.printf(" %.5f", scratch[i])}
			fmt.printf("\n")
		} else {
			fmt.printf("%-24s MISSING\n", name)
		}
	}
}

error_usage :: proc() {
	fmt.eprintln("Usage:   qwen3 <FP32 GGUF file> [options]")
	fmt.eprintln("Example: ./qwen3 Qwen3-0.6B-FP32.gguf")
	fmt.eprintln("Options:")
	fmt.eprintln("  -t <float>  temperature in [0,inf], default 0.6")
	fmt.eprintln("  -p <float>  p value in top-p sampling in [0,1] default 0.95")
	fmt.eprintln("  -s <int>    random seed, default time(NULL)")
	fmt.eprintln("  -m <int>    multi-turn: 0 = off (default), 1 = on")
	fmt.eprintln("  -k <int>    reasoning: 0 = off (default), 1 = on")
	fmt.eprintln("  -r <int>    TPS: 0 = off (default), 1 = on")
	fmt.eprintln("  -f <int>    TTFT: 0 = off (default), 1 = on")
	fmt.eprintln("  -j <int>    matmul threads (default: CPU cores)")
	fmt.eprintln("  -c <int>    max context length (default 4096; caps KV cache memory)")
	fmt.eprintln("  -g <int>    GPU (Metal): 0 = off (default), 1 = on (Apple Silicon)")
	os.exit(1)
}

main :: proc() {
	checkpoint_path: string
	temperature: f32 = 0.6
	topp: f32 = 0.95
	rng_seed: u64 = 0
	multi_turn := false
	think_on := false
	tps := false
	ttft := false
	num_threads := os.get_processor_core_count()
	max_ctx := DEFAULT_MAX_CONTEXT
	use_metal := false

	args := os.args
	if len(args) < 2 {
		error_usage()
	}
	checkpoint_path = args[1]
	debug_tokens = os.get_env("QDBG", context.temp_allocator) == "1"

	if len(args) >= 3 && args[2] == "--dump" {
		dump_gguf(checkpoint_path)
		return
	}

	i := 2
	for i < len(args) {
		if i + 1 >= len(args) do error_usage()
		if args[i][0] != '-' || len(args[i]) != 2 do error_usage()

		switch args[i][1] {
		case 't':
			if val, ok := strconv.parse_f64(args[i + 1]); ok {
				temperature = f32(val)
			}
		case 'p':
			if val, ok := strconv.parse_f64(args[i + 1]); ok {
				topp = f32(val)
			}
		case 's':
			if val, ok := strconv.parse_u64(args[i + 1]); ok {
				rng_seed = val
			}
		case 'm':
			multi_turn = args[i + 1] == "1"
		case 'k':
			think_on = args[i + 1] == "1"
		case 'r':
			tps = args[i + 1] == "1"
		case 'f':
			ttft = args[i + 1] == "1"
		case 'j':
			if val, ok := strconv.parse_int(args[i + 1]); ok {
				num_threads = val
			}
		case 'c':
			if val, ok := strconv.parse_int(args[i + 1]); ok {
				max_ctx = val
			}
		case 'g':
			use_metal = args[i + 1] == "1"
		case:
			error_usage()
		}
		i += 2
	}

	if rng_seed == 0 {
		rng_seed = u64(time.time_to_unix(time.now()))
	}
	if temperature < 0 do temperature = 0
	if topp < 0 || topp > 1 do topp = 0.9
	matmul_num_threads = max(num_threads, 1)

	transformer: Transformer
	build_transformer(&transformer, checkpoint_path, max_ctx)
	defer free_transformer(&transformer)

	if use_metal {
		when ODIN_OS == .Darwin {
			if !metal_init(&transformer) {
				fmt.eprintln("metal: init failed, falling back to CPU")
			}
		} else {
			fmt.eprintln("metal: only supported on macOS; using CPU")
		}
	}
	when ODIN_OS == .Darwin {
		defer metal_destroy()
	}

	tokenizer: Tokenizer
	build_tokenizer(&tokenizer)
	defer free_tokenizer(&tokenizer)
	if !verify_tokenizer(&tokenizer) {
		fmt.eprintln("Tokenizer self-check failed. Run from the project directory containing vocab.txt and merges.txt.")
		os.exit(1)
	}

	tb: Token_Buffer
	build_token_buffer(&tb)
	defer free_token_buffer(&tb)

	sampler: Sampler
	build_sampler(&sampler, int(transformer.config.vocab_size), temperature, topp, rng_seed)
	defer free_sampler(&sampler)

	fmt.printf(
		"Multi-turn = %s, thinKing = %s, tps(R) = %s, ttFt = %s, threads = %d, Temperature = %.2f, top-P = %.2f\n",
		multi_turn ? "on" : "off",
		think_on ? "on" : "off",
		tps ? "on" : "off",
		ttft ? "on" : "off",
		num_threads,
		temperature,
		topp,
	)
	fmt.println("Press Enter to exit the chat")

	chat(&transformer, &tokenizer, &sampler, think_on, multi_turn, tps, ttft, &tb)

	destroy_matmul_pool()
}
