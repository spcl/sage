import math
from os import pread
from random import randrange, seed
import argparse


class Args:
    v = None


TAB = " " * 4
header_name = "cuda_src.h"

INSTRUCTION_BYTES = 16

# It should be in range [1;32] (not [0;31])
STARTING_MOD_SHIFT = 7

FIRST_STATE_REG = 9
MAX_REGS = 30


def shift_to_byte(shift):
    assert 1 <= shift and shift <= 32
    return (32 - shift) << 3


def byte_to_shift(byte):
    assert 0 <= byte and byte < (2 ** 8)
    assert byte & 0b111 == 0
    return 32 - (byte >> 3)


def shift_to_byte_cpp(shift):
    return f"((32 - ({shift})) << 3)"


def byte_to_shift_cpp(byte):
    return f"(32 - (({byte}) >> 3))"


# mapping of SASS registers to the variables
var_to_reg = {
    "i": 2,
    "stack_ptr": 1,
    "c": 0,
    "base_lo": 3,
    "output": 4,  # use in epilogue
    "mlo": 4,
    "base_hi": 5,  # base_hi have to be right after mlo as 64-bit register pair
    "shift_byte": 6,
    "mval": 7,
    "j": 8,
    **{f"s{si}": ri for si, ri in enumerate(range(FIRST_STATE_REG, MAX_REGS))},
}

# ld requires to wait both read and write dependencies
LD_WAIT_REG_WRITE = 3
LD_WAIT_REG_READ = 4
LD_BARRIER_WRITE = "..3..."
LD_BARRIER_READ = ".4...."

# st has only read register dependency
ST_WAIT_REG_READ = 5
ST_BARRIER_READ = "5....."


def as_reg(var_name):
    if isinstance(var_name, int) or var_name is None:
        return var_name  # leave numeric constants unchanged
    elif var_name.startswith("PLACEHOLDER"):
        return var_name
    elif var_name == "p":
        return "P5"  # the number is randomly chosen
    elif var_name == "q":
        return "P4"
    elif var_name in var_to_reg:
        return "R" + str(var_to_reg[var_name])
    else:
        # leave everything else unchanged
        raise Exception(f"Unknown var name {var_name}")


class InstructionCounter:
    def __init__(self):
        self.counter = 0

    def add(self, s, kind):
        self.counter += 1
        if kind == "sass":
            return s
        else:
            return ""


class InstructionWriter:
    def __init__(
        self, state_size, num_iters, num_shifts, first_alu, kind, instruction_counter
    ):
        self.alu = first_alu
        self.kind = kind
        assert kind in ("ptx", "cpp", "sass")
        self.written = 0
        self.instruction_counter = instruction_counter

        # mapping that stores how many times we shifted each free register
        self.left_shifted_count = {r: 0 for r in range(MAX_REGS - FIRST_STATE_REG)}
        self.right_shifted_count = {r: 0 for r in range(MAX_REGS - FIRST_STATE_REG)}
        # keep track of recently used registers to avoid dependencies
        self.recently_used_regs = [-1 for _ in range(5)]

    def shift_once(self):
        shift = randrange(3, 30)
        # use ALU for right shifts, FMA for left shifts
        shift_dict = self.right_shifted_count if self.alu else self.left_shifted_count
        shift_direction = ">>" if self.alu else "<<"
        instr_name = "shradd" if self.alu else "shladd"
        ordered_regs = sorted(shift_dict, key=shift_dict.get)
        for reg in ordered_regs:
            if reg in self.recently_used_regs:
                continue
            else:
                self.recently_used_regs = self.recently_used_regs[1:] # drop oldest element
                self.recently_used_regs.append(reg) # add newest element

                shift_dict[reg] += 1
                
                reg_name = f"s{reg}"
                return self.get_instr(
                    f"{reg_name} += {reg_name} {shift_direction} {shift}",
                    instr_name,
                    reg_name,
                    [reg_name, shift],
                )
        raise Exception("Unreachable")

    def shift_sequence(self, n):
        code = ""
        for i in range(n):
            code += self.shift_once()
        return code

    def shift_until_alu(self):
        if not self.alu:
            return self.shift_once()
        else:
            return ""

    def shift_until_fma(self):
        if self.alu:
            return self.shift_once()
        else:
            return ""

    def get_instr(
        self, expl, instr, dst, srcs, ctrl=f"B......|R.|W.|Y1|S1|"
    ):
        self.instruction_counter.counter += 1

        # LOP3 LUT
        tA = 0b11110000
        tB = 0b11001100
        tC = 0b10101010

        # replace variable names by SASS registers
        if self.kind == "sass":
            dst = as_reg(dst)
            srcs = [as_reg(s) for s in srcs]

        if instr == "inc":
            assert len(srcs) == 1
            assert self.alu
            if self.kind == "ptx":
                instr_text = f'asm volatile("add.u32 {dst}, {srcs[0]}, 1;");'
            elif self.kind == "cpp":
                instr_text = f"{dst} = {srcs[0]} + 1;"
            elif self.kind == "sass":
                instr_text = f"{ctrl} IADD3 {dst}, {srcs[0]}, 0x1, RZ;"
            # This is an alternative non-alu implementation that requires one register that contains value "1"
            # if self.kind == "ptx":
            #     if self.alu:
            #         instr_text = f'asm volatile("add.u32 {dst}, {srcs[0]}, 1;");'
            #     else:
            #         instr_text = f'asm volatile("mad.lo.u32 {dst}, {srcs[0]}, R1, 1;");'
            # elif self.kind == "cpp":
            #     instr_text = f"{dst} = {srcs[0]} + 1;"
            # elif self.kind == "sass":
            #     if self.alu:
            #         instr_text = f"IADD3 {dst}, {srcs[0]}, 0x1, RZ;"
            #     else:
            #         instr_text = f"IMAD {dst}, {srcs[0]}, R1, 0x1;"
        elif instr == "and":
            assert self.alu
            if self.kind == "ptx":
                instr_text = f'asm volatile("and.b32 {dst}, {srcs[0]}, {srcs[1]};");'
            elif self.kind == "cpp":
                instr_text = f"{dst} = {srcs[0]} & {srcs[1]};"
            elif self.kind == "sass":
                instr_text = (
                    f"{ctrl} LOP3.LUT {dst}, {srcs[0]}, {srcs[1]}, RZ, {tA & tB}, !PT;"
                )
        elif instr == "add":
            if self.kind == "ptx":
                if self.alu:
                    instr_text = (
                        f'asm volatile("add.u32 {dst}, {srcs[0]}, {srcs[1]};");'
                    )
                else:
                    instr_text = (
                        f'asm volatile("mad.lo.u32 {dst}, {srcs[0]}, 1, {srcs[1]};");'
                    )
            elif self.kind == "cpp":
                instr_text = f"{dst} = {srcs[0]} + {srcs[1]};"
            elif self.kind == "sass":
                if self.alu:
                    instr_text = f"{ctrl} IADD3 {dst}, {srcs[0]}, {srcs[1]}, RZ;"
                else:
                    instr_text = f"{ctrl} IMAD.U32 {dst}, {srcs[0]}, 1, {srcs[1]};"
        elif instr == "assign":
            if self.kind == "ptx":
                if self.alu:
                    instr_text = f'asm volatile("add.u32 {dst}, {srcs[0]}, 0;");'
                else:
                    instr_text = f'asm volatile("mad.lo.u32 {dst}, 0, 0, {srcs[0]};");'
            elif self.kind == "cpp":
                instr_text = f"{dst} = {srcs[0]};"
            elif self.kind == "sass":
                if self.alu:
                    instr_text = f"{ctrl} IADD3 {dst}, RZ, {srcs[0]}, RZ;"
                else:
                    instr_text = f"{ctrl} IMAD.U32 {dst}, RZ, RZ, {srcs[0]};"
        elif instr == "shr":
            shift = srcs[1]
            multiplier = 2 ** (32 - shift)
            if self.kind == "ptx":
                if self.alu:
                    instr_text = f'asm volatile("shr.b32 {dst}, {srcs[0]}, {shift};"); // {expl}\n'
                else:
                    instr_text = f'asm volatile("mul.hi.u32 {dst}, {srcs[0]}, {multiplier};"); // {expl}\n'
            elif self.kind == "cpp":
                instr_text = f"{dst} = {srcs[0]} >> {shift};\n"
            elif self.kind == "sass":
                if self.alu:
                    instr_text = f"{ctrl} SHF.R.U32.HI {dst}, RZ, {shift}, {srcs[0]};\n"
                else:
                    instr_text = (
                        f"{ctrl} IMAD.HI.U32 {dst}, {srcs[0]}, {multiplier}, RZ;\n"
                    )
        elif instr == "setp":
            assert len(srcs) == 2
            if self.kind == "ptx":
                assert self.alu
                instr_text = (
                    f'asm volatile("setp.eq.u32 {dst}, {srcs[0]}, {srcs[1]};");'
                )
            elif self.kind == "cpp":
                instr_text = f"{dst} = {srcs[0]} == {srcs[1]};"
            elif self.kind == "sass":
                # my guess on operands of ISETP (from reading PTX ISA https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#comparison-and-selection-instructions-setp)
                # ISETP.CMP_OP.BOOL_OP P, Q, A, B, C
                # T = CMP_OP(A, B)
                # P = BOOL_OP(T, C)
                # Q = BOOL_OP(!T, C)
                #
                # PT == always True (like RZ always zero)
                # when Q is unused, PT is placed instead of it
                instr_text = f"{ctrl} ISETP.EQ.AND {dst}, PT, {srcs[0]}, {srcs[1]}, PT;"
        elif instr == "sel":
            assert len(srcs) == 3
            if self.kind == "ptx":
                assert self.alu
                instr_text = (
                    f'asm volatile("selp.b32 {dst}, {srcs[1]}, {srcs[2]}, {srcs[0]};");'
                )
            elif self.kind == "cpp":
                instr_text = f"{dst} = {srcs[0]} ? {srcs[1]} : {srcs[2]};"
            elif self.kind == "sass":
                # this instruciton allows to negate predicate register as part of it
                # SEL D, A, B, [!]P
                # D = [!]P ? A : B
                # here we assume srcs[1] is immediate and srcs[2] is a register
                instr_text = f"{ctrl} SEL {dst}, {srcs[2]}, {srcs[1]}, !{srcs[0]};"
        elif instr == "bra":
            assert len(srcs) == 2
            pred = srcs[0]
            off = srcs[1]
            if self.kind == "ptx":
                instr_text = f'asm volatile("@!{pred} bra.uni {off};");'
            elif self.kind == "cpp":
                instr_text = f"if (!{pred}) goto {off};"
            elif self.kind == "sass":
                instr_text = f"{ctrl} @!{pred} BRA {off};"
        elif instr == "ld":
            assert len(srcs) == 2
            mlo = srcs[0]
            mhi = srcs[1]
            if self.kind == "ptx":
                # probably it doesn't require alu or fma
                instr_text = (
                    'asm volatile("{\\n\\t'
                    ".reg .u64 addr;\\n\\t"
                    f"mov.b64 addr, {{ {mlo}, {mhi} }};\\n\\t"
                    f"ld.u32 {dst}, [addr];\\n\\t"
                    '}");  // {dst} = *({mlo} + ({mhi} << 32))'
                )
            elif self.kind == "cpp":
                instr_text = f"{dst} = *(uint32_t*)({mlo} + (((uint64_t){mhi}) << 32));"
            elif self.kind == "sass":
                mlo_reg_idx = mlo[1:]
                mhi_reg_idx = mhi[1:]
                if int(mlo_reg_idx) % 2 != 0:
                    raise Exception("An index of LO register should be even")
                if int(mhi_reg_idx) != int(mlo_reg_idx) + 1:
                    raise Exception("LO and HI registers should be adjacent")
                instr_text = f"{ctrl} LD.E.SYS {dst}, [{mlo}];"
        elif instr == "st8":
            # instruction to store shift_byte
            assert len(srcs) == 4
            val, mlo, mhi, off = tuple(srcs)
            if self.kind == "ptx":
                # we are not storing anything in the reference implementation
                instr_text = (
                    'asm volatile("{\\n\\t'
                    ".reg .u64 addr;\\n\\t"
                    ".reg .u32 mlo_off;\\n\\t"
                    f"add.u32 mlo_off, {mlo}, {off};\\n\\t"
                    f"mov.b64 addr, {{ mlo_off, {mhi} }};\\n\\t"
                    f"st.u8 [addr], {val};\\n\\t"
                    '}");  //  *(off + mlo + (mhi << 32)) = val'
                )
            elif self.kind == "cpp":
                instr_text = (
                    f"*(uint8_t*)({off} + {mlo} + (((uint64_t){mhi}) << 32)) = {val}; "
                )
            elif self.kind == "sass":
                mlo_reg_idx = mlo[1:]
                mhi_reg_idx = mhi[1:]
                if int(mlo_reg_idx) % 2 != 0:
                    raise Exception("An index of LO register should be even")
                if int(mhi_reg_idx) != int(mlo_reg_idx) + 1:
                    raise Exception("LO and HI registers should be adjacent")
                instr_text = f"{ctrl} ST.E.U8.SYS [{mlo}+{off}], {val};"
        elif instr == "shladd":
            assert len(srcs) == 2
            src = srcs[0]
            shift = srcs[1]
            if self.kind == "ptx":
                if self.alu:
                    instr_text = f'asm volatile("vshl.u32.u32.u32.wrap.add {dst}, {src}, {shift}, {src};");'
                else:
                    instr_text = f'asm volatile("mad.lo.u32 {dst}, {src}, {2 ** shift}, {src};");'
            elif self.kind == "cpp":
                instr_text = f"{dst} += {src} << {shift};"
            elif self.kind == "sass":
                if self.alu:
                    # 64-bits = (hi:lo)
                    # LEA d, a, b, s ===> d = (a << s).lo + b
                    instr_text = f"{ctrl} LEA {dst}, {src}, {src}, {shift};"
                else:
                    # IMAD d, a, b, c ===> d = a * b + c
                    instr_text = f"{ctrl} IMAD.U32 {dst}, {src}, {2 ** shift}, {src};"
        elif instr == "shradd":
            assert len(srcs) == 2
            assert (
                self.alu
            )  # because IMAD.HI requries 64-bit register in the last position!
            src = srcs[0]
            shift = srcs[1]
            if self.kind == "ptx":
                if self.alu:
                    instr_text = f'asm volatile("vshr.u32.u32.u32.wrap.add {dst}, {src}, {shift}, {src};");'
                else:
                    instr_text = f'asm volatile("mad.hi.u32 {dst}, {src}, {2 ** (32 - shift)}, {src};");'
            elif self.kind == "cpp":
                instr_text = f"{dst} += {src} >> {shift};"
            elif self.kind == "sass":
                if self.alu:
                    # 64-bits = (hi:lo)
                    # LEA.HI d, a, b, c, s ===> d = (c:a << s).hi + b
                    # c = RZ, s = 0-31 ===> d = (a >> (32 - s)) + b
                    instr_text = f"{ctrl} LEA.HI {dst}, {src}, {src}, RZ, {32 - shift};"
                else:
                    if src[0] == "R" and int(src[1:]) % 2 != 0:
                        raise Exception("odd register in IMAD.HI")
                    instr_text = (
                        f"{ctrl} IMAD.HI.U32 {dst}, {src}, {2 ** (32 - shift)}, {src};"
                    )
        elif instr == "shraddmod":
            assert len(srcs) == 2
            assert self.alu
            src = srcs[0]
            shift_byte = srcs[1]
            if self.kind == "ptx":
                # use clamp variant because it allows "imm=32-shift" to be in range [0;32] while we targeting [1;32] with shift=[0;31]
                instr_text = f'asm volatile("vshr.u32.u32.u32.clamp.add {dst}, {src}, {STARTING_MOD_SHIFT}, {src};");'
            elif self.kind == "cpp":
                instr_text = (
                    "{"
                    f"uint32_t real_shift;"
                    "if (copied_memory) {"
                    f"real_shift = {STARTING_MOD_SHIFT};"
                    "} else {"
                    f"real_shift = {byte_to_shift_cpp(shift_byte)};"
                    "}"
                    "if (real_shift != 32) {"
                    f"{dst} += {src} >> real_shift;"
                    "}"
                    "}"
                )
            elif self.kind == "sass":
                instr_text = f"{ctrl} LEA.HI {dst}, {src}, {src}, RZ, {32-STARTING_MOD_SHIFT};"
        else:
            raise Exception(f"Unknown instruction {instr}")
        self.alu = not self.alu  # switch to another instruction kind (FMA/ALU)
        instr_text += f" // {expl}\n"
        self.written += 1
        return instr_text


class Generator:
    def __init__(self, num_iters, num_inner_iters, num_shifts, mem_size=None):
        self.state_size = MAX_REGS - FIRST_STATE_REG
        self.num_iters = num_iters
        self.num_inner_iters = num_inner_iters
        self.num_shifts = num_shifts
        self.loop_instr = None
        self.loop_size = None
        self.mem_size = mem_size

    def generate_header(self):
        if (
            (self.loop_instr is None)
            or (self.loop_size is None)
            or (self.mem_size is None)
        ):
            raise Exception(
                "Generate code before generating header to get the number of instructions"
            )

        expected_clocks = (self.loop_instr - self.inner_loop_instr) * self.num_iters + self.inner_loop_instr * self.num_iters * self.num_inner_iters
        code = (
            "// --- DO NOT EDIT! THIS IS GENERATED FILE --- //\n"
            "#pragma once\n"
            "\n"
            "#include <cstdlib>\n"
            "#include <cstdint>\n"
            "#include <cassert>\n"
            "#include <cstdio>\n"
            "\n"
            f"constexpr size_t STATE_SIZE = {self.state_size};\n"
            f"constexpr size_t MEM_SIZE = {self.mem_size};\n"
            f"constexpr size_t EXPECTED_CLOCKS = {expected_clocks};\n"
            "\n"
            "struct State {\n"
            "    uint32_t c;\n"
            "    uint32_t d[STATE_SIZE];\n"
            "    uint32_t mem_lo;\n"
            "    uint32_t mem_hi;\n"
            "};\n"
        )

        return code

    def generate_src(self, kind):
        assert kind in ("ptx", "cpp", "sass")

        seed(12345)

        if kind in ("ptx", "cpp"):
            code = "// --- DO NOT EDIT! THIS IS GENERATED FILE --- //\n"
            code += f'#include "{header_name}"\n\n'
        elif kind == "sass":
            code = ""  # init

        # === kernel launches

        if kind == "ptx":
            code += "#include <cuda.h>\n\n"
            code += "#include <cooperative_groups/details/helpers.h>\n\n"
            code += "__device__ State checksum_ptx(State s);\n\n"
            code += "using checksum_function_ptr = State (*)(State);\n\n"
            code += "__device__ checksum_function_ptr checksum_func = nullptr;\n"

            code += """
                __device__ unsigned sync_bar; // initalize to 0 before use!
                __device__ void grid_sync() {
                    cooperative_groups::details::grid::sync(&sync_bar);
                }
                __device__ int grid_rank() {
                    return cooperative_groups::details::grid::thread_rank();
                }
                __device__ int grid_size() {
                    return cooperative_groups::details::grid::size();
                }

            """

            code += """
                __device__ uint32_t global_reduce_tmp;
                __device__ void global_reduce_init() { 
                    if (grid_rank() == 0) {
                        global_reduce_tmp = 0;
                    }
                    grid_sync();
                }
                __device__ uint32_t global_reduce(uint32_t local) {
                    // warp reduce
                    uint32_t v = local;
                    #pragma unroll
                    for (int i = 1; i < warpSize; i = i * 2) {
                        v += __shfl_xor_sync(0xffffffff, v, i);
                    }
                    // block reduce
                    __shared__ uint32_t shared_val;
                    if (threadIdx.x == 0) { // first thread of each block does initialization
                        shared_val = 0;
                    }
                    __syncthreads();
                    if (threadIdx.x % warpSize == 0) { // first thread of each warp participates
                        atomicAdd_block(&shared_val, v);
                    }
                    __syncthreads(); // block barrier
                    // grid reduce 
                    if (threadIdx.x == 0) { // first thread of each block participates
                        atomicAdd(&global_reduce_tmp, shared_val);
                    }
                    grid_sync(); // grid barrier
                    return global_reduce_tmp;
                }
            """

            code += """
                __device__ void checksum_kernel_caller(
                    State* state, uint32_t* data_ptr,
                    uint64_t* clocks, checksum_function_ptr func
                ) {
                    uint32_t id = grid_rank();

                    if (id == 0) {
                        printf("base address %p\\n", data_ptr);
                    }
                    if ( (((uint64_t)data_ptr) + MEM_SIZE) >> 32 != ((uint64_t)data_ptr) >> 32) {
                        if (id == 0) {
                            printf("Unexpected data pointer alignment!\\n");
                        }
                        return;
                    }
                    if (id == 0) {
                        printf("Data pointer alignment is good!\\n");
                    }

                    uint64_t clock_start = clock64();
                    State cur_state = *state;
                    cur_state.mem_lo = ((uint64_t)data_ptr);
                    cur_state.mem_hi = ((uint64_t)data_ptr) >> 32;
                    for (int i = 0; i < STATE_SIZE; i++) {
                        uint32_t mask = i + id * STATE_SIZE;
                        mask = (mask << 17) | (mask >> 15);
                        cur_state.d[i] ^= mask;
                    }

                    global_reduce_init();
                    State new_state = func(cur_state);

                    uint32_t acc = new_state.c;
                    for (int i = 0; i < STATE_SIZE; i++) {
                        acc += new_state.d[i];
                    }
                    acc = global_reduce(acc);

                    if (id == 0) {
                        state->c = acc;
                        *clocks = clock64() - clock_start;
                    }
                }
            """

            code += """
                extern "C"
                __global__ void init_kernel() {
                    // this prevents checksum function to be optimized away so we will be able to extract it from cubin
                    checksum_func = checksum_ptx;
                }

                extern "C"
                __global__ void checksum_kernel(
                    State* state, uint32_t* data_ptr, bool copy_memory, uint64_t* clocks
                ) {
                    // call init_kernel to initialize checksum_func
                    uint32_t* current_data_ptr = data_ptr + blockIdx.x * MEM_SIZE / sizeof(*data_ptr);
                    checksum_kernel_caller(state, current_data_ptr, clocks, checksum_func);
                }

                extern "C"
                __global__ void checksum_kernel_from_data(
                    State* state, uint32_t* data_ptr, bool copy_memory, uint64_t* clocks
                ) {
                    uint32_t* current_data_ptr = data_ptr + blockIdx.x * MEM_SIZE / sizeof(*data_ptr);
                    uint32_t* func_ptr = current_data_ptr;
                    if (copy_memory) {
                        func_ptr = data_ptr + gridDim.x * MEM_SIZE / sizeof(*data_ptr);
                    }
                    checksum_kernel_caller(state, current_data_ptr, clocks, (checksum_function_ptr) func_ptr);
                }
            """
        elif kind == "cpp":
            code += """
                State checksum_cpp(State s, bool copied_memory);

                extern "C"
                void checksum_kernel_reference(State* state, uint32_t* data_ptr, uint32_t grid_size, uint32_t block_size, bool copied_memory) {
                    uint32_t chk = 0;
                    uint32_t total_threads = grid_size * block_size;

                    printf("Computing checksum on host...   0%%");
                    #pragma omp parallel for
                    for (uint32_t blk = 0; blk < grid_size; blk++) {
                        uint32_t* block_data_ptr = data_ptr + (MEM_SIZE / sizeof(*data_ptr)) * blk;
                        state->mem_lo = ((uint64_t)block_data_ptr);
                        state->mem_hi = ((uint64_t)block_data_ptr) >> 32;

                        for (uint32_t thr = 0; thr < block_size; thr++) {
                            uint32_t id = thr + block_size * blk;
                            State cur_state = *state;

                            for (uint32_t i = 0; i < STATE_SIZE; i++) {
                                uint32_t mask = i + id * STATE_SIZE;
                                mask = (mask << 17) | (mask >> 15);
                                cur_state.d[i] ^= mask;
                            }

                            State new_state = checksum_cpp(cur_state, copied_memory);
                            
                            uint32_t chktmp = 0;
                            chktmp += new_state.c;
                            for (uint32_t i = 0; i < STATE_SIZE; i++) {
                                chktmp += new_state.d[i];
                            }
                            #pragma omp atomic
                            chk += chktmp;
                            size_t step = total_threads / 20;
                            if (step == 0) step = 1;
                            if (id % step == 0) {
                                printf("\\rComputing checksum on host... %3d%% ", (int)(id * 100. / total_threads));
                            }
                        }
                    }
                    printf("\\rComputing checksum on host... 100%%\\n");

                    state->c = chk;
                }
            """
        elif kind == "sass":
            pass

        # === checksum function

        if kind == "cpp":
            code += f"State checksum_cpp(State s, bool copied_memory) {{\n"
        elif kind == "ptx":
            code += f"__device__ State checksum_ptx(State s) {{\n"
        elif kind == "sass":
            pass

        # === variable declarations

        # unique variables
        if kind == "ptx":
            code += TAB + 'asm volatile(".reg .u32 i, j, shift_byte, mval, c;");\n'
            code += TAB + 'asm volatile(".reg .u32 base_lo, mlo, base_hi;");\n'
            code += TAB + 'asm volatile(".reg .pred p, q;");\n'
        elif kind == "cpp":
            code += TAB + "uint32_t i, j, shift_byte, mval, c;\n"
            code += TAB + "uint32_t base_lo, mlo, base_hi;\n"
            code += TAB + "bool p, q;\n"
        elif kind == "sass":
            pass

        # state variables
        if kind == "ptx":
            code += (
                TAB
                + 'asm volatile(".reg .u32 '
                + ", ".join([f"s{v}" for v in range(self.state_size)])
                + ';");\n'
            )
        elif kind == "cpp":
            code += (
                TAB
                + "uint32_t "
                + ", ".join([f"s{v}" for v in range(self.state_size)])
                + ";\n"
            )
        elif kind == "sass":
            pass

        # === assign values to registers

        # unique variables
        if kind == "ptx":
            code += TAB + 'asm volatile("mov.u32 i, 0;");\n'
            code += TAB + 'asm volatile("mov.b32 base_lo, %0;" :: "r"(s.mem_lo));\n'
            code += TAB + 'asm volatile("mov.b32 base_hi, %0;" :: "r"(s.mem_hi));\n'
            code += (
                TAB
                + f'asm volatile("mov.u32 shift_byte, {shift_to_byte(STARTING_MOD_SHIFT)};");\n'
            )
        elif kind == "cpp":
            code += TAB + "i = 0;\n"
            code += TAB + "base_lo = s.mem_lo;\n"
            code += TAB + "base_hi = s.mem_hi;\n"
            code += TAB + f"shift_byte = {shift_to_byte(STARTING_MOD_SHIFT)};\n"
        elif kind == "sass":
            pass

        # state variables
        if kind == "ptx":
            code += TAB + 'asm volatile("mov.u32 c, %0;" :: "r"(s.c));\n'
            for i in range(self.state_size):
                code += TAB + f'asm volatile("mov.u32 s{i}, %0;" :: "r"(s.d[{i}]));\n'
        elif kind == "cpp":
            code += TAB + "c = s.c;\n"
            for i in range(self.state_size):
                code += TAB + f"s{i} = s.d[{i}];\n"
        elif kind == "sass":
            pass

        stored_regs = [2, 4] + list(range(16, MAX_REGS))
        num_stored_regs = len(stored_regs)

        ic = InstructionCounter()

        # first we put registers that we are not allowed to modify (according to calling convention) on the stack
        # R1 stores a stack pointer itself
        # R2 (I don't know what it stores), R4 (stores return value address) and R16-R31 should be saved
        code += ic.add(
            f"B......|R.|W.|Y1|S8| IADD3 R1, R1, -{num_stored_regs * 4}, RZ; // aquire memory on stack\n",
            kind,
        )
        for idx, reg in enumerate(stored_regs):
            code += ic.add(f"B......|R0|W.|Y1|S4| STL [R1+{idx * 4}], R{reg};\n", kind)
        code += ic.add(f"B.....0|R.|W.|Y1|S8| NOP;\n", kind)
        # initialize counter i
        code += ic.add(f"B......|R.|W.|Y1|S8| IADD3 {as_reg('i')}, RZ, 0, RZ;\n", kind)
        # move input arguments into registers
        # checksum
        code += ic.add(
            f"B......|R1|W2|Y1|S4| LDL {as_reg('c')}, [R1 + {num_stored_regs * 4}];\n",
            kind,
        )
        # state
        for i in range(self.state_size):
            rn = as_reg(f"s{i}")
            code += ic.add(
                f"B......|R1|W2|Y1|S4| LDL {rn}, [R1 + {(num_stored_regs + i + 1) * 4}];\n",
                kind,
            )
        mem_addr_offset = (num_stored_regs + self.state_size + 1) * 4
        # memory address
        code += ic.add(
            f"B......|R1|W2|Y1|S4| LDL {as_reg('base_lo')}, [R1 + {mem_addr_offset}];\n",
            kind,
        )
        code += ic.add(
            f"B......|R1|W2|Y1|S4| LDL {as_reg('base_hi')}, [R1 + {mem_addr_offset + 4}];\n",
            kind,
        )
        code += ic.add(f"B...21.|R.|W.|Y1|S8| NOP;\n", kind)

        # === checksum blocks

        # generate each block
        writer = InstructionWriter(
            self.state_size, self.num_iters, self.num_shifts, True, kind, ic
        )

        if kind == "ptx":
            code += TAB + f'asm volatile("OUTER_LOOP_START:");\n'
        elif kind == "cpp":
            code += TAB + f"OUTER_LOOP_START:\n"
        elif kind == "sass":
            code += ""

        prefix_instructions = ic.counter

        loop_body = ""

        loop_body += writer.shift_until_alu()
        loop_body += writer.get_instr(f"i++", "inc", "i", ["i"])

        # +++ compute store address
        loop_body += writer.shift_until_alu()
        loop_body += writer.get_instr(
            "(st) mem_offset_lo = mem_base",
            "add",
            "mlo",
            ["base_lo", 0],
        )
        # === compute store address

        # +++ compute modified shift amount for store
        loop_body += writer.shift_until_alu()
        loop_body += writer.get_instr(
            "shift_byte = c & instr_mask", "and", "shift_byte", ["c", 0b11111000]
        )
        # === compute modified shift amount for store

        # consume some instructions to make store address ready
        loop_body += writer.shift_sequence(6)

        # +++ perform store, it is expected to be visible on the next iteration
        if Args.v.with_self_modification:
            loop_body += writer.get_instr(
                "store shift to mem",
                "st8",
                None,
                ["shift_byte", "mlo", "base_hi", "PLACEHOLDER_MOD_LOCATION"],
                ctrl=f"B......|R{ST_WAIT_REG_READ}|W.|Y1|S1|",
            )
        # === perform store

        # +++ check if this is the last iteration
        loop_body += writer.shift_until_alu()
        loop_body += writer.get_instr(
            f"p = (i == num_iters [{self.num_iters}])",
            "setp",
            "p",
            ["i", self.num_iters],
        )
        # === check if this is the last iteration

        # give store some time to consume address register
        loop_body += writer.shift_sequence(20)

        # give store time to write value into memory
        # +++ large chunk of shifts
        loop_body += writer.shift_until_fma()
        for _ in range(self.num_shifts):
            shift = randrange(3, 30)
            loop_body += writer.get_instr(
                f"c += c << {shift}", "shladd", "c", ["c", shift]
            )

            even_shifts = self.state_size + (self.state_size % 2)
            loop_body += writer.shift_sequence(even_shifts)

            shift = randrange(3, 30)
            loop_body += writer.get_instr(
                f"c += c >> {shift}", "shradd", "c", ["c", shift]
            )

            loop_body += writer.shift_sequence(even_shifts)
        # === large chunk of shifts

        # +++ compute load address offset
        loop_body += writer.shift_until_alu()
        loop_body += writer.get_instr(
            f"load_offset = 0b0..01..100 & checksum",
            "and",
            "mlo",
            ["c", "PLACEHOLDER_LOAD_MASK"],
            # wait address register to became available
            ctrl=f"B{ST_BARRIER_READ}|R.|W.|Y1|S1|",
        )
        # === compute load address offset

        # give some time load address offset to be computed
        loop_body += writer.shift_sequence(6)

        # +++ add load offset to the memory base address
        loop_body += writer.shift_until_alu()
        loop_body += writer.get_instr(
            "(ld) mem_offset_lo = mem_base + mem_lo",
            "add",
            "mlo",
            ["mlo", "base_lo"],
        )
        # === add load offset to the memory base address

        # give some time absolute load address to be computed
        loop_body += writer.shift_sequence(6)

        # +++ initiate memory load
        loop_body += writer.get_instr(
            "load from mem",
            "ld",
            "mval",
            ["mlo", "base_hi"],
            ctrl=f"B......|R{LD_WAIT_REG_READ}|W{LD_WAIT_REG_WRITE}|Y1|S1|",
        )
        # === initiate memory load

        # +++ a small loop to hide instruction cache misses +++
        if Args.v.with_inner_loop:
            loop_body += writer.get_instr(f"j = 0", "assign", "j", [0])
            loop_body += writer.shift_sequence(6)  # make sure that j is ready

            loop_body += writer.shift_until_alu()

            if kind == "ptx":
                loop_body += TAB + f'asm volatile("INNER_LOOP_START:");\n'
            elif kind == "cpp":
                loop_body += TAB + f"INNER_LOOP_START:\n"
            elif kind == "sass":
                loop_body += ""

            inner_loop_start_instr = ic.counter

            loop_body += writer.get_instr(f"j++", "inc", "j", ["j"])

            loop_body += writer.shift_sequence(200)  # computations of inner loop

            loop_body += writer.shift_until_alu()
            loop_body += writer.get_instr(
                f"q = (j == num_inner_iters [{self.num_inner_iters}])",
                "setp",
                "q",
                ["j", self.num_inner_iters],
            )

            loop_body += writer.shift_sequence(12)  # make sure that q is ready

            if kind == "sass":
                inner_addr_offset = "PLACEHOLDER_INNER_LOOP_START"
            else:
                inner_addr_offset = "INNER_LOOP_START"
            loop_body += writer.get_instr(
                f"jump to [{inner_addr_offset}] with predicate",
                "bra",
                None,
                ["q", inner_addr_offset],
                ctrl=f"B......|R.|W.|Y1|S5|",
            )

            self.inner_loop_instr = ic.counter - inner_loop_start_instr
        else:
            self.inner_loop_instr = 0
        # === a small loop to hide instruction cache misses ===

        # give load time to write value into memory
        # +++ large chunk of shifts
        loop_body += writer.shift_until_fma()
        for _ in range(self.num_shifts):
            shift = randrange(3, 30)
            loop_body += writer.get_instr(
                f"c += c << {shift}", "shladd", "c", ["c", shift]
            )

            even_shifts = self.state_size + (self.state_size % 2)
            loop_body += writer.shift_sequence(even_shifts)

            shift = randrange(3, 30)
            loop_body += writer.get_instr(
                f"c += c >> {shift}", "shradd", "c", ["c", shift]
            )

            loop_body += writer.shift_sequence(even_shifts)
        # === large chunk of shifts

        # +++ add value from memory into checksum
        loop_body += writer.shift_until_alu()
        loop_body += writer.get_instr(
            f"c = c + mval",
            "add",
            "c",
            ["c", "mval"],
            # wait load to complete
            ctrl=f"B{LD_BARRIER_WRITE}|R.|W.|Y1|S1|",
        )
        # === add value from memory into checksum

        # adversarial instruction
        if Args.v.with_adversarial_nop:
            loop_body += ic.add(f"B......|R.|W.|Y1|S1| NOP;\n", kind)

        # give some time to update checksum value
        loop_body += writer.shift_sequence(6)

        # +++ perform self-modifying shift
        if Args.v.with_self_modification:
            loop_body += writer.shift_until_alu()
            mod_instruction = ic.counter
            loop_body += writer.get_instr(
                "c += c >> (32 - (shift_byte >> 3))", "shraddmod", "c", ["c", "shift_byte"]
            )
        else:
            mod_instruction = ic.counter
        # === perform self-modifying shift

        # +++ jump to the start of the loop or exit
        if kind == "sass":
            addr_offset = "PLACEHOLDER_OUTER_LOOP_START"
        else:
            addr_offset = "OUTER_LOOP_START"
        loop_body += writer.get_instr(
            f"jump to [{addr_offset}] with predicate",
            "bra",
            None,
            ["p", addr_offset],
            ctrl=f"B{LD_BARRIER_READ}|R.|W.|Y1|S5|",
        )
        # === jump to the start of the loop or exit

        self.loop_instr = ic.counter - prefix_instructions

        # fill remainder of block with NOPs
        for _ in range(10):
            loop_body += ic.add(f"B......|R.|W.|Y1|S1| NOP;\n", kind)

        self.loop_size = ic.counter

        code += loop_body

        # load output register (it was stored in R4)
        code += ic.add(
            f"B......|R1|W2|Y1|S4| LDL {as_reg('output')}, [R1 + {stored_regs.index(4) * 4}];\n",
            kind,
        )
        code += ic.add(f"B...21.|R.|W.|Y1|S8| NOP;\n", kind)
        # output checksum and state registers
        #    checksum
        code += ic.add(
            f"B......|R0|W.|Y1|S4| STL [{as_reg('output')}], {as_reg('c')};\n", kind
        )
        #    state
        for i in range(self.state_size):
            rn = as_reg(f"s{i}")
            code += ic.add(
                f"B......|R0|W.|Y1|S6| STL [{as_reg('output')} + {(i + 1) * 4}], {rn};\n",
                kind,
            )
        mem_addr_offset = (num_stored_regs + self.state_size + 1) * 4
        code += ic.add(f"B.....0|R.|W.|Y1|S8| NOP;\n", kind)
        # restore state of registers
        for idx, reg in enumerate(stored_regs):
            code += ic.add(f"B......|R1|W2|Y1|S6| LDL R{reg}, [R1+{idx * 4}];\n", kind)
        code += ic.add(f"B...21.|R.|W.|Y1|S8| NOP;\n", kind)
        # return stack pointer to the initial position
        code += ic.add(
            f"B......|R.|W.|Y1|S8| IADD3 R1, R1, {num_stored_regs * 4}, RZ; // free memory on stack\n",
            kind,
        )
        # return from the function call to the address stored in R20,R21 (according to reverse-engineered calling convention)
        code += ic.add("B......|R.|W.|Y1|S7| RET.ABS.NODEC R20 0x0;\n", kind)
        # infinite loop (jump to the same instruction)
        # I don't know why it is necessary, but ptxas does it
        code += ic.add(
            "B......|R.|W.|Y1|S6| BRA -0x10; // jump to itself (compensate advancing PC by one instruction);\n",
            kind,
        )
        for i in range(8):
            code += ic.add(
                "B......|R.|W.|Y1|S6| NOP;\n", kind
            )  # usually functions have NOPs at the end

        # store result
        if kind == "ptx":
            code += TAB + 'asm volatile("mov.u32 %0, c;" : "=r"(s.c));\n'
            for i in range(self.state_size):
                code += TAB + f'asm volatile("mov.u32 %0, s{i};" : "=r"(s.d[{i}]));\n'
        elif kind == "cpp":
            code += TAB + "s.c = c;\n"
            for i in range(self.state_size):
                code += TAB + f"s.d[{i}] = s{i};\n"

        # output function end

        if kind in ("cpp", "ptx"):
            code += TAB + "return s;\n"
            code += "}\n\n"  # func end

        min_mem_size = 2 ** math.ceil(math.log2(ic.counter * INSTRUCTION_BYTES))
        if self.mem_size is None:
            self.mem_size = min_mem_size
        else:
            if self.mem_size < min_mem_size:
                raise Exception("Specified memory size is too small to hold checksum code") 

        # +++ fill placeholders
        shift_imm_byte_in_instruction = 9
        mod_location = (
            mod_instruction * INSTRUCTION_BYTES + shift_imm_byte_in_instruction
        )
        code = code.replace("PLACEHOLDER_MOD_LOCATION", str(mod_location))

        and_const = 0
        for i in range(2, int(math.log2(self.mem_size))):
            and_const |= 1 << i
        code = code.replace("PLACEHOLDER_LOAD_MASK", str(and_const))

        outer_loop_start_offset = - self.loop_instr * INSTRUCTION_BYTES
        code = code.replace("PLACEHOLDER_OUTER_LOOP_START", str(outer_loop_start_offset))

        inner_loop_start_offset = - self.inner_loop_instr * INSTRUCTION_BYTES
        code = code.replace("PLACEHOLDER_INNER_LOOP_START", str(inner_loop_start_offset))
        # === fill placeholders

        return code


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--with_adversarial_nop', action=argparse.BooleanOptionalAction)
    parser.add_argument('--with_inner_loop', action=argparse.BooleanOptionalAction)
    parser.add_argument('--with_self_modification', action=argparse.BooleanOptionalAction)
    parser.add_argument('--num_iters', type=int, default=100000)
    parser.add_argument('--num_inner_iters', type=int, default=5000)
    parser.add_argument('--num_shifts', type=int, default=4)
    parser.add_argument('--mem_size', type=int, default=2 ** 19)
    Args.v = parser.parse_args()
    args = Args.v
    
    if args.with_self_modification and args.num_shifts < 90:
        raise Exception("Self-modification requires instruction sequence to fill all 8 kB of L2 cache. Increase the number of shifts to achieve that.")

    g = Generator(
        num_iters=args.num_iters,
        num_inner_iters=args.num_inner_iters,
        num_shifts=args.num_shifts,
        mem_size=args.mem_size,
    )

    ptx = g.generate_src("ptx")
    with open("cuda_src_ptx.cu", "w") as f:
        f.write(ptx)
    cpp = g.generate_src("cpp")
    with open("cuda_src_cpp.cpp", "w") as f:
        f.write(cpp)
    sass = g.generate_src("sass")
    with open("cuda_src_sass.sass", "w") as f:
        f.write(sass)

    # write to file
    header = g.generate_header()
    with open(header_name, "w") as f:
        f.write(header)
