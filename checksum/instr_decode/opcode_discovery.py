# /usr/bin/env python

import subprocess
import sys
import re
import struct
import os
import multiprocessing
import json


SM = 75

# according to the https://arxiv.org/pdf/1804.06826.pdf
# there up to 13 LSB used to encode instruction opcode
# according to the https://github.com/cloudcores/CuAssembler/blob/master/CuAsm/InsAsmRepos/DefaultInsAsmRepos.sm_75.txt#L14898
# uniform datapath instrucitons contain one opcode bit at position 91
OPCODE_BITS = [*range(0, 13), 91]


INSTR_SIZE = 128
INSTR_SIZE_BYTES = INSTR_SIZE // 8
INSTR_MAX = 2 ** INSTR_SIZE
INSTR_LAST = INSTR_MAX - 1

ALWAYS_ZERO_HI_MSB = 2

REUSE_BITS_SIZE = 4
REUSE_BITS_OFFSET = INSTR_SIZE - REUSE_BITS_SIZE - ALWAYS_ZERO_HI_MSB

BARRIER_BITS_SIZE = 6
BARRIER_BITS_OFFSET = REUSE_BITS_OFFSET - BARRIER_BITS_SIZE

READ_BITS_SIZE = 3
READ_BITS_OFFSET = BARRIER_BITS_OFFSET - READ_BITS_SIZE

WRITE_BITS_SIZE = 3
WRITE_BITS_OFFSET = READ_BITS_OFFSET - WRITE_BITS_SIZE

YIELD_FLAG_OFFSET = WRITE_BITS_OFFSET - 1

STALL_BITS_SIZE = 4
STALL_BITS_OFFSET = YIELD_FLAG_OFFSET - STALL_BITS_SIZE


class Instruction:
    def __init__(self, value):
        assert 0 <= value and value < INSTR_MAX
        self.v = value

    @staticmethod
    def NOP():
        instr = Instruction(0)
        # read and write unused only if all their bits set, this is default state
        instr.read = 0b111
        instr.write = 0b111 
        return instr

    def clear_bit(self, idx):
        assert 0 <= idx and idx < INSTR_SIZE
        self.v &= INSTR_LAST - (1 << idx)

    def set_bit(self, idx, enable=True):
        assert 0 <= idx and idx < INSTR_SIZE
        self.clear_bit(idx)
        self.v |= int(enable) << idx

    def get_bit(self, idx):
        assert 0 <= idx and idx < INSTR_SIZE
        return (self.v >> idx) & 1

    def set_range(self, start, end, value):
        assert 0 <= start and start < end and end < INSTR_SIZE
        for idx, bit in enumerate(range(start, end)):
            self.set_bit(bit, (value >> idx) & 1)

    def get_range(self, start, end):
        assert 0 <= start and start < end and end < INSTR_SIZE
        return (self.v & (2 ** end - 1)) >> start

    @property
    def reuse(self):
        return [self.get_bit(REUSE_BITS_OFFSET + i) for i in range(REUSE_BITS_SIZE)]

    def set_reuse(self, index, enable=True):
        assert 0 <= index and index < REUSE_BITS_SIZE
        self.set_bit(REUSE_BITS_OFFSET + index, enable)

    @property
    def barrier(self):
        return [
            self.get_bit(BARRIER_BITS_OFFSET + i) for i in range(BARRIER_BITS_SIZE)
        ]

    def set_barrier(self, index, enable=True):
        assert 0 <= index and index < BARRIER_BITS_SIZE
        self.set_bit(BARRIER_BITS_OFFSET + index, enable)

    @property
    def read(self):
        return self.get_range(READ_BITS_OFFSET, READ_BITS_OFFSET + READ_BITS_SIZE)

    @read.setter
    def read(self, value):
        assert 0 <= value and value < (2 ** READ_BITS_SIZE)
        self.set_range(READ_BITS_OFFSET, READ_BITS_OFFSET + READ_BITS_SIZE, False)
        self.v |= value << READ_BITS_OFFSET

    @property
    def write(self):
        return self.get_range(WRITE_BITS_OFFSET, WRITE_BITS_OFFSET + WRITE_BITS_SIZE)

    @write.setter
    def write(self, value):
        assert 0 <= value and value < (2 ** WRITE_BITS_SIZE)
        self.set_range(WRITE_BITS_OFFSET, WRITE_BITS_OFFSET + WRITE_BITS_SIZE, False)
        self.v |= value << WRITE_BITS_OFFSET

    @property
    def ctrl_yield(self):
        return self.get_bit(YIELD_FLAG_OFFSET)

    @ctrl_yield.setter
    def ctrl_yield(self, enable):
        self.set_bit(YIELD_FLAG_OFFSET, enable)

    @property
    def stall(self):
        return self.get_range(STALL_BITS_OFFSET, STALL_BITS_OFFSET + STALL_BITS_SIZE)

    @stall.setter
    def stall(self, value):
        assert 0 <= value and value < (2 ** STALL_BITS_SIZE)
        self.set_range(STALL_BITS_OFFSET, STALL_BITS_OFFSET + STALL_BITS_SIZE, False)
        self.v |= value << STALL_BITS_OFFSET

    def __repr__(self):
        return (
            f"Instruction(lo=0x{self.lo:016x}, hi=0x{self.hi:016x}, reuse={self.reuse}, "
            f"barrier={self.barrier}, read={self.read}, write={self.write}, yield={self.ctrl_yield}, "
            f"opcode=0b{self.opcode:014b})"
        )

    @property
    def opcode_bits(self):
        return {b: self.get_bit(b) for b in OPCODE_BITS}

    @opcode_bits.setter
    def opcode_bits(self, value):
        assert len(value) == len(OPCODE_BITS)
        for b, v in zip(OPCODE_BITS, value):
            self.set_bit(b, v)

    def __getitem__(self, key):
        if isinstance(key, slice):
            assert key.step is None
            return self.get_range(key.start, key.stop)
        else:
            return self.get_bit(key)

    def __setitem__(self, key, val):
        if isinstance(key, slice):
            assert key.step is None
            self.set_range(key.start, key.stop, val)
        else:
            self.set_bit(key, val)


def make_dummy_cubin():
    with open("dummy.cu", "w") as f:
        f.write("__device__ void foo() {}")

    subprocess.run(
        f"nvcc -rdc=true -cubin -arch=sm_{SM} dummy.cu -o dummy.cubin",
        shell=True,
        check=True,
    )

    sp = subprocess.run(
        "objdump -h dummy.cubin", shell=True, check=True, capture_output=True
    )
    sections = sp.stdout.decode(sys.stdout.encoding)

    match = re.search(".text._Z3foov\s*([^\s]*)\s*[^\s]*\s*[^\s]*\s*([^\s]*)", sections)
    section_size = match.group(1)
    section_offset = match.group(2)
    # print(section_size, section_offset)

    # target instruction should be at the beginning of offset
    with open("dummy.cubin", "rb") as f:
        original_binary = f.read()

    off_start = int(section_offset, 16)

    return off_start, original_binary


class Checker:
    def __init__(self):
        start, cubin = make_dummy_cubin()
        self.start = start
        self.cubin = cubin
        
        # to avoid calls to cubojdump/nvdisasm that we already tried
        self.not_saved = 0
        try:
            with open('checker_cache.json', 'r') as f:
                self.cache = json.load(f)
        except:
            self.cache = dict()

    def check_instr(self, instr):
        instr_code_str = str(instr.v)
        if instr_code_str in self.cache:
            return self.cache[instr_code_str]

        start = self.start
        cubin = self.cubin

        patched = bytearray(cubin)
        filename = f"isntr_{instr.v}.cubin" 
        with open(filename, "wb") as f:
            before = patched[start:start + INSTR_SIZE_BYTES]
            patched[start:start + INSTR_SIZE_BYTES] = instr.v.to_bytes(INSTR_SIZE_BYTES, "little")
            f.write(patched)

        result = None
        try:
            sp = subprocess.run(
                f"cuobjdump -sass {filename}", shell=True, check=True, capture_output=True
            )
            disasm = sp.stdout.decode(sys.stdout.encoding)
            match = re.search('/.0000./\s*([^;]*);', disasm)
            if match is not None:
                result = (instr.v, hex(instr.v), True, match.group(1))
            else:
                result = (instr.v, hex(instr.v), False, "UNKNOWN")
        except subprocess.CalledProcessError as e:
            result = (instr.v, hex(instr.v), False, e.stderr.decode(sys.stdout.encoding))
        os.remove(filename)

        self.cache[instr_code_str] = result
        self.not_saved += 1
        if self.not_saved == 100:
            # dump cache to disk
            print('+++ Saving checker cache to disk')
            with open('checker_cache.json', 'w') as f:
                json.dump(self.cache, f)
            print('=== Saving checker cache to disk')
            self.not_saved = 0

        return result

    def check_opcode(self, opcode):
        instr = Instruction.NOP()
        instr.opcode_bits = [((opcode >> i) & 1) for i in range(len(OPCODE_BITS))]
        result = self.check_instr(instr)
        return result

if __name__ == "__main__":
    checker = Checker()
    
    #status, i = checker.check_instr(Instruction(0x000078e00ff60000000ff057424), 'patched.cubin')
    #print(status, i)

    parallel_workers = 128
    pool = multiprocessing.Pool(parallel_workers)
    final_result = pool.map(checker.check_opcode, range(0, 2 ** len(OPCODE_BITS)))

    # seqiential equivalent of the above code
    # final_result = []
    # for i in range(0, 2 ** len(OPCODE_BITS)):
    #     final_result.append(checker.check_opcode(i))

    descovered_opcodes = {i: (h, s, r) for i, h, s, r in final_result}
    with open('discovered_opcodes_raw.json', "w") as f:
        f.write(json.dumps(descovered_opcodes, indent=4, sort_keys=True))