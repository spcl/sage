#!/usr/bin/env python

#
# This script adds control information in human readable form to the output of nvdisasm and cuobjdump.
# It expects specific format of output, so it may not work for old/future versions.
# (Tested with CUDA 11.3).
# Supports Volta, Turing, Ampere SASS encodings.
# Usage:
#   cuobjdump ./checksum -sass | python ./sass_ctrl.py
#   nvdisasm checksum.cubin -hex | python ./sass_ctrl.py
#
# Hint:
#   To extract cubin from binary it is possible to use:
#   mv $(cuobjdump ./checksum -xelf all | awk '{ print $5; }') checksum.cubin
#
# For the meaning of control information refer to:
#   * https://arxiv.org/pdf/1804.06826.pdf Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking
#   * https://github.com/NervanaSystems/maxas/wiki/Control-Codes (for Maxwell, but still relevant)
#

import sys
import re

if __name__ == '__main__':
    # matches the instruction address, the instruction in human readable format and the first 64 bit of the instruction
    pattern1 = re.compile(r"^ */\*[0-9a-f]*\*/ *([^ ]*[^;]*;) */\* *0x[0-9a-f]* \*/")
    # matches the second 64 bit of the instruction
    pattern2 = re.compile(r"^ */\* (0x[0-9a-f]*) \*/")

    for line in sys.stdin:
        line = line
        match1 = pattern1.match(line)
        match2 = pattern2.match(line)
        if match1:
            start_idx = line.index(match1.group(1))
        if match2:
            end_idx = line.index('/')
            # get the second 64 bit of the instruction in hex
            val = int(match2.group(1), 16)
            # extract the relevant bits
            stall_cycles = (val >> 41) & 15
            yield_flag = (val >> 45) & 1
            write_barrier = (val >> 46) & 7
            read_barrier = (val >> 49) & 7
            barrier_mask = (val >> 52) & 63

            read_barrier_str = ''
            if read_barrier != 7:
                read_barrier_str = f'Read={read_barrier} '

            write_barrier_str = ''
            if write_barrier != 7:
                write_barrier_str = f'Write={write_barrier} '

            barrier_mask_str = ''
            for i in range(6):
                barrier_bit = (barrier_mask >> i) & 1
                if barrier_bit:
                    barrier_mask_str += f' {i}'
            if barrier_mask_str:
                barrier_mask_str = 'Wait=[' + barrier_mask_str + ' ]'

            my_output = f"Stall={stall_cycles} Yield={yield_flag} {read_barrier_str}{write_barrier_str}{barrier_mask_str}" 
            replace_size = end_idx - start_idx
            padded_output = my_output + ' ' * (replace_size - len(my_output))
            sys.stdout.write(line[:start_idx] + padded_output + line[end_idx:])
        else:
            sys.stdout.write(line)

