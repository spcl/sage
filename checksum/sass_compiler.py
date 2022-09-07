#!/usr/bin/env python

import sys

from instr_decode.opcode_discovery import Instruction
import re


def isREG(s: str):
    return -1 != s.find('R')

def reg_idx(s: str):
    idx = s[s.find('R')+1:]
    return 255 if idx == 'Z' else int(idx)

def reg_sign(s: str):
    return s[0] == '-' or s[0] == '~'

def pred_idx(s):
    if s is None:
        return 7
    idx = s[s.find('P')+1:]
    return 7 if idx == 'T' else int(idx)

def pred_sign(s):
    if s is None:
        return 0
    return s[0] == '!'

def isPRED(s: str):
    return -1 != s.find('R')

def isIMM(s: str):
    try:
        int(s, 0)
        return True
    except:
        return False

def isCONST(s: str):
    if s.startswith('-'):
        s = s[1:]
    return s[0] == 'c'

def two_complement(number, bits):
    max_num = 2 ** (bits - 1)
    if not ((-max_num <= number) and (number < max_num)):
        raise Exception("Number doesn't fit in the specified bits")
    if number >= 0:
        return number
    else:
        return 2 ** bits + number

def toIADD3(instr, suff_str, args_str):
    args_list = [s.strip() for s in args_str.split(',')]

    regs = []

    special = None

    for arg in args_list:
        if isREG(arg[0]):
            regs.append(arg)
        elif isPRED(arg[0]):
            raise Exception('Not implemented')
        else:
            special = arg

    if special is None:
        assert len(regs) == 4
        special = regs.pop(2)
    else:
        assert len(regs) == 3
        
    if isREG(special):
        instr[0:12] = 0x210

        instr[32:40] = reg_idx(special)
        instr[63] = reg_sign(special)
    elif isIMM(special):
        opcode = 0x810
        instr[0:12] = opcode

        instr[32:64] = two_complement(int(special, 0), 32)
    elif isCONST(special):
        raise Exception('Not implemented')
    else:
        raise Exception(f"Unknown argument type of {special}")

    instr[16:24] = reg_idx(regs[0])
    instr[24:32] = reg_idx(regs[1])
    instr[64:72] = reg_idx(regs[2])
    instr[75] = reg_sign(regs[2])

    instr[81:84] = pred_idx('PT') # PB
    instr[84:87] = pred_idx('PT') # PC

    return instr

def toSTL(instr, suff_str, args_str):
    m = re.match('\[(R[^\s+\]]+)\s*(\+\s*([^\s]+)\s*)?\],\s+(R[^\s]+)', args_str)

    RA = m.group(1)
    IMM = m.group(3)
    RB = m.group(4)

    if IMM is None: IMM = '0'

    instr[0:12] = 0x387 # opcode

    instr[24:32] = reg_idx(RA)
    instr[32:40] = reg_idx(RB)
    instr[40:64] = two_complement(int(IMM, 0), 64-40)

    if suff_str is None:
        raise Exception('Not implemented')
    instr[73:76] = 4 # U8,S8,U16,S16,32(empty),64,128,INVALID7
    instr[84:87] = 1 # EF,(empty),EL,LU,EU,NA,INVALID6,INVALID7

    return instr

def toLDL(instr, suff_str, args_str):
    m = re.match('(R[^\s]+),\s+\[(R[^\s]+)\s*\+\s*([^\s]+)\s*\]', args_str)

    RA = m.group(1)
    RB = m.group(2)
    IMM = m.group(3)
    
    instr[0:12] = 0x983 # opcode
    instr[16:24] = reg_idx(RA)
    instr[24:32] = reg_idx(RB)
    instr[40:64] = two_complement(int(IMM, 0), 64 - 40)

    if suff_str is None:
        raise Exception('Not implemented')
    instr[73:76] = 4 # U8,S8,U16,S16,32(empty),64,128,INVALID7
    instr[84:87] = 1 # EF,(empty),EL,LU,EU,NA,INVALID6,INVALID7

    return instr

def toLD(instr, suff_str, args_str):

    evs = ['EF','','EL','LU','EU','NA']
    ltcs = ['', 'LTC64B', 'LTC128B']
    types = ['U8', 'S8', 'U16', 'S16', '', '64', '128']
    params = ['CONSTANT', '', 'STRONG', 'MMIO']
    scopes = ['CTA', 'SM', 'GPU', 'SYS']

    pE = "(\.(E))?"
    pEV = f"(\.({'|'.join(evs)}))?"
    pLTC = f"(\.({'|'.join(ltcs)}))?"
    pTYPE = f"(\.({'|'.join(types)}))?"
    pPARAM = f"(\.({'|'.join(params)}))?"
    pSCOPE = f"\.({'|'.join(scopes)})"

    m = re.match(f"{pE}{pEV}{pLTC}{pTYPE}{pPARAM}{pSCOPE}", suff_str)

    E = m.group(2)
    EV = m.group(4) 
    LTC = m.group(6)
    TYPE = m.group(8)
    PARAM = m.group(10)
    SCOPE = m.group(11)

    instr[72] = E == 'E'
    instr[84:87] = evs.index(EV or '')
    instr[68:70] = ltcs.index(LTC or '')
    instr[73:76] = types.index(TYPE or '')
    instr[79:81] = params.index(PARAM or '')
    instr[77:79] = scopes.index(SCOPE or '')
    
    pRA = "(R[^,]+)"
    pRB = "(R[^\s+]+)"
    pIMM = "([^\s\]]+)"
    pPB = "(,\s*([^\s]+))?"

    m = re.match(f"{pRA},\s*\[\s*{pRB}\s*(\+\s*{pIMM}\s*)?\]\s*{pPB}", args_str)

    RA = m.group(1)
    RB = m.group(2)
    IMM = m.group(4) or '0'
    PB = m.group(6)

    instr[0:12] = 0x980

    instr[16:24] = reg_idx(RA)
    instr[24:32] = reg_idx(RB)
    instr[32:64] = two_complement(int(IMM, 0), 64 - 32)

    instr[64:67] = 7 - pred_idx(PB) # inverted
    instr[67] = pred_sign(PB)

    return instr


def toMOV(instr, suff_str, args_str):
    args_list = [s.strip() for s in args_str.split(',')]

    if len(args_list) != 2:
        raise Exception('Not implemented')

    RA = args_list[0]
    special = args_list[1]

    instr[16:24] = reg_idx(RA)

    if isIMM(special):
        instr[0:12] = 0x802 # opcode
        instr[32:64] = int(special, 0)
    else:
        raise Exception('Not implemented')
    
    return instr

def toIMAD(instr, suff_str, args_str):
    args_list = [s.strip() for s in args_str.split(',')]

    if len(args_list) != 4:
        raise Exception("Not implemented")

    instr[16:24] = reg_idx(args_list[0]) # RA
    instr[24:32] = reg_idx(args_list[1]) # RB

    instr[81:84] = pred_idx('PT') # PC
    
    if -1 != suff_str.find('HI'):
        instr[0:4] = 7
    elif -1 != suff_str.find('WIDE'):
        instr[0:4] = 5
    else:
        instr[0:4] = 4

    if isREG(args_list[2]):
        if not isIMM(args_list[3]):
            raise Exception("Not implemented")
        instr[4:12] = 0x42

        RD = args_list[2]
        
        instr[32:64] = two_complement(int(args_list[3], 0), 64 - 32) # CONST
    elif isIMM(args_list[2]):
        if not isREG(args_list[3]):
            raise Exception("Not implemented")
        instr[4:12] = 0x82

        instr[32:64] = two_complement(int(args_list[2], 0), 64 - 32) # CONST

        RD = args_list[3]
    else:
        raise Exception("Not implemented")

    if -1 != suff_str.find('HI'):
        if reg_idx(RD) % 2 != 0:
            raise Exception("Last argument of IMAD.HI have to contain even register!")

    instr[64:72] = reg_idx(RD)

    signed = -1 == suff_str.find('U32')
    instr[73] = int(signed)

    return instr

def toLOP3(instr, suff_str, args_str):
    args_list = [s.strip() for s in args_str.split(',')]

    if len(args_list) != 6:
        raise Exception("Not implemented")

    instr[16:24] = reg_idx(args_list[0]) # RA
    instr[24:32] = reg_idx(args_list[1]) # RB

    instr[81:84] = pred_idx('PT') # PB

    if isREG(args_list[2]):
        instr[0:12] = 0x212
        instr[32:40] = reg_idx(args_list[2])
    elif isIMM(args_list[2]):
        instr[0:12] = 0x812
        instr[32:64] = two_complement(int(args_list[2], 0), 64 - 32)
    else:
        raise Exception("Not implemented")

    instr[64:72] = reg_idx(args_list[3]) # RD

    instr[72:80] = int(args_list[4], 0) # UIMM, const operation

    PC = args_list[5]

    instr[87:90] = pred_idx(PC)
    instr[90] = pred_sign(PC)

    return instr

def toLEA(instr, suff_str, args_str):
    if suff_str is None:
        suff_str = ''

    HI = -1 != suff_str.find('.HI')
    X = -1 != suff_str.find('.X')
    SX32 = -1 != suff_str.find('.SX32')

    pRA = '(R[^,]+)'
    pPB = '(,\s+(P[0-6]))?'
    pRB = '(,\s+(R[^,]+))'
    pSC = '(,\s+([^,]+))'
    pRD = '(,\s+(R[^,]+))?'
    pUIMM2 = '(,\s+([^,\s]+))'
    pPC = '(,\s+(P[0-6]))?'

    m = re.match(f'{pRA}{pPB}{pRB}{pSC}{pRD}{pUIMM2}{pPC}', args_str)

    RA = m.group(1)
    PB = m.group(3)
    RB = m.group(5)
    SC = m.group(7)
    RD = m.group(9)
    UIMM2 = m.group(11)
    PC = m.group(13)

    instr[16:24] = reg_idx(RA)
    instr[81:84] = pred_idx(PB)
    instr[24:32] = reg_idx(RB)

    if isREG(SC):
        instr[0:12] = 0x211
        instr[32:40] = reg_idx(SC)
    elif isIMM(SC):
        instr[0:12] = 0x811
        instr[32:64] = int(SC, 0)
    else:
        raise Exception("Not implemented")

    assert bool(RD) == bool(HI) 
    if RD:
        instr[80] = 1
        instr[64:72] = reg_idx(RD)
    else:
        instr[80] = 0
        instr[64:72] = reg_idx('RZ')
    
    instr[75:80] = int(UIMM2, 0)

    assert bool(X) == bool(PC)
    if PC: 
        instr[74] = 1
        instr[87:90] = pred_idx(PC)
    else:
        instr[74] = 0
        instr[87:90] = pred_idx('PT')
    
    if SX32:
        assert HI
    instr[73] = SX32

    return instr


def toISETP(instr, suff_str, args_str):
    pCMP = '\.([^.]+)'
    pSIGN = '(\.U32)?'
    pBOOL = '\.([^.\s]+)'
    pEX = '(\.EX)?'

    m = re.match(f'{pCMP}{pSIGN}{pBOOL}{pEX}', suff_str)
    
    CMP = m.group(1)
    SIGN = m.group(2)
    BOOL = m.group(3)
    EX = m.group(4)

    instr[76:79] = ['F','LT','EQ','LE','GT','NE','GE','T'].index(CMP)
    instr[73] = SIGN is None 
    instr[74:76] = ['AND', 'OR', 'XOR'].index(BOOL)
    instr[72] = EX is not None

    pPB = '(P[^,]+)'
    pPC = ',\s+(P[0-6T])'
    pRA = ',\s+(R[^,]+)'
    pSB = ',\s+([^,]+)'
    pPD = ',\s+(!?P[0-6T])'
    pPE = '(,\s+(!?P[0-6T]))?'

    m = re.match(f"{pPB}{pPC}{pRA}{pSB}{pPD}{pPE}", args_str)

    PB = m.group(1)
    PC = m.group(2)
    RA = m.group(3)
    SB = m.group(4)
    PD = m.group(5)
    PE = m.group(7)

    instr[81:84] = pred_idx(PB)
    instr[84:87] = pred_idx(PC)

    instr[24:32] = reg_idx(RA)

    if isREG(SB):
        instr[0:12] = 0x20c
        instr[32:40] = reg_idx(SB)
    elif isIMM(SB):
        instr[0:12] = 0x80c
        instr[32:64] = two_complement(int(SB, 0), 64 - 32)
    else:
        raise Exception("Not implemented")

    instr[87:90] = pred_idx(PD)
    instr[90] = pred_sign(PD)
    instr[68:71] = pred_idx(PE)
    instr[71] = pred_sign(PE)

    return instr

def toSEL(instr, suff_str, args_str):
    args_list = [s.strip() for s in args_str.split(',')]

    RA = args_list[0]
    RB = args_list[1]
    SC = args_list[2]
    PB = args_list[3]

    instr[16:24] = reg_idx(RA)
    instr[24:32] = reg_idx(RB)
    instr[87:90] = pred_idx(PB)
    instr[90] = pred_sign(PB)

    if isREG(SC):
        instr[0:12] = 0x207
        instr[32:40] = reg_idx(SC)
    elif isIMM(SC):
        instr[0:12] = 0x807
        instr[32:64] = int(SC, 0)
    else:
        raise Exception("Not implemented")

    return instr


def toBRX(instr, suff_str, args_str):
    pred_offset = re.split('\s*,\s*', args_str)

    suff_str = suff_str or ''
    if suff_str != '':
        raise Exception('Not implemented')

    if len(pred_offset) == 1:
        PB = 'PT'
        offset = pred_offset[0]
    elif len(pred_offset) == 2:
        PB = pred_offset[0]
        offset = pred_offset[1]
    else:
        raise Exception("Incorrect argument string format")

    reg_imm = offset.split()

    if len(reg_imm) == 1:
        RA = reg_imm[0]
        IMM = '0'
    elif len(reg_imm) == 2:
        RA = reg_imm[0]
        IMM = reg_imm[1]
    else:
        raise Exception("Incorrect argument string format")

    instr[0:12] = 0x949
    instr[87:90] = pred_idx(PB)
    instr[90] = pred_sign(PB)
    instr[24:32] = reg_idx(RA)

    IMM_VAL = int(IMM, 0)
    assert IMM_VAL % 4 == 0

    instr[34:82] = IMM_VAL // 4

    return instr

def toNOP(instr, suff_str, args_str):
    instr[0:12] = 0x918
    return instr

def toRET(instr, suff_str, args_str):

    if suff_str != '.ABS.NODEC':
        raise Exception('Not implemented')

    if args_str != 'R20 0x0':
        raise Exception('Not implemented')

    instr[0:12] = 0x950 # opcode
    instr[85] = 1 # ABS
    instr[86] = 1 # NODEC
    instr[87:90] = pred_idx('PT') # PB
    instr[90] = pred_sign('PT') # !PB
    instr[24:32] = reg_idx('R20') # RA
    return instr

def toBRA(instr, suff_str, args_str):
    if suff_str != '':
        raise Exception('Not implemented')

    required_offset = int(args_str, 0)
    assert required_offset % 4 == 0
    sass_offset = required_offset // 4

    instr[0:12] = 0x947
    instr[34:82] = two_complement(sass_offset, 82 - 34)
    instr[87:90] = pred_idx('PT')

    return instr

def toST(instr, suff_str, args_str):
    evs = ['EF','','EL','LU','EU','NA']
    types = ['U8', 'S8', 'U16', 'S16', '', '64', '128']
    params = ['CONSTANT', '', 'STRONG', 'MMIO']
    scopes = ['CTA', 'SM', 'GPU', 'SYS']

    pE = "(\.(E))?"
    pEV = f"(\.({'|'.join(evs)}))?"
    pTYPE = f"(\.({'|'.join(types)}))?"
    pPARAM = f"(\.({'|'.join(params)}))?"
    pSCOPE = f"\.({'|'.join(scopes)})"

    m = re.match(f"{pE}{pEV}{pTYPE}{pPARAM}{pSCOPE}", suff_str)

    E = m.group(2)
    EV = m.group(4) 
    TYPE = m.group(6)
    PARAM = m.group(8)
    SCOPE = m.group(9)

    instr[72] = E == 'E'
    instr[84:87] = evs.index(EV or '')
    instr[73:76] = types.index(TYPE or '')
    instr[79:81] = params.index(PARAM or '')
    instr[77:79] = scopes.index(SCOPE or '')

    pRA = "(R[^\s+\]]+)"
    pRB = "(R[^\s]+)"
    pIMM = "([^\s\]]+)"

    m = re.match(f"\[\s*{pRA}\s*(\+\s*{pIMM}\s*)?\],\s*{pRB}", args_str)

    RA = m.group(1)
    IMM = m.group(3) or '0'
    RB = m.group(4)

    instr[0:12] = 0x385

    instr[24:32] = reg_idx(RA)
    instr[32:64] = two_complement(int(IMM, 0), 64 - 32)
    instr[64:72] = reg_idx(RB)

    return instr

def sass_encode(line):
    ctrl = "B([\.5][\.4][\.3][\.2][\.1][\.0])\|R([\.0-6])\|W([\.0-6])\|Y([\.1])\|S([\.1-9a-f])\|"
    pred = "@!?P."
    op = "([^\.\s\;]+)"
    suff = "((\.[^\.\s]+)*)"
    args = "([^;]*)"
    m = re.match(f'^\s*{ctrl}(\s+({pred}))?\s+{op}{suff}(\s+{args})?;.*$', line)
    
    if m is None:
        raise Exception(f"Can't parse string: {line}")

    barrier_str = m.group(1)
    read_str = m.group(2)
    write_str = m.group(3)
    yield_str = m.group(4)
    stall_str = m.group(5)
    pred_str = m.group(7)
    op_str = m.group(8)
    suff_str = m.group(9)
    args_str = m.group(12)


    instr = Instruction(0)

    barrier_val = 0
    for i, c in enumerate(reversed(barrier_str)):
        if c != '.':
            barrier_val |= (1 << i)
    instr[116:122] = barrier_val

    read_val = 7 if read_str == '.' else int(read_str)
    instr[113:116] = read_val

    write_val = 7 if write_str == '.' else int(write_str)
    instr[110:113] = write_val

    yield_val = 0 if yield_str == '.' else int(yield_str)
    instr[109] = yield_val

    stall_val = int(stall_str, 16)
    instr[105:109] = stall_val

    if pred_str is None:
        pred_val = pred_idx('PT')
        pred_neg = pred_sign('PT')
    else:
        assert pred_str[0] == '@'
        pred_val = pred_idx(pred_str[1:])
        pred_neg = pred_sign(pred_str[1:])
    instr[12:15] = pred_val
    instr[15] = pred_neg

    if op_str == 'IADD3':
        instr = toIADD3(instr, suff_str, args_str)
    elif op_str == 'STL':
        instr = toSTL(instr, suff_str, args_str)
    elif op_str == 'MOV':
        instr = toMOV(instr, suff_str, args_str)
    elif op_str == 'LDL':
        instr = toLDL(instr, suff_str, args_str)
    elif op_str == 'IMAD':
        instr = toIMAD(instr, suff_str, args_str)
    elif op_str == 'LOP3':
        instr = toLOP3(instr, suff_str, args_str)
    elif op_str == 'LEA':
        instr = toLEA(instr, suff_str, args_str)
    elif op_str == 'ISETP':
        instr = toISETP(instr, suff_str, args_str)
    elif op_str == 'LD':
        instr = toLD(instr, suff_str, args_str)
    elif op_str == 'SEL':
        instr = toSEL(instr, suff_str, args_str)
    elif op_str == 'BRX':
        instr = toBRX(instr, suff_str, args_str)
    elif op_str == 'NOP':
        instr = toNOP(instr, suff_str, args_str)
    elif op_str == 'RET':
        instr = toRET(instr, suff_str, args_str)
    elif op_str == 'BRA':
        instr = toBRA(instr, suff_str, args_str)
    elif op_str == 'ST':
        instr = toST(instr, suff_str, args_str)
    else:
        raise Exception(f"Unknown instruction {op_str}")
    
    return instr.v.to_bytes(length=16, byteorder='little')

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} input.sass output.bin")

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    i = Instruction.NOP()

    with open(input_file, "r") as inp:
        with open(output_file, "wb") as out:
            for line in inp:
                out.write(sass_encode(line))

    print('OK')



