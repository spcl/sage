#!/usr/bin/env python

import opcode_discovery as ocd

if __name__ == "__main__":
    checker = ocd.Checker()
    
    # bincode, hexstr, status, disasm = checker.check_instr(ocd.Instruction(0x0fc7fff7ffffff00000000000003d1))
    # print(bincode, hexstr, status, disasm)

    # IADD3_base = 0xfc000000000000000000000000810 # IMM version
    # print(checker.check_instr(ocd.Instruction(IADD3_base)))
    # for i in range(128):
    #     curr = IADD3_base ^ (1 << i)
    #     print(i, checker.check_instr(ocd.Instruction(curr)))

    # print('========= IADD predicate =========')
    # for i in range(16):
    #     print(checker.check_instr(ocd.Instruction(IADD3_base ^ (i << 12))))

    # print('========= IADD reg0 =========')
    # for i in range(256):
    #     print(checker.check_instr(ocd.Instruction(IADD3_base ^ (i << 16))))
    
    # print('========= IADD carry and const =========')
    # IADD3_carry = IADD3_base | (1 << 74)
    # for i in range(128):
    #     curr = IADD3_carry ^ (1 << i)
    #     print(i, checker.check_instr(ocd.Instruction(curr)))

    # print('========= IADD carry and RD =========')
    # IADD3_carry = (IADD3_base | (1 << 74)) ^ (1 << 11)
    # for i in range(128):
    #     curr = IADD3_carry ^ (1 << i)
    #     print(i, checker.check_instr(ocd.Instruction(curr)))

    # IADD3_const = IADD3_base | (1 << 9)
    # for i in range(128):
    #     curr = IADD3_const ^ (1 << i)
    #     print(i, checker.check_instr(ocd.Instruction(curr)))

    # IADD3_reg = (IADD3_base | (1 << 9)) ^ (1 << 11)
    # for i in range(128):
    #     curr = IADD3_reg ^ (1 << i)
    #     print(i, checker.check_instr(ocd.Instruction(curr)))

    # STL_base = 0xfc000000000000000000000000387
    # print(checker.check_instr(ocd.Instruction(STL_base)))
    # for i in range(128):
    #     curr = STL_base ^ (1 << i)
    #     print(i, checker.check_instr(ocd.Instruction(curr)))

    # STL type
    # for i in range(16):
    #     print(checker.check_instr(ocd.Instruction(STL_base | (i << 73))))

    # STL cache eviction policy
    # for i in range(16):
    #     print(checker.check_instr(ocd.Instruction(STL_base | (i << 84))))

    # MOV_base = 0xfc000000000000000000000000202
    # print(checker.check_instr(ocd.Instruction(MOV_base)))
    # for i in range(128):
    #     curr = MOV_base ^ (1 << i)
    #     print(i, checker.check_instr(ocd.Instruction(curr)))

    # MOV_base = 0xfc000000000000000000000000a02
    # print(checker.check_instr(ocd.Instruction(MOV_base)))
    # for i in range(128):
    #     curr = MOV_base ^ (1 << i)
    #     print(i, checker.check_instr(ocd.Instruction(curr)))

    # MOV_base = 0xfc000000000000000000000000802
    # print(checker.check_instr(ocd.Instruction(MOV_base)))
    # for i in range(128):
    #     curr = MOV_base ^ (1 << i)
    #     print(i, checker.check_instr(ocd.Instruction(curr)))

    # MOV_base = 0xfc000080000000000000000000c02
    # print(checker.check_instr(ocd.Instruction(MOV_base)))
    # for i in range(128):
    #     curr = MOV_base ^ (1 << i)
    #     print(i, checker.check_instr(ocd.Instruction(curr)))

    # LDL_base = 0xfc000000000000000000000000983
    # print(checker.check_instr(ocd.Instruction(LDL_base)))
    # for i in range(128):
    #     curr = LDL_base ^ (1 << i)
    #     print(i, checker.check_instr(ocd.Instruction(curr)))

    # LDL_base = 0xfc000000000000000000000000983 
    # print(checker.check_instr(ocd.Instruction(LDL_base)))
    # for i in range(8):
    #     curr = LDL_base ^ (i << 84)
    #     print(i, checker.check_instr(ocd.Instruction(curr)))

    # LDL_base = 0xfc000080000000000000000000983
    # print(checker.check_instr(ocd.Instruction(LDL_base)))
    # for i in range(128):
    #     curr = LDL_base ^ (1 << i)
    #     print(i, checker.check_instr(ocd.Instruction(curr)))

    # IMAD_base = 0xfc000000000000000000000000224
    # print(checker.check_instr(ocd.Instruction(IMAD_base)))
    # for i in range(128):
    #     curr = IMAD_base ^ (1 << i)
    #     print(i, checker.check_instr(ocd.Instruction(curr)))

    # IMAD_base = 0xfc000000000000000000000000824 | (1 << 74)
    # print(checker.check_instr(ocd.Instruction(IMAD_base)))
    # for i in range(128):
    #     curr = IMAD_base ^ (1 << i)
    #     print(i, checker.check_instr(ocd.Instruction(curr)))

    # LOP3_base = 0xfc000000000000000000000000212 | (1 << 80)
    # print(checker.check_instr(ocd.Instruction(LOP3_base)))
    # for i in range(128):
    #     curr = LOP3_base ^ (1 << i)
    #     print(i, checker.check_instr(ocd.Instruction(curr)))

    # LEA_base = 0xfc000000000000000000000000a11 | (1 << 74)
    # print(checker.check_instr(ocd.Instruction(LEA_base)))
    # for i in range(128):
    #     curr = LEA_base ^ (1 << i)
    #     print(i, checker.check_instr(ocd.Instruction(curr)))

    # LD_base = 0xfc000000000000000000000000980
    # print(checker.check_instr(ocd.Instruction(LD_base)))
    # for i in range(128):
    #     curr = LD_base ^ (1 << i)
    #     print(i, checker.check_instr(ocd.Instruction(curr)))

    # LD_base = 0xfc000000000000000000000000980 
    # print(checker.check_instr(ocd.Instruction(LD_base)))
    # for i in range(16):
    #     curr = LD_base ^ (i << 77)
    #     print(i, checker.check_instr(ocd.Instruction(curr)))


    LD_base = 0x000fc000000000000000000000000212
    print(checker.check_instr(ocd.Instruction(LD_base)))
    for i in range(128):
        curr = LD_base ^ (1 << i)
        print(i, checker.check_instr(ocd.Instruction(curr)))

    # for i in range(4):
    #     curr = LD_base ^ (i << 32)
    #     print(i, checker.check_instr(ocd.Instruction(curr)))