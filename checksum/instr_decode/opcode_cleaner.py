import opcode_discovery
import json

if __name__ == "__main__":

    with open("discovered_opcodes_raw.json", "r") as f:
        opcodes_raw = json.load(f)

    err = "nvdisasm error   : Unrecognized operation for functional unit 'uC' at address 0x00000000\n"

    opcodes_clean = {
        h: v for i, (h, s, v) in opcodes_raw.items() if v != err and s == True
    }

    with open("discovered_opcodes.json", "w") as f:
        f.write(json.dumps(opcodes_clean, indent=4, sort_keys=True))
