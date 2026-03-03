## test_ane.nim — Nim → Apple Neural Engine round-trip test
## Compiles MIL, sends fp16 data through ANE, verifies output.

import bridge/ane

echo "=== ANE Nim Bridge Test ==="

# MIL program: double every element (y = x + x)
# Shape [1, 64, 1, 32] = NCHW with 64 channels, 32 spatial
let mil = MIL_HDR &
  "    func main<ios18>(tensor<fp16, [1, 64, 1, 32]> x) {\n" &
  "        tensor<fp16, [1,64,1,32]> y = add(x=x, y=x)[name=string(\"y\")];\n" &
  "    } -> (y);\n}\n"

let tensorBytes = 64 * 32 * 2  # 64 channels * 32 spatial * 2 bytes (fp16)

echo "Compiling MIL..."
let kernel = compile(mil, [tensorBytes], [tensorBytes])
echo "  Compiled OK"

# Write test pattern: first 3 values are 1.0, 2.0, 3.0 in fp16
# Rest is zero. We'll check the first 3 outputs are 2.0, 4.0, 6.0
var input = newSeq[uint16](64 * 32)
input[0] = 0x3C00'u16  # 1.0
input[1] = 0x4000'u16  # 2.0
input[2] = 0x4200'u16  # 3.0

echo "Running on ANE..."
let output = kernel.run(input)

echo "  Input:    [", input[0], ", ", input[1], ", ", input[2], "]"
echo "  Output:   [", output[0], ", ", output[1], ", ", output[2], "]"
echo "  Expected: [16384, 17408, 17920]"  # 2.0h, 4.0h, 6.0h

let ok = output[0] == 0x4000'u16 and  # 2.0
         output[1] == 0x4400'u16 and  # 4.0
         output[2] == 0x4600'u16      # 6.0

if ok:
  echo "\nPASS: ANE executed correctly through Nim bridge"
else:
  echo "\nFAIL: output mismatch"
  quit(1)
