## ane.nim — Ergonomic Nim wrapper for Apple Neural Engine
## Wraps the C bridge (ane_bridge.h) into type-safe, memory-managed API.

# --- FFI: link to libane_bridge.dylib ---
const libName = "bridge/libane_bridge.dylib"

type
  ANEKernelHandle = pointer  ## Opaque handle from C

# --- Raw C imports (the "Silicon" layer) ---
proc c_init(): cint
    {.importc: "ane_bridge_init", dynlib: libName.}
proc c_free(k: ANEKernelHandle)
    {.importc: "ane_bridge_free", dynlib: libName.}
proc c_eval(k: ANEKernelHandle): bool
    {.importc: "ane_bridge_eval", dynlib: libName.}
proc c_compile(milText: cstring, milLen: csize_t,
               weightData: pointer, weightLen: csize_t,
               nInputs: cint, inputSizes: ptr csize_t,
               nOutputs: cint, outputSizes: ptr csize_t): ANEKernelHandle
    {.importc: "ane_bridge_compile", dynlib: libName.}
proc c_write_input(k: ANEKernelHandle, idx: cint, data: pointer, bytes: csize_t)
    {.importc: "ane_bridge_write_input", dynlib: libName.}
proc c_read_output(k: ANEKernelHandle, idx: cint, data: pointer, bytes: csize_t)
    {.importc: "ane_bridge_read_output", dynlib: libName.}

# --- High-level "Human" API ---

type
  Kernel* = object
    handle: ANEKernelHandle
    inputBytes*: seq[int]
    outputBytes*: seq[int]

proc close*(k: var Kernel) =
  ## Release ANE resources. Called automatically when Kernel goes out of scope.
  if k.handle != nil:
    c_free(k.handle)
    k.handle = nil

proc `=destroy`*(k: Kernel) =
  if k.handle != nil:
    c_free(k.handle)

proc `=copy`*(dst: var Kernel, src: Kernel) {.error: "Kernel cannot be copied".}

# MIL header matching what the ANE compiler expects (from stories_mil.h)
const MIL_HDR* = """program(1.3)
[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}})]
{
"""

proc compile*(mil: string, inputSizes, outputSizes: openArray[int]): Kernel =
  ## Compile a MIL program into an ANE kernel.
  ## inputSizes/outputSizes are byte sizes for each I/O tensor.
  var inSizes = newSeq[csize_t](inputSizes.len)
  var outSizes = newSeq[csize_t](outputSizes.len)
  for i, s in inputSizes: inSizes[i] = s.csize_t
  for i, s in outputSizes: outSizes[i] = s.csize_t

  let h = c_compile(
    mil.cstring, mil.len.csize_t,
    nil, 0.csize_t,
    inputSizes.len.cint, addr inSizes[0],
    outputSizes.len.cint, addr outSizes[0])

  if h == nil:
    raise newException(ValueError, "ANE compile failed — check MIL syntax")

  result = Kernel(
    handle: h,
    inputBytes: @inputSizes,
    outputBytes: @outputSizes)

proc writeInput*(k: Kernel, idx: int, data: pointer, bytes: int) =
  c_write_input(k.handle, idx.cint, data, bytes.csize_t)

proc readOutput*(k: Kernel, idx: int, data: pointer, bytes: int) =
  c_read_output(k.handle, idx.cint, data, bytes.csize_t)

proc eval*(k: Kernel): bool =
  ## Run the compiled kernel on ANE hardware.
  c_eval(k.handle)

proc run*(k: Kernel, input: seq[uint16]): seq[uint16] =
  ## One-shot: write fp16 input, evaluate, read fp16 output.
  let inBytes = input.len * sizeof(uint16)
  c_write_input(k.handle, 0, unsafeAddr input[0], inBytes.csize_t)

  if not c_eval(k.handle):
    raise newException(ValueError, "ANE eval failed")

  let outElems = k.outputBytes[0] div sizeof(uint16)
  result = newSeq[uint16](outElems)
  c_read_output(k.handle, 0, addr result[0], k.outputBytes[0].csize_t)

# Initialize bridge on import
let initResult = c_init()
if initResult != 0:
  raise newException(OSError, "Failed to initialize ANE bridge (is this Apple Silicon?)")
